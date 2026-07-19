#!/usr/bin/env python
"""

Usage:
    # Conventional (random split), no CV:
    python main.py --data-path /path/to/output_dir --model mlp --epochs 10

    # K-fold cross-validation only:
    python main.py --data-path /path/to/output_dir --model mlp --k-folds 5 --cv-only

    # K-fold CV + final training + test evaluation:
    python main.py --data-path /path/to/output_dir --model mlp --k-folds 5 --patience 10

    # Cross-dataset generalization with CV:
    python main.py --data-path /path/to/output_dir --model mlp \\
        --training-method cross --train-dataset zipf --test-dataset emoji --k-folds 5
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
    DATASET_TYPES,
)
from attacker_detector.models import get_model
from attacker_detector.data import (
    load_npy_dataset,
    prepare_npy_data,
    prepare_npy_data_by_dataset_type,
)
from attacker_detector.training import Trainer
from attacker_detector.training.trainer import run_k_fold_cv
from attacker_detector.analysis import run_sensitivity_analysis, plot_sensitivity_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate attacker detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to dataset'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['mlp', 'gan', 'attention'],
        help='Model type to use'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DEFAULT_DROPOUT,
        help='Dropout rate'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_TEST_SIZE,
        help='Test set fraction'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Validation set fraction --> From train set'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save model and plots'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip sensitivity plots'
    )

    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for k-fold CV'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--cv-only',
        action='store_true',
        default=False,
        help='Run only k-fold CV; skip final training and test evaluation'
    )

    parser.add_argument(
        '--training-method',
        type=str,
        default='none',
        choices=['none', 'cross', 'three-way'],
        help=(
            'Training method: '
            'none = conventional, '
            'cross = train on one dataset_type and test on another, '
            'three-way = train on one, test on another, evaluate on a third'
        )
    )
    parser.add_argument(
        '--train-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for training (required for cross / three-way)'
    )
    parser.add_argument(
        '--test-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for testing (required for cross / three-way)'
    )
    parser.add_argument(
        '--eval-dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type used for evaluation (required for three-way)'
    )

    args = parser.parse_args()

    if args.training_method in ('cross', 'three-way'):
        if not args.train_dataset or not args.test_dataset:
            parser.error(
                f"--training-method={args.training_method} requires "
                "both --train-dataset and --test-dataset"
            )
        if args.train_dataset == args.test_dataset:
            parser.error("--train-dataset and --test-dataset must be different")

    if args.training_method == 'three-way':
        if not args.eval_dataset:
            parser.error("--training-method=three-way requires --eval-dataset")
        if args.eval_dataset in (args.train_dataset, args.test_dataset):
            parser.error(
                "--eval-dataset must differ from --train-dataset and --test-dataset"
            )

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nLoading dataset from: {args.data_path}")
    ds = load_npy_dataset(args.data_path)

    n_features = ds.features.shape[1]
    print(f"Feature count: {n_features}")
    print(f"Feature names: {ds.feature_names}")
    print(f"Training method: {args.training_method}")

    # Hold out set
    has_test_set = True 

    use_val = args.val_size > 0

    if args.training_method == 'none':
        print("\nPreparing data...")
        split = prepare_npy_data(
            ds,
            test_size=args.test_size,
            val_size=args.val_size if use_val else 0.0,
            random_state=args.seed,
        )
    else:
        eval_type = args.eval_dataset if args.training_method == 'three-way' else None
        print(
            f"\nPreparing data (train={args.train_dataset}, "
            f"test={args.test_dataset}"
            + (f", eval={eval_type}" if eval_type else "")
            + ")..."
        )
        split = prepare_npy_data_by_dataset_type(
            ds,
            train_type=args.train_dataset,
            test_type=args.test_dataset,
            eval_type=eval_type,
            val_size=args.val_size if use_val else 0.0,
            random_state=args.seed,
        )

    X_train = split['X_train']
    y_train = split['y_train']

    if args.k_folds > 0:
        if 'X_trainval' in split:
            X_trainval = split['X_trainval']
            y_trainval = split['y_trainval']
        else:
            X_trainval = X_train
            y_trainval = y_train

        cv_results = run_k_fold_cv(
            model_type=args.model,
            input_dim=n_features,
            dropout_rate=args.dropout,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            n_folds=args.k_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            device=device,
            seed=args.seed,
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            cv_df = pd.DataFrame(cv_results['fold_results'])
            cv_path = os.path.join(args.output_dir, 'cv_results.csv')
            cv_df.to_csv(cv_path, index=False)
            print(f"\nCV results saved to: {cv_path}")

    if args.cv_only:
        print("\n--cv-only set, skipping final training and test evaluation.")
        print("Done!")
        return


    print("\n" + "=" * 70)
    print("Final Training")
    print("=" * 70)

    model = get_model(
        args.model,
        input_dim=n_features,
        dropout_rate=args.dropout,
    )
    print(model)

    trainer = Trainer(
        model, device,
        learning_rate=args.lr,
        model_type=args.model,
        epochs=args.epochs,
    )

    if use_val and 'X_val' in split:
        print("Training with early stopping on validation set...")
        train_result = trainer.fit_with_validation(
            X_train, y_train,
            split['X_val'], split['y_val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            history_df = pd.DataFrame(train_result['history'])
            history_path = os.path.join(args.output_dir, 'training_history.csv')
            history_df.to_csv(history_path, index=False)
            print(f"Training history saved to: {history_path}")
    else:
        trainer.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, 'model.pt')
        trainer.save(model_path)

    if has_test_set:
        X_test = split['X_test']
        y_test = split['y_test']
        test_indices = split['test_indices']

        test_label = (
            f"Test ({args.test_dataset})"
            if args.training_method != 'none'
            else "Test"
        )
        trainer.evaluate(X_test, y_test, label=test_label)

        print("\nRunning Sensitivity Analysis on test set...")
        test_config = ds.config[test_indices]

        sensitivity_df = run_sensitivity_analysis(
            model,
            X_test,
            y_test,
            device,
            config_array=test_config,
            batch_size=4096,
        )

        print("\nSensitivity Analysis Results:")
        print(sensitivity_df.to_string(index=False))

        if args.output_dir:
            results_path = os.path.join(args.output_dir, 'sensitivity_test_results.csv')
            sensitivity_df.to_csv(results_path, index=False)
            print(f"\nTest results saved to: {results_path}")

        if not args.no_plot:
            _save_sensitivity_plots(sensitivity_df, 'test', args.output_dir)


    if args.training_method == 'three-way':
        X_eval       = split['X_eval']
        y_eval       = split['y_eval']
        eval_indices = split['eval_indices']

        trainer.evaluate(X_eval, y_eval, label=f"Eval ({args.eval_dataset})")

        print("\nRunning Sensitivity Analysis on eval set...")
        eval_config = ds.config[eval_indices]

        eval_sensitivity_df = run_sensitivity_analysis(
            model,
            X_eval,
            y_eval,
            device,
            config_array=eval_config,
            batch_size=4096,
        )

        print("\nSensitivity Analysis Results (Eval):")
        print(eval_sensitivity_df.to_string(index=False))

        if args.output_dir:
            eval_results_path = os.path.join(
                args.output_dir, 'sensitivity_eval_results.csv'
            )
            eval_sensitivity_df.to_csv(eval_results_path, index=False)
            print(f"\nEval results saved to: {eval_results_path}")

        if not args.no_plot:
            _save_sensitivity_plots(eval_sensitivity_df, 'eval', args.output_dir)

    print("\nDone!")


def _save_sensitivity_plots(sensitivity_df, split_name, output_dir):
    """Save sensitivity plots for a given split (test or eval)."""
    metrics = ['F1_Score', 'Accuracy', 'Precision', 'Recall']

    for metric in metrics:
        print(f"\nPlotting {metric.replace('_', ' ')} ({split_name})...")
        save_path = None
        if output_dir:
            save_path = os.path.join(
                output_dir,
                f'sensitivity_{split_name}_{metric.lower()}.png'
            )
        plot_sensitivity_metric(sensitivity_df, metric=metric, save_path=save_path)


if __name__ == '__main__':
    main()
