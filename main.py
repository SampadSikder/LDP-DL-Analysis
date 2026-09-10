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

    # Cross-dataset with combined training types + HP grid search (default when k-folds > 0):
    python main.py --data-path /path/to/output_dir --model mlp \\
        --training-method cross --train-dataset zipf emoji --test-dataset fire --k-folds 5

    # Same, but skip HP search and use --lr/--dropout directly:
    python main.py --data-path /path/to/output_dir --model mlp \\
        --training-method cross --train-dataset zipf emoji --test-dataset fire \\
        --k-folds 5 --no-hp-search

    # Also grid-search pos_weight (auto + a couple fixed values):
    python main.py --data-path /path/to/output_dir --model mlp \\
        --training-method cross --train-dataset zipf emoji --test-dataset fire \\
        --k-folds 5 --hp-pos-weight auto 1.0 5.77
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
    DEFAULT_TABULAR_HP_GRID,
    DEFAULT_HIDDEN_SIZE_GRID,
    DEFAULT_FT_TRANSFORMER_GRID,
)
from attacker_detector.models import get_model, FT_TRANSFORMER_HP_KEYS
from attacker_detector.data import (
    load_npy_dataset,
    prepare_npy_data,
    prepare_npy_data_by_dataset_type,
)
from attacker_detector.training import Trainer
from attacker_detector.training.trainer import run_k_fold_cv, run_hp_search_cv
from attacker_detector.analysis import run_sensitivity_analysis, plot_sensitivity_metric


def _save_cv_summary(cv_results: dict, output_dir: str) -> str:
    """Save the mean/std summary of a k-fold CV run to cv_summary.csv."""
    summary_df = pd.DataFrame([
        {'metric': metric, 'mean': mean_val, 'std': cv_results['std'][metric]}
        for metric, mean_val in cv_results['mean'].items()
    ])
    summary_path = os.path.join(output_dir, 'cv_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def _parse_pos_weight(value: str):
    """'auto' -> None (auto-compute neg/pos ratio); otherwise parse as float."""
    return None if value == 'auto' else float(value)


def _parse_hidden_sizes(value: str):
    """'64,32,16' -> [64, 32, 16]. 'default' defers to DEFAULT_HIDDEN_SIZE_GRID."""
    if value == 'default':
        return 'default'
    try:
        sizes = [int(v) for v in value.split(',') if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid hidden sizes {value!r}: expected comma-separated integers"
        )
    if not sizes or any(s < 1 for s in sizes):
        raise argparse.ArgumentTypeError(
            f"invalid hidden sizes {value!r}: widths must be positive integers"
        )
    return sizes


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
        choices=['mlp', 'gan', 'attention', 'ft_transformer'],
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
        '--hidden-sizes',
        type=_parse_hidden_sizes,
        default=None,
        help="MLP hidden layer widths, comma-separated (e.g. --hidden-sizes 64,32,16). "
             "Default applies the geometric pyramid rule to the feature count."
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
        '--no-hp-search',
        action='store_true',
        default=False,
        help='Skip HP grid search; use --lr/--dropout directly for CV and final training'
    )
    parser.add_argument(
        '--hp-lr',
        type=float,
        nargs='+',
        default=None,
        help='Learning rate values to search (default: config.DEFAULT_TABULAR_HP_GRID)'
    )
    parser.add_argument(
        '--hp-dropout',
        type=float,
        nargs='+',
        default=None,
        help='Dropout values to search (default: config.DEFAULT_TABULAR_HP_GRID)'
    )
    parser.add_argument(
        '--pos-weight',
        type=str,
        default='auto',
        help="Positive class weight for BCE loss. 'auto' computes neg_count / pos_count "
             "from the training data, or provide a float. Used directly with "
             "--no-hp-search, or as the fallback for configs when --hp-pos-weight is unset."
    )
    parser.add_argument(
        '--hp-pos-weight',
        type=str,
        nargs='+',
        default=None,
        help="pos_weight values to search, e.g. --hp-pos-weight auto 1.0 5.77 "
             "(each value is 'auto' or a float). Not searched by default — only "
             "--lr/--dropout are searched unless this is given."
    )
    parser.add_argument(
        '--hp-hidden-sizes',
        type=_parse_hidden_sizes,
        nargs='+',
        default=None,
        help="MLP shapes to search, e.g. --hp-hidden-sizes 256,128,64 64,32,16. "
             "Not searched by default. Pass 'default' to use config.DEFAULT_HIDDEN_SIZE_GRID."
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
        nargs='+',
        default=None,
        choices=DATASET_TYPES,
        help='dataset_type(s) used for training, e.g. --train-dataset zipf emoji '
             '(required for cross / three-way)'
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
        if args.test_dataset in args.train_dataset:
            parser.error("--test-dataset must not also appear in --train-dataset")

    if args.training_method == 'three-way':
        if not args.eval_dataset:
            parser.error("--training-method=three-way requires --eval-dataset")
        if args.eval_dataset == args.test_dataset or args.eval_dataset in args.train_dataset:
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
            f"\nPreparing data (train={'+'.join(args.train_dataset)}, "
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

    # Resolved training params — may be overridden by HP search below
    pos_weight_arg = _parse_pos_weight(args.pos_weight)
    best_lr = args.lr
    best_dropout = args.dropout
    best_pos_weight = pos_weight_arg
    best_hidden_sizes = args.hidden_sizes
    best_ft_config = {}

    if args.k_folds > 0:
        if 'X_trainval' in split:
            X_trainval = split['X_trainval']
            y_trainval = split['y_trainval']
        else:
            X_trainval = X_train
            y_trainval = y_train

        if args.no_hp_search:
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
                pos_weight=pos_weight_arg,
                hidden_sizes=args.hidden_sizes,
            )

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                cv_df = pd.DataFrame(cv_results['fold_results'])
                cv_path = os.path.join(args.output_dir, 'cv_results.csv')
                cv_df.to_csv(cv_path, index=False)
                print(f"\nCV results saved to: {cv_path}")

                summary_path = _save_cv_summary(cv_results, args.output_dir)
                print(f"CV summary saved to: {summary_path}")
        else:
            hp_grid = {}
            hp_grid['lr'] = args.hp_lr if args.hp_lr is not None else DEFAULT_TABULAR_HP_GRID['lr']
            if args.model == 'ft_transformer':
                for key in FT_TRANSFORMER_HP_KEYS:
                    hp_grid[key] = DEFAULT_FT_TRANSFORMER_GRID[key]
            else:
                hp_grid['dropout'] = args.hp_dropout if args.hp_dropout is not None else DEFAULT_TABULAR_HP_GRID['dropout']
            if args.hp_pos_weight is not None:
                hp_grid['pos_weight'] = [_parse_pos_weight(v) for v in args.hp_pos_weight]
            if args.hp_hidden_sizes is not None:
                if 'default' in args.hp_hidden_sizes:
                    hp_grid['hidden_sizes'] = DEFAULT_HIDDEN_SIZE_GRID
                else:
                    hp_grid['hidden_sizes'] = args.hp_hidden_sizes

            search_results = run_hp_search_cv(
                model_type=args.model,
                input_dim=n_features,
                X_trainval=X_trainval,
                y_trainval=y_trainval,
                hp_grid=hp_grid,
                n_folds=args.k_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                device=device,
                seed=args.seed,
                base_learning_rate=args.lr,
                base_dropout_rate=args.dropout,
                base_pos_weight=pos_weight_arg,
                base_hidden_sizes=args.hidden_sizes,
            )

            best_config = search_results['best_config']
            best_lr = best_config.get('lr', args.lr)
            best_dropout = best_config.get('dropout', args.dropout)
            best_pos_weight = best_config.get('pos_weight', pos_weight_arg)
            best_hidden_sizes = best_config.get('hidden_sizes', args.hidden_sizes)
            best_ft_config = {k: best_config[k] for k in FT_TRANSFORMER_HP_KEYS if k in best_config}

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)

                search_df = pd.DataFrame(search_results['all_results'])
                search_path = os.path.join(args.output_dir, 'hp_search_results.csv')
                search_df.to_csv(search_path, index=False)
                print(f"\nHP search results saved to: {search_path}")

                cv_df = pd.DataFrame(search_results['best_cv_results']['fold_results'])
                cv_path = os.path.join(args.output_dir, 'cv_results.csv')
                cv_df.to_csv(cv_path, index=False)
                print(f"Best config CV results saved to: {cv_path}")

                summary_path = _save_cv_summary(search_results['best_cv_results'], args.output_dir)
                print(f"Best config CV summary saved to: {summary_path}")

    if args.cv_only:
        print("\n--cv-only set, skipping final training and test evaluation.")
        print("Done!")
        return


    print("\n" + "=" * 70)
    print("Final Training")
    print("=" * 70)
    if args.k_folds > 0 and not args.no_hp_search:
        print(
            f"  Using best HP config from CV search: "
            f"lr={best_lr}, dropout={best_dropout}, "
            f"pos_weight={'auto' if best_pos_weight is None else best_pos_weight}, "
            f"hidden_sizes={best_hidden_sizes or 'default'}"
            + (f", ft_config={best_ft_config}" if best_ft_config else "")
        )

    model_kwargs = {'dropout_rate': best_dropout}
    if args.model == 'mlp' and best_hidden_sizes is not None:
        model_kwargs['hidden_sizes'] = best_hidden_sizes
    if args.model == 'ft_transformer' and best_ft_config:
        model_kwargs.update(best_ft_config)

    model = get_model(
        args.model,
        input_dim=n_features,
        **model_kwargs,
    )
    print(model)

    trainer = Trainer(
        model, device,
        learning_rate=best_lr,
        model_type=args.model,
        epochs=args.epochs,
        pos_weight=best_pos_weight,
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

    else:
        train_result = trainer.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        history_df = pd.DataFrame(train_result['history'])
        history_path = os.path.join(args.output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to: {history_path}")

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
        test_metrics = trainer.evaluate(X_test, y_test, label=test_label)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            test_metrics_df = pd.DataFrame([{
                'label': test_label,
                'n_samples': len(y_test),
                **test_metrics,
            }])
            test_metrics_path = os.path.join(args.output_dir, 'test_results.csv')
            test_metrics_df.to_csv(test_metrics_path, index=False)
            print(f"Test results saved to: {test_metrics_path}")

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
            print(f"\nSensitivity test results saved to: {results_path}")

        if not args.no_plot:
            _save_sensitivity_plots(sensitivity_df, 'test', args.output_dir)


    if args.training_method == 'three-way':
        X_eval       = split['X_eval']
        y_eval       = split['y_eval']
        eval_indices = split['eval_indices']

        eval_label = f"Eval ({args.eval_dataset})"
        eval_metrics = trainer.evaluate(X_eval, y_eval, label=eval_label)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            eval_metrics_df = pd.DataFrame([{
                'label': eval_label,
                'n_samples': len(y_eval),
                **eval_metrics,
            }])
            eval_metrics_path = os.path.join(args.output_dir, 'eval_results.csv')
            eval_metrics_df.to_csv(eval_metrics_path, index=False)
            print(f"Eval results saved to: {eval_metrics_path}")

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
            print(f"\nSensitivity eval results saved to: {eval_results_path}")

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
