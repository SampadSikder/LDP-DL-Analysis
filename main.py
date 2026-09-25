#!/usr/bin/env python
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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
    prepare_npy_data_oat,
)
from attacker_detector.training import Trainer
from attacker_detector.training.trainer import run_k_fold_cv, run_hp_search_cv
from attacker_detector.analysis import run_sensitivity_analysis, plot_sensitivity_metric


FT_HP_TYPES = {
    'd_token': int,
    'n_heads': int,
    'n_layers': int,
    'ffn_d_multiplier': float,
    'attention_dropout': float,
    'residual_dropout': float,
}


def _validate_ft_dims(d_tokens, n_heads_list, parser=None):
    """d_token must divide evenly by n_heads -- catch it now, not hours into a grid search."""
    bad = [(d, h) for d in d_tokens for h in n_heads_list if d % h != 0]
    if bad:
        combos = ', '.join(f"d_token={d}/n_heads={h}" for d, h in bad[:8])
        msg = (f"invalid FT-Transformer dimensions: {combos}"
               f"{' ...' if len(bad) > 8 else ''}. d_token must be divisible by n_heads.")
        if parser:
            parser.error(msg)
        raise ValueError(msg)


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

    # FT-Transformer architecture hyperparameters. --<key> pins a value (used with
    # --no-hp-search / --k-folds 0, and collapses that axis during search);
    # --hp-<key> narrows what the grid search explores.
    for ft_key, ft_cast in FT_HP_TYPES.items():
        dashed = ft_key.replace('_', '-')
        parser.add_argument(
            f'--{dashed}', type=ft_cast, default=None, dest=ft_key,
            help=f"FT-Transformer {ft_key} (single value). Used directly with "
                 f"--no-hp-search or --k-folds 0; pins this axis during HP search."
        )
        parser.add_argument(
            f'--hp-{dashed}', type=ft_cast, nargs='+', default=None, dest=f'hp_{ft_key}',
            help=f"{ft_key} values to grid search "
                 f"(default: config.DEFAULT_FT_TRANSFORMER_GRID[{ft_key!r}] = "
                 f"{DEFAULT_FT_TRANSFORMER_GRID[ft_key]})"
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
    parser.add_argument(
        '--protocol',
        type=str,
        default=None,
        help='Keep only rows with this protocol, e.g. OUE, OLH_Server, HST_User, HST_Server'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help='Keep only rows with this dataset_type (training-method none only)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Randomly subsample to at most this many rows (stratified by label), '
             'applied after --protocol/--dataset filtering and before splitting. '
             'Ignored for one-at-a-time datasets (use --train-samples / --test-samples).'
    )

    oat = parser.add_argument_group(
        'one-at-a-time datasets',
        'Used when the dataset directory has a design.json with design "oat" '
        '(generate_dataset.py --design oat). Train/val are drawn only from '
        'training LDP runs and test only from held-out runs.'
    )
    oat.add_argument(
        '--train-samples', type=int, default=500_000,
        help='Train + val rows, 50/50 attacker/benign inside every config; '
             '--val-size is carved out of this'
    )
    oat.add_argument(
        '--test-samples', type=int, default=60_000,
        help='Headline test rows from held-out runs, stratified by config, at '
             'natural attacker prevalence. Sensitivity curves use every held-out row.'
    )
    parser.add_argument(
        '--balanced-batches', action='store_true',
        help='Every training batch is exactly half attackers, half benign'
    )

    args = parser.parse_args()

    if args.dataset and args.training_method != 'none':
        parser.error("--dataset only applies to --training-method none")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be a positive integer")
    if args.train_samples < 2 or args.test_samples < 1:
        parser.error("--train-samples must be >= 2 and --test-samples >= 1")
    if not 0.0 <= args.val_size < 1.0:
        parser.error("--val-size must be in [0, 1)")

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

    idx = np.arange(len(ds))
    if args.protocol:
        idx = idx[ds.protocols[idx] == args.protocol]
    if args.dataset:
        idx = idx[ds.dataset_types[idx] == args.dataset]
    if len(idx) == 0:
        raise ValueError(
            f"No rows for protocol={args.protocol!r}, dataset={args.dataset!r}. "
            f"Available protocols: {sorted(set(ds.protocols))}"
        )
    if ds.is_oat and args.max_samples:
        # Subsampling here would drop held-out runs and unbalance the configs
        # before prepare_npy_data_oat gets to them.
        print(f"  [WARN] --max-samples ignored for a one-at-a-time dataset; "
              f"sizes come from --train-samples {args.train_samples:,} and "
              f"--test-samples {args.test_samples:,}")
    elif args.max_samples and len(idx) > args.max_samples:
        idx, _ = train_test_split(
            idx,
            train_size=args.max_samples,
            random_state=args.seed,
            stratify=ds.labels[idx].astype(int),
        )
        idx = np.sort(idx)
    if len(idx) < len(ds):
        ds.features = ds.features[idx]
        ds.labels = ds.labels[idx]
        ds.config = ds.config[idx]
        print(f"Using {len(ds):,} rows (protocol={args.protocol or 'all'}, "
              f"dataset={args.dataset or 'all'}, max_samples={args.max_samples or 'all'})")

    n_features = ds.features.shape[1]
    print(f"Feature count: {n_features}")
    print(f"Feature names: {ds.feature_names}")
    print(f"Training method: {args.training_method}")

    # Hold out set
    has_test_set = True 

    use_val = args.val_size > 0

    if ds.is_oat:
        if args.training_method == 'three-way':
            raise ValueError("--training-method three-way is not supported for "
                             "one-at-a-time datasets")
        train_mask = test_mask = None
        if args.training_method == 'cross':
            dt = ds.dataset_types
            train_mask = np.isin(dt, args.train_dataset)
            test_mask = dt == args.test_dataset
        print(f"\nPreparing one-at-a-time data (defaults: {ds.design['defaults']})"
              + (f", train={'+'.join(args.train_dataset)}, test={args.test_dataset}"
                 if args.training_method == 'cross' else "") + "...")
        split = prepare_npy_data_oat(
            ds,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            val_size=args.val_size,
            random_state=args.seed,
            train_mask=train_mask,
            test_mask=test_mask,
        )
    elif args.training_method == 'none':
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
    # Pinned --<key> values are the starting point; an HP search overwrites them below.
    best_ft_config = {
        key: getattr(args, key)
        for key in FT_TRANSFORMER_HP_KEYS
        if getattr(args, key) is not None
    }
    if args.model == 'ft_transformer' and 'd_token' in best_ft_config and 'n_heads' in best_ft_config:
        _validate_ft_dims([best_ft_config['d_token']], [best_ft_config['n_heads']])

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
                ft_config=best_ft_config or None,
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
                    searched = getattr(args, f'hp_{key}')
                    pinned = getattr(args, key)
                    if searched is not None:
                        hp_grid[key] = searched
                    elif pinned is not None:
                        hp_grid[key] = [pinned]
                    else:
                        hp_grid[key] = DEFAULT_FT_TRANSFORMER_GRID[key]
                _validate_ft_dims(hp_grid['d_token'], hp_grid['n_heads'])
                n_configs = 1
                for values in hp_grid.values():
                    n_configs *= len(values)
                print(f"\nHP grid: {n_configs} configs x {args.k_folds} folds = "
                      f"{n_configs * args.k_folds} training runs")
                for key, values in hp_grid.items():
                    print(f"  {key}: {values}")
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
            balanced_batches=args.balanced_batches,
        )

    else:
        if args.balanced_batches:
            print("  [WARN] --balanced-batches needs a validation set (--val-size > 0); "
                  "training with a plain shuffle")
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

        # One-at-a-time datasets: curves come from every held-out row (not just
        # the headline test sample), sliced so each parameter varies alone.
        oat_defaults = ds.design['defaults'] if ds.is_oat else None
        sens_indices = split.get('sens_indices', test_indices)

        print(f"\nRunning Sensitivity Analysis on {len(sens_indices):,} "
              f"{'held-out' if ds.is_oat else 'test'} rows...")
        sensitivity_df = run_sensitivity_analysis(
            model,
            ds.features[sens_indices],
            ds.labels[sens_indices],
            device,
            config_array=ds.config[sens_indices],
            batch_size=4096,
            oat_defaults=oat_defaults,
        )

        print("\nSensitivity Analysis Results:")
        print(sensitivity_df.to_string(index=False))

        if args.output_dir:
            results_path = os.path.join(args.output_dir, 'sensitivity_test_results.csv')
            sensitivity_df.to_csv(results_path, index=False)
            print(f"\nSensitivity test results saved to: {results_path}")

        if 'sens_balanced_indices' in split:
            bal = split['sens_balanced_indices']
            print(f"\nRunning Sensitivity Analysis on the balanced held-out view "
                  f"({len(bal):,} rows, 50/50 per config -- NOT comparable to "
                  f"natural-prevalence results)...")
            balanced_df = run_sensitivity_analysis(
                model, ds.features[bal], ds.labels[bal], device,
                config_array=ds.config[bal], batch_size=4096,
                oat_defaults=oat_defaults,
            )
            if args.output_dir:
                bal_path = os.path.join(args.output_dir, 'sensitivity_test_balanced.csv')
                balanced_df.to_csv(bal_path, index=False)
                print(f"Balanced sensitivity results saved to: {bal_path}")

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
