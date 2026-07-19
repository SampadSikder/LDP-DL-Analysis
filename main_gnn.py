#!/usr/bin/env python

import argparse
import os

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT,
    DEFAULT_SEED,
    DEFAULT_GNN_HIDDEN_DIM,
    DEFAULT_GNN_NUM_HEADS,
    DEFAULT_GNN_LAMBDA_AGG,
    DEFAULT_GNN_LAMBDA_UTILITY,
    DEFAULT_GNN_PATIENCE,
    DEFAULT_GNN_K_FOLDS,
    DEFAULT_GNN_BATCH_SIZE,
    DEFAULT_GNN_INPUT_DIM,
    DEFAULT_GNN_HP_GRID,
)
from attacker_detector.models import get_model
from attacker_detector.data.graph_dataset import GraphDatasetLoader
from attacker_detector.training.gnn_trainer import (
    GNNTrainer,
    init_weights,
    run_k_fold_cv,
    run_hp_search_cv,
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate GNN attacker detection models on graph data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='dataset_pca.pt',
        help='Path to .pt file containing list of PyG Data objects',
    )
    parser.add_argument(
        '--max-graphs',
        type=int,
        default=None,
        help='Limit number of graphs to load (for debugging)',
    )

    # Model
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['gat', 'graphsage'],
        help='GNN model type',
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=DEFAULT_GNN_HIDDEN_DIM,
        help='Hidden dimension per head',
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=DEFAULT_GNN_NUM_HEADS,
        help='Number of attention heads',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DEFAULT_DROPOUT,
        help='Dropout rate',
    )

    # Training
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Maximum training epochs',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate',
    )
    parser.add_argument(
        '--lambda-agg',
        type=float,
        default=DEFAULT_GNN_LAMBDA_AGG,
        help='Aggregation loss weight (lambda)',
    )
    parser.add_argument(
        '--lambda-utility',
        type=float,
        default=DEFAULT_GNN_LAMBDA_UTILITY,
        help='Utility loss weight (lambda_utility)',
    )
    parser.add_argument(
        '--utility-metric',
        type=str,
        default='js',
        choices=['js', 'wasserstein'],
        help='Metric for utility loss calculation',
    )
    parser.add_argument(
        '--pos-weight',
        type=str,
        default='auto',
        help='Positive class weight for BCE loss. '
             "'auto' computes neg/pos ratio from data, "
             "or provide a float",
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_GNN_BATCH_SIZE,
        help='Number of graphs per batch',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=DEFAULT_GNN_PATIENCE,
        help='Early stopping patience (epochs)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for predictions (lower = more attacker predictions)',
    )

    # Initialization
    parser.add_argument(
        '--init-method',
        type=str,
        default='xavier_uniform',
        choices=['xavier_uniform', 'kaiming', 'orthogonal', 'default'],
        help='Weight initialization method',
    )

    # Cross-validation & hyperparameter search
    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for k-fold CV (0 to skip)',
    )
    parser.add_argument(
        '--hp-lambda-agg',
        type=float,
        nargs='+',
        default=None,
        help='Lambda agg values to search',
    )
    parser.add_argument(
        '--hp-lambda-utility',
        type=float,
        nargs='+',
        default=None,
        help='Lambda utility values to search',
    )
    parser.add_argument(
        '--hp-num-heads',
        type=int,
        nargs='+',
        default=None,
        help='Number of attention heads to search (GAT only, e.g. --hp-num-heads 2 4 8)',
    )
    parser.add_argument(
        '--hp-init-method',
        type=str,
        nargs='+',
        default=None,
        choices=['xavier_uniform', 'kaiming', 'orthogonal', 'default'],
        help='Weight initialization methods to search',
    )
    parser.add_argument(
        '--cv-only',
        action='store_true',
        default=False,
        help='Run only CV / HP search and skip final training + test evaluation',
    )
    parser.add_argument(
        '--no-hp-search',
        action='store_true',
        default=False,
        help='Skip HP grid search; use CLI params directly for CV and final training',
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Fraction of graphs for test set',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Fraction of graphs for validation set',
    )

    # Cross-protocol / cross-dataset evaluation
    parser.add_argument(
        '--split-by',
        type=str,
        default='none',
        choices=['none', 'protocol', 'dataset_type', 'both'],
        help='After test evaluation, print per-group metrics sliced by this metadata key',
    )
    parser.add_argument(
        '--cross-eval-path',
        type=str,
        default=None,
        help='Path to a second .pt dataset for out-of-distribution evaluation',
    )

    # Output
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save model, metrics, and plots',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed',
    )

    return parser.parse_args()


def run_gnn_sensitivity_analysis(per_graph_results: list) -> pd.DataFrame:

    rows = []
    df = pd.DataFrame(per_graph_results)

    param_cols = {
        'epsilon':      'Epsilon ($\\epsilon$)',
        'ratio':        'Attacker Ratio ($\\beta$)',
        'protocol':     'Protocol',
        'dataset_type': 'Dataset Type',
    }

    for col, label in param_cols.items():
        if col not in df.columns:
            continue
        for val in sorted(df[col].dropna().unique()):
            subset = df[df[col] == val]
            rows.append({
                'Parameter_Type':  col,
                'Parameter_Label': label,
                'Value':           val,
                'Accuracy':        subset['Accuracy'].mean(),
                'Precision':       subset['Precision'].mean(),
                'Recall':          subset['Recall'].mean(),
                'F1_Score':        subset['F1_Score'].mean(),
                'Count':           len(subset),
            })

    return pd.DataFrame(rows)


def _print_grouped_metrics(
    per_graph_results: list,
    split_by: str,
    output_dir: str = None,
    label: str = 'Test',
) -> None:
    """Print per-group metrics sliced by protocol, dataset_type, or both."""
    df = pd.DataFrame(per_graph_results)
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

    if split_by == 'both':
        group_keys = ['protocol', 'dataset_type']
    else:
        group_keys = [split_by]

    # Filter out keys not present in data
    group_keys = [k for k in group_keys if k in df.columns]
    if not group_keys:
        print(f"  No metadata columns found for split-by={split_by}")
        return

    print(f"\n{'─'*70}")
    print(f"Grouped {label} Metrics (split-by: {split_by})")
    print(f"{'─'*70}")

    grouped = df.groupby(group_keys)
    rows = []
    for group_val, group_df in sorted(grouped):
        group_label = group_val if isinstance(group_val, str) else ' / '.join(str(v) for v in group_val)
        row = {k: group_label if i == 0 else '' for i, k in enumerate(group_keys)}
        row.update({k: group_label for k in group_keys} if len(group_keys) == 1 else dict(zip(group_keys, group_val)))
        row['Count'] = len(group_df)
        for m in metric_cols:
            row[m] = group_df[m].mean()
        rows.append(row)
        print(f"  {group_label}: "
              f"F1={row['F1_Score']:.4f}  "
              f"Prec={row['Precision']:.4f}  "
              f"Rec={row['Recall']:.4f}  "
              f"Acc={row['Accuracy']:.4f}  "
              f"(n={row['Count']})")

    if output_dir:
        summary_df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f'grouped_metrics_{label.lower().replace(" ", "_")}.csv')
        summary_df.to_csv(path, index=False)
        print(f"  Saved to: {path}")


def _save_gnn_sensitivity_plots(sensitivity_df: pd.DataFrame, output_dir: str = None) -> None:
    metrics = ['F1_Score', 'Accuracy', 'Precision', 'Recall']
    numeric_params = ['epsilon', 'ratio']
    categorical_params = ['protocol', 'dataset_type']

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for metric in metrics:
        params  = [p for p in numeric_params + categorical_params
                   if p in sensitivity_df['Parameter_Type'].values]
        n_plots = len(params)
        if n_plots == 0:
            continue

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for ax, param in zip(axes, params):
            data = sensitivity_df[sensitivity_df['Parameter_Type'] == param].copy()
            if data.empty:
                ax.set_title(f"No data for {param}")
                continue

            label = data['Parameter_Label'].iloc[0]

            if param in numeric_params:
                data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
                ax.plot(
                    data['Value'], data[metric],
                    marker='X', markersize=8,
                    linestyle='--', linewidth=2,
                    color='#1f77b4', label='GAT Model'
                )
                ax.set_xlabel(label, fontsize=14, fontweight='bold')
            else:
                ax.bar(data['Value'].astype(str), data[metric], color='#1f77b4')
                ax.set_xlabel(label, fontsize=14, fontweight='bold')

            ax.set_ylabel(metric.replace('_', ' '), fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Impact of Parameters on {metric.replace('_', ' ')}",
            fontsize=16, y=1.05
        )
        plt.tight_layout()

        fname = f"sensitivity_{metric.lower()}.png"
        save_path = os.path.join(output_dir, fname) if output_dir else fname
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
        plt.close(fig)


def build_model_kwargs(args, input_dim: int) -> dict:
    kwargs = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout,
    }
    if args.model == 'gat':
        kwargs['num_heads'] = args.num_heads
    return kwargs


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = get_device()

    print(f"\nLoading graph dataset from: {args.data_path}")
    dataset = GraphDatasetLoader(args.data_path, max_graphs=args.max_graphs)
    input_dim = dataset.input_dim

    print(f"\nNode feature dimension: {input_dim}")
    print(f"Model: {args.model}")

    model_kwargs = build_model_kwargs(args, input_dim)


    # ── Hyperparameter selection via k-fold CV ──────────────────────────
    # Resolved training params — may be overridden by HP search below
    best_lambda_agg = args.lambda_agg
    best_lambda_utility = args.lambda_utility
    best_init_method = args.init_method
    best_model_kwargs = dict(model_kwargs)
    pos_weight_arg = None if args.pos_weight == 'auto' else float(args.pos_weight)

    if args.k_folds > 0:
        model_class = type(get_model(args.model, **model_kwargs))

        if args.no_hp_search:
            # Plain k-fold CV with fixed CLI params
            cv_results = run_k_fold_cv(
                model_class=model_class,
                model_kwargs=model_kwargs,
                dataset_loader=dataset,
                n_folds=args.k_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                lambda_agg=args.lambda_agg,
                lambda_utility=args.lambda_utility,
                utility_metric=args.utility_metric,
                patience=args.patience,
                init_method=args.init_method,
                model_type=args.model,
                device=device,
                seed=args.seed,
                pos_weight=pos_weight_arg,
                threshold=args.threshold,
            )

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                cv_df = pd.DataFrame(cv_results['fold_results'])
                cv_path = os.path.join(args.output_dir, 'cv_results.csv')
                cv_df.to_csv(cv_path, index=False)
                print(f"\nCV results saved to: {cv_path}")

        else:
            # HP grid search via k-fold CV
            hp_grid = {}
            if args.hp_lambda_agg is not None:
                hp_grid['lambda_agg'] = args.hp_lambda_agg
            elif 'lambda_agg' in DEFAULT_GNN_HP_GRID:
                hp_grid['lambda_agg'] = DEFAULT_GNN_HP_GRID['lambda_agg']

            if args.hp_lambda_utility is not None:
                hp_grid['lambda_utility'] = args.hp_lambda_utility
            elif 'lambda_utility' in DEFAULT_GNN_HP_GRID:
                hp_grid['lambda_utility'] = DEFAULT_GNN_HP_GRID['lambda_utility']

            if args.model == 'gat':
                if args.hp_num_heads is not None:
                    hp_grid['num_heads'] = args.hp_num_heads
                elif 'num_heads' in DEFAULT_GNN_HP_GRID:
                    hp_grid['num_heads'] = DEFAULT_GNN_HP_GRID['num_heads']

            if args.hp_init_method is not None:
                hp_grid['init_method'] = args.hp_init_method
            elif 'init_method' in DEFAULT_GNN_HP_GRID:
                hp_grid['init_method'] = DEFAULT_GNN_HP_GRID['init_method']

            search_results = run_hp_search_cv(
                model_class=model_class,
                base_model_kwargs=model_kwargs,
                dataset_loader=dataset,
                hp_grid=hp_grid,
                n_folds=args.k_folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                init_method=args.init_method,
                model_type=args.model,
                device=device,
                seed=args.seed,
                base_learning_rate=args.lr,
                base_lambda_agg=args.lambda_agg,
                base_lambda_utility=args.lambda_utility,
                utility_metric=args.utility_metric,
                pos_weight=pos_weight_arg,
                threshold=args.threshold,
            )

            best_config = search_results['best_config']

            best_lambda_agg = best_config.get('lambda_agg', args.lambda_agg)
            best_lambda_utility = best_config.get('lambda_utility', args.lambda_utility)
            best_init_method = best_config.get('init_method', args.init_method)
            if 'num_heads' in best_config and args.model == 'gat':
                best_model_kwargs['num_heads'] = best_config['num_heads']

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

    if args.cv_only:
        print("\n--cv-only set, skipping final training and test evaluation.")
        print("Done!")
        return

    print("\n" + "=" * 70)
    print("Final Training ( Train+Val → Test)")
    print("=" * 70)
    if args.k_folds > 0:
        print(f"  Using best HP config from CV search:")
        print(f"    Learning rate:  {args.lr}")
        print(f"    Lambda (agg):   {best_lambda_agg}")
        print(f"    Lambda (util):  {best_lambda_utility} ({args.utility_metric})")
        print(f"    Model kwargs:   {best_model_kwargs}")

    splits = dataset.stratified_split(
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    pin = device.type == 'cuda'
    train_loader = dataset.get_dataloader(
        splits['train'], batch_size=args.batch_size, shuffle=True, pin_memory=pin
    )
    val_loader = dataset.get_dataloader(
        splits['val'], batch_size=args.batch_size, shuffle=False, pin_memory=pin
    )
    test_loader = dataset.get_dataloader(
        splits['test'], batch_size=args.batch_size, shuffle=False, pin_memory=pin
    )

    model = get_model(args.model, **best_model_kwargs)
    init_weights(model, method=best_init_method)
    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    trainer = GNNTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        lambda_agg=best_lambda_agg,
        lambda_utility=best_lambda_utility,
        utility_metric=args.utility_metric,
        patience=args.patience,
        model_type=args.model,
        pos_weight=pos_weight_arg,
        threshold=args.threshold,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f'{args.model}_model.pt')
        trainer.save(model_path)

        history_df = pd.DataFrame(history)
        history_path = os.path.join(args.output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to: {history_path}")


    print("\n" + "=" * 70)
    print("Generalization: Test Evaluation")
    print("=" * 70)

    test_metrics = trainer.evaluate(test_loader, label="Test")

    per_graph_loader = dataset.get_dataloader(
        splits['test'], batch_size=1, shuffle=False, pin_memory=pin
    )
    per_graph_results = trainer.evaluate_per_graph(per_graph_loader)

    print("\nRunning Sensitivity Analysis on test graphs...")
    sensitivity_df = run_gnn_sensitivity_analysis(per_graph_results)

    print("\nSensitivity Analysis Results (Test):")
    print(sensitivity_df.to_string(index=False))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        test_metrics_path = os.path.join(args.output_dir, 'test_metrics.csv')
        pd.DataFrame([test_metrics]).to_csv(test_metrics_path, index=False)
        print(f"\nTest metrics saved to: {test_metrics_path}")

        per_graph_df = pd.DataFrame(per_graph_results)
        per_graph_path = os.path.join(args.output_dir, 'per_graph_results.csv')
        per_graph_df.to_csv(per_graph_path, index=False)
        print(f"Per-graph results saved to: {per_graph_path}")

        sensitivity_path = os.path.join(args.output_dir, 'sensitivity_test_results.csv')
        sensitivity_df.to_csv(sensitivity_path, index=False)
        print(f"Sensitivity results saved to: {sensitivity_path}")

    print("\nPlotting sensitivity analysis...")
    _save_gnn_sensitivity_plots(sensitivity_df, output_dir=args.output_dir)

    # ── Grouped evaluation (split-by protocol / dataset_type / both) ───
    if args.split_by != 'none':
        _print_grouped_metrics(
            per_graph_results,
            split_by=args.split_by,
            output_dir=args.output_dir,
            label='Test',
        )

    # ── Out-of-distribution evaluation on a second dataset ─────────────
    if args.cross_eval_path:
        print("\n" + "=" * 70)
        print(f"OOD Evaluation: {args.cross_eval_path}")
        print("=" * 70)

        ood_dataset = GraphDatasetLoader(args.cross_eval_path)
        ood_all_idx = list(range(len(ood_dataset)))

        ood_loader = ood_dataset.get_dataloader(
            ood_all_idx, batch_size=args.batch_size, shuffle=False, pin_memory=pin
        )
        ood_metrics = trainer.evaluate(ood_loader, label="OOD")

        ood_per_graph_loader = ood_dataset.get_dataloader(
            ood_all_idx, batch_size=1, shuffle=False, pin_memory=pin
        )
        ood_per_graph = trainer.evaluate_per_graph(ood_per_graph_loader)

        print("\nRunning Sensitivity Analysis on OOD graphs...")
        ood_sensitivity_df = run_gnn_sensitivity_analysis(ood_per_graph)
        print("\nSensitivity Analysis Results (OOD):")
        print(ood_sensitivity_df.to_string(index=False))

        if args.split_by != 'none':
            _print_grouped_metrics(
                ood_per_graph,
                split_by=args.split_by,
                output_dir=args.output_dir,
                label='OOD',
            )

        if args.output_dir:
            ood_metrics_path = os.path.join(args.output_dir, 'ood_metrics.csv')
            pd.DataFrame([ood_metrics]).to_csv(ood_metrics_path, index=False)
            print(f"\nOOD metrics saved to: {ood_metrics_path}")

            ood_per_graph_df = pd.DataFrame(ood_per_graph)
            ood_per_graph_path = os.path.join(args.output_dir, 'ood_per_graph_results.csv')
            ood_per_graph_df.to_csv(ood_per_graph_path, index=False)
            print(f"OOD per-graph results saved to: {ood_per_graph_path}")

            ood_sens_path = os.path.join(args.output_dir, 'ood_sensitivity_results.csv')
            ood_sensitivity_df.to_csv(ood_sens_path, index=False)
            print(f"OOD sensitivity results saved to: {ood_sens_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
