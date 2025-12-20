#!/usr/bin/env python
"""
Attacker Detection CLI - Train and evaluate attacker detection models.

Usage:
    python main.py --data-path /path/to/dataset.csv --model mlp
    python main.py --data-path data.csv --model mlp --epochs 10 --lr 0.0005
"""

import argparse
import os
import torch
import pandas as pd

from config import (
    TRAINING_FEATURES,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT,
    DEFAULT_TEST_SIZE,
    DEFAULT_SEED,
)
from attacker_detector.models import get_model
from attacker_detector.data import load_data, prepare_data
from attacker_detector.training import Trainer
from attacker_detector.analysis import run_sensitivity_analysis, plot_sensitivity_metric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate attacker detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to CSV dataset'
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nTraining features ({len(TRAINING_FEATURES)}): {TRAINING_FEATURES}")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler, test_indices = prepare_data(
        df,
        TRAINING_FEATURES,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(
        args.model,
        input_dim=len(TRAINING_FEATURES),
        dropout_rate=args.dropout
    )
    print(model)
    
    # Train
    trainer = Trainer(model, device, learning_rate=args.lr, model_type=args.model, epochs=args.epochs)
    trainer.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model if output dir specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, 'model.pt')
        trainer.save(model_path)
    
    # Sensitivity analysis
    print("\nRunning Sensitivity Analysis...")
    df_test = df.iloc[test_indices].reset_index(drop=True)
    
    sensitivity_df = run_sensitivity_analysis(
        model,
        df_test,
        scaler,
        TRAINING_FEATURES,
        device,
        batch_size=4096
    )
    
    print("\nSensitivity Analysis Results:")
    print(sensitivity_df.to_string(index=False))
    
    # Save results
    if args.output_dir:
        results_path = os.path.join(args.output_dir, 'sensitivity_results.csv')
        sensitivity_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    
    # Plot all metrics
    if not args.no_plot:
        metrics = ['F1_Score', 'Accuracy', 'Precision', 'Recall']
        
        for metric in metrics:
            print(f"\nPlotting {metric.replace('_', ' ')}...")
            save_path = None
            if args.output_dir:
                save_path = os.path.join(args.output_dir, f'sensitivity_{metric.lower()}.png')
            
            plot_sensitivity_metric(sensitivity_df, metric=metric, save_path=save_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
