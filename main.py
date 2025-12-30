#!/usr/bin/env python3
"""
Convolutional Neural Network Digit Recognition
===============================================

A portfolio project demonstrating:
- Building a CNN from scratch with PyTorch
- Data augmentation for better generalization
- Training on MNIST (60,000 handwritten digits)
- Cross-dataset evaluation on EMNIST (different writers)
- Stability testing with dropout variance analysis
- Rich visualizations for understanding CNN internals

CNN Architecture: ~1,024,000 parameters
- Conv(32) → Conv(64) → FC(320) → Output(10)
- Data augmentation: rotation (±10°), shift (±10%), scale (±10%)

Usage:
    python main.py                    # Default: 32 epochs, 32 stability runs
    python main.py --epochs 50        # More training + 50 stability runs
    python main.py --no-viz           # Skip visualizations

Expected Results:
    MNIST Test:  99%+ (same distribution as training)
    EMNIST Test: 96%+ (different writers - cross-dataset)
"""

import argparse
from pathlib import Path

from src.model import DigitRecognizer
from src.data import DataManager
from src.trainer import Trainer
from src.visualizer import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN on MNIST, test on EMNIST with stability analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                     # 32 epochs + 32 stability test runs
    python main.py --epochs 50         # 50 epochs + 50 stability test runs
    python main.py --epochs 10         # Quick run with 10 epochs
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=32,
        help='Number of training epochs (default: 32)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer (default: 0.001)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Directory for outputs (default: outputs)'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Skip saving model'
    )
    parser.add_argument(
        '--no-stability', action='store_true',
        help='Skip stability test'
    )
    
    return parser.parse_args()


def print_header(args):
    """Print program header and configuration."""
    print("=" * 60)
    print("CNN DIGIT RECOGNITION")
    print("Train on MNIST → Test on MNIST + EMNIST + Stability")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Stability test runs: {args.epochs}")
    print()


def main():
    """Main training, evaluation, and stability analysis pipeline."""
    args = parse_args()
    print_header(args)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading Datasets")
    print("=" * 60)
    print()
    
    data = DataManager(batch_size=args.batch_size)
    print(data)
    
    # =========================================================================
    # Step 2: Create Model
    # =========================================================================
    print("=" * 60)
    print("STEP 2: Creating CNN")
    print("=" * 60)
    print()
    
    model = DigitRecognizer()
    print(model)
    
    # =========================================================================
    # Step 3: Train
    # =========================================================================
    print("=" * 60)
    print("STEP 3: Training")
    print("=" * 60)
    
    trainer = Trainer(model, data, learning_rate=args.learning_rate)
    trainer.train(epochs=args.epochs)
    
    # =========================================================================
    # Step 4: Save Model
    # =========================================================================
    if not args.no_save:
        print()
        print("=" * 60)
        print("STEP 4: Saving Model")
        print("=" * 60)
        print()
        trainer.save_model(f"{args.output_dir}/model.pt")
    
    # =========================================================================
    # Step 5: Generate Training Visualizations
    # =========================================================================
    viz = None
    if not args.no_viz:
        print()
        print("=" * 60)
        print("STEP 5: Generating Visualizations")
        print("=" * 60)
        
        viz = Visualizer(trainer, output_dir=args.output_dir)
        viz.generate_all()
    
    # =========================================================================
    # Step 6: Stability Test
    # =========================================================================
    stability_results = None
    if not args.no_stability:
        print()
        print("=" * 60)
        print("STEP 6: Stability Test")
        print("=" * 60)
        
        # Run stability test with num_runs = epochs
        stability_results = trainer.stability_test(num_runs=args.epochs)
        
        # Generate stability visualizations
        if not args.no_viz and viz is not None:
            print()
            print("Generating stability visualizations...")
            viz.plot_stability_results(stability_results)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print()
    
    h = trainer.history
    mnist_acc = h['mnist_test_accuracy'][-1]
    emnist_acc = h['emnist_test_accuracy'][-1] if h['emnist_test_accuracy'] else 0
    
    print("Final Results:")
    print(f"  MNIST Test Accuracy:  {mnist_acc:.2f}%")
    if emnist_acc > 0:
        print(f"  EMNIST Test Accuracy: {emnist_acc:.2f}%")
    
    if stability_results and stability_results['accuracies']:
        print()
        print("Stability Test (with dropout):")
        print(f"  Mean:  {stability_results['mean']:.2f}%")
        print(f"  Std:   {stability_results['std']:.2f}%")
        print(f"  Range: {stability_results['min']:.2f}% - {stability_results['max']:.2f}%")
    
    print()
    
    # Interpret results
    if emnist_acc >= 96:
        print("✓ Excellent cross-dataset generalization!")
        print("  The CNN learned highly robust features.")
    elif emnist_acc >= 92:
        print("✓ Very good cross-dataset generalization!")
        print("  The CNN transfers very well to new writers.")
    elif emnist_acc >= 90:
        print("✓ Good cross-dataset generalization!")
        print("  The CNN transfers well to new writers.")
    elif emnist_acc > 0:
        print("  Consider training for more epochs.")
        print("  Try: --epochs 50")
    
    if stability_results and stability_results['std'] < 1.0:
        print()
        print("✓ Model is stable (low variance across test runs)")
    
    if not args.no_viz:
        print()
        print(f"Visualizations saved to: {args.output_dir}/")
        print()
        print("Generated files:")
        print("  01_training_history.png      - Loss and accuracy curves")
        print("  02_sample_data.png           - MNIST vs EMNIST samples")
        print("  03_predictions.png           - Model predictions with confidence")
        print("  04_cross_dataset_eval.png    - Generalization comparison")
        print("  05_conv_filters.png          - Learned convolutional filters")
        print("  06_feature_maps.png          - How CNN sees digits")
        print("  07_confusion_matrix.png      - Error analysis")
        print("  08_confidence_analysis.png   - Confidence vs accuracy")
        if not args.no_stability:
            print("  09_stability_test.png        - Stability analysis (line + box plot)")
    
    print()


if __name__ == "__main__":
    main()
