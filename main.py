#!/usr/bin/env python3
"""
Convolutional Neural Network Digit Recognition
===============================================

A portfolio project demonstrating:
- Building a CNN from scratch with PyTorch
- Training on MNIST (60,000 handwritten digits)
- Cross-dataset evaluation on EMNIST (different writers)
- Rich visualizations for understanding CNN internals

CNN achieves much better cross-dataset generalization than MLP because
convolutional layers learn local features (edges, curves) that transfer
well across different handwriting styles.

Usage:
    python main.py                    # Default: 10 epochs
    python main.py --epochs 20        # More training
    python main.py --no-viz           # Skip visualizations

Expected Results:
    MNIST Test:  99%+ (same distribution as training)
    EMNIST Test: 90-95% (different writers - cross-dataset)
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
        description='Train CNN on MNIST, test on EMNIST',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                     # Quick run with defaults
    python main.py --epochs 20         # Train longer for best results
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)'
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
    
    return parser.parse_args()


def print_header(args):
    """Print program header and configuration."""
    print("=" * 60)
    print("CNN DIGIT RECOGNITION")
    print("Train on MNIST → Test on MNIST + EMNIST")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print()


def main():
    """Main training and evaluation pipeline."""
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
    # Step 5: Generate Visualizations
    # =========================================================================
    if not args.no_viz:
        print()
        print("=" * 60)
        print("STEP 5: Generating Visualizations")
        print("=" * 60)
        
        viz = Visualizer(trainer, output_dir=args.output_dir)
        viz.generate_all()
    
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
    print()
    
    # Interpret results
    if emnist_acc >= 90:
        print("✓ Excellent cross-dataset generalization!")
        print("  The CNN learned robust features that work")
        print("  very well on completely different handwriting.")
    elif emnist_acc >= 85:
        print("✓ Good cross-dataset generalization!")
        print("  The CNN transfers well to new writers.")
    elif emnist_acc > 0:
        print("  Consider training for more epochs.")
        print("  Try: --epochs 20")
    
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
    
    print()


if __name__ == "__main__":
    main()
