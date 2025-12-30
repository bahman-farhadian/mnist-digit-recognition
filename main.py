#!/usr/bin/env python3
"""
Neural Network Digit Recognition
=================================

A portfolio project demonstrating:
- Building a neural network from scratch with PyTorch
- Training on MNIST (60,000 handwritten digits)
- Cross-dataset evaluation on EMNIST (different writers)
- Rich visualizations for understanding what the network learns

MNIST and EMNIST are both from the NIST database, but with different
writers. Good EMNIST accuracy proves the model learned GENERAL digit
features, not just MNIST-specific patterns.

Usage:
    python main.py                    # Default: 5 epochs, 1024 neurons
    python main.py --epochs 10        # More training
    python main.py --hidden-size 512  # Smaller network
    python main.py --no-viz           # Skip visualizations

Expected Results:
    MNIST Test:  97-98% (same distribution as training)
    EMNIST Test: 85-95% (different writers - cross-dataset)
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
        description='Train neural network on MNIST, test on EMNIST',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                     # Quick run with defaults
    python main.py --epochs 10         # Train longer
    python main.py --hidden-size 2048  # Larger network
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate for Adam optimizer (default: 0.001)'
    )
    parser.add_argument(
        '--hidden-size', type=int, default=1024,
        help='Number of hidden neurons (default: 1024)'
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
    print("NEURAL NETWORK DIGIT RECOGNITION")
    print("Train on MNIST → Test on MNIST + EMNIST")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Hidden neurons: {args.hidden_size}")
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
    print("STEP 2: Creating Neural Network")
    print("=" * 60)
    print()
    
    model = DigitRecognizer(hidden_size=args.hidden_size)
    print(model)
    print()
    
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
    if emnist_acc >= 85:
        print("✓ Excellent cross-dataset generalization!")
        print("  The model learned general digit features that work")
        print("  well on completely different handwriting samples.")
    elif emnist_acc >= 75:
        print("✓ Good cross-dataset generalization!")
        print("  The model transfers reasonably well to new writers.")
    elif emnist_acc > 0:
        print("  EMNIST accuracy could be improved with more training")
        print("  or a larger network. Try: --epochs 10 --hidden-size 2048")
    
    if not args.no_viz:
        print()
        print(f"Visualizations saved to: {args.output_dir}/")
        print()
        print("Generated files:")
        print("  01_training_history.png      - Loss and accuracy curves")
        print("  02_sample_data.png           - MNIST vs EMNIST samples")
        print("  03_predictions.png           - Model predictions with confidence")
        print("  04_cross_dataset_eval.png    - Generalization comparison")
        print("  05_neuron_weights.png        - Learned patterns (64 neurons)")
        print("  06_activations_by_digit.png  - How digits activate neurons")
        print("  07_confusion_matrix.png      - Error analysis")
        print("  08_confidence_analysis.png   - Confidence vs accuracy")
    
    print()


if __name__ == "__main__":
    main()
