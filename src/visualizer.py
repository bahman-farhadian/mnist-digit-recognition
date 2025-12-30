"""
Visualization Module for Neural Network Analysis

Creates educational visualizations showing:
- Training progress (loss and accuracy curves)
- Sample data from both MNIST and EMNIST datasets
- Model predictions with confidence scores
- Cross-dataset performance comparison
- What patterns neurons have learned
- Confusion matrices for error analysis
- Confidence calibration analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .trainer import Trainer


class Visualizer:
    """
    Creates visualizations for understanding neural network behavior.
    
    All visualizations are saved as high-quality PNG images.
    """
    
    def __init__(self, trainer: Trainer, output_dir: str = "outputs"):
        """
        Initialize visualizer.
        
        Args:
            trainer: Trained Trainer instance
            output_dir: Directory to save visualizations
        """
        self.trainer = trainer
        self.model = trainer.model
        self.data = trainer.data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean plot style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    def _save(self, filename: str, dpi: int = 150):
        """Save figure and close it."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")
    
    def plot_training_history(self, filename: str = "01_training_history.png"):
        """Plot training loss and accuracy over epochs."""
        h = self.trainer.history
        epochs = h['epoch']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(epochs, h['train_loss'], 'b-o', linewidth=2, markersize=6)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12)
        axes[0].set_title('Training Loss Over Time\n(Lower = Better)', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, h['train_accuracy'], 'g-o', linewidth=2, 
                    markersize=6, label='Train (MNIST)')
        axes[1].plot(epochs, h['mnist_test_accuracy'], 'b-s', linewidth=2,
                    markersize=6, label='Test (MNIST)')
        if h['emnist_test_accuracy'] and h['emnist_test_accuracy'][0] > 0:
            axes[1].plot(epochs, h['emnist_test_accuracy'], 'r-^', linewidth=2,
                        markersize=6, label='Test (EMNIST)')
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy Over Training\n(Higher = Better)', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_sample_data(self, filename: str = "02_sample_data.png"):
        """Show sample images from MNIST and EMNIST datasets."""
        fig, axes = plt.subplots(2, 10, figsize=(16, 4))
        fig.suptitle('Sample Digits: MNIST (Training) vs EMNIST (Cross-Dataset Test)', 
                    fontsize=14, y=1.02)
        
        # MNIST samples (top row)
        mnist_imgs, mnist_lbls = self.data.get_sample_batch("mnist_train", 10)
        for i in range(10):
            axes[0, i].imshow(mnist_imgs[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'{mnist_lbls[i].item()}', fontsize=11)
            axes[0, i].axis('off')
        axes[0, 0].set_ylabel('MNIST', fontsize=12, rotation=0, ha='right', va='center')
        
        # EMNIST samples (bottom row)
        if self.data.emnist_test_loader is not None:
            emnist_imgs, emnist_lbls = self.data.get_sample_batch("emnist_test", 10)
            for i in range(10):
                axes[1, i].imshow(emnist_imgs[i].squeeze(), cmap='gray')
                axes[1, i].set_title(f'{emnist_lbls[i].item()}', fontsize=11)
                axes[1, i].axis('off')
            axes[1, 0].set_ylabel('EMNIST', fontsize=12, rotation=0, ha='right', va='center')
        else:
            for i in range(10):
                axes[1, i].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                axes[1, i].axis('off')
            axes[1, 0].set_ylabel('EMNIST', fontsize=12, rotation=0, ha='right', va='center')
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_predictions(self, filename: str = "03_predictions.png"):
        """Show model predictions on MNIST test set with confidence scores."""
        images, labels, preds, probs = self.trainer.get_predictions(
            self.data.mnist_test_loader, 48
        )
        
        fig, axes = plt.subplots(6, 8, figsize=(16, 12))
        fig.suptitle('Model Predictions on MNIST Test Set\n'
                    '(Green ✓ = Correct, Red ✗ = Wrong)', fontsize=14)
        
        for i in range(48):
            ax = axes[i // 8, i % 8]
            image = images[i].squeeze()
            true_label = labels[i].item()
            pred_label = preds[i].item()
            confidence = probs[i][pred_label].item() * 100
            
            ax.imshow(image, cmap='gray')
            
            if true_label == pred_label:
                color = 'green'
                title = f'{pred_label} ✓\n{confidence:.0f}%'
            else:
                color = 'red'
                title = f'{pred_label} ✗ (was {true_label})\n{confidence:.0f}%'
            
            ax.set_title(title, color=color, fontsize=9)
            ax.axis('off')
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_cross_dataset_comparison(self, filename: str = "04_cross_dataset_evaluation.png"):
        """Compare performance on MNIST vs EMNIST (cross-dataset)."""
        h = self.trainer.history
        
        if not h['emnist_test_accuracy'] or h['emnist_test_accuracy'][0] == 0:
            print("  Skipped: EMNIST not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of final accuracies
        mnist_acc = h['mnist_test_accuracy'][-1]
        emnist_acc = h['emnist_test_accuracy'][-1]
        
        bars = axes[0].bar(
            ['MNIST Test\n(Same Distribution)', 'EMNIST Test\n(Different Writers)'],
            [mnist_acc, emnist_acc],
            color=['steelblue', 'coral'],
            width=0.6
        )
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Cross-Dataset Generalization\n(Trained on MNIST Only)', fontsize=14)
        axes[0].set_ylim(0, 105)
        
        # Add value labels
        for bar, acc in zip(bars, [mnist_acc, emnist_acc]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
        
        # Accuracy over epochs
        epochs = h['epoch']
        axes[1].plot(epochs, h['mnist_test_accuracy'], 'b-s', linewidth=2,
                    markersize=6, label='MNIST Test')
        axes[1].plot(epochs, h['emnist_test_accuracy'], 'r-^', linewidth=2,
                    markersize=6, label='EMNIST Test')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Test Accuracy Over Training', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_neuron_weights(self, filename: str = "05_neuron_weights.png", num_neurons: int = 64):
        """
        Visualize what patterns hidden neurons learned to detect.
        
        Each 28×28 image shows one neuron's input weights:
        - Red pixels: Activate the neuron (positive weights)
        - Blue pixels: Inhibit the neuron (negative weights)
        """
        weights = self.model.get_hidden_weights()  # [hidden_size, 784]
        num_neurons = min(num_neurons, weights.shape[0])
        
        cols = 8
        rows = (num_neurons + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
        fig.suptitle(
            f'Hidden Neuron Weight Patterns ({num_neurons} of {self.model.hidden_size})\n'
            'Red = Activating, Blue = Inhibiting',
            fontsize=12
        )
        
        vmax = weights[:num_neurons].abs().max().item() * 0.7
        
        for i in range(num_neurons):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            pattern = weights[i].reshape(28, 28).numpy()
            ax.imshow(pattern, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'N{i}', fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_neurons, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_activations_by_digit(self, filename: str = "06_activations_by_digit.png"):
        """Show how different digits activate different neurons."""
        fig = plt.figure(figsize=(16, 10))
        
        # Show one sample of each digit
        for digit in range(10):
            ax = fig.add_subplot(3, 10, digit + 1)
            image, _ = self.data.get_samples_by_digit(digit, num_samples=1)
            ax.imshow(image[0].squeeze(), cmap='gray')
            ax.set_title(f'{digit}', fontsize=12)
            ax.axis('off')
        
        # Collect activations
        activations = []
        self.model.eval()
        with torch.no_grad():
            for digit in range(10):
                image, _ = self.data.get_samples_by_digit(digit, num_samples=1)
                image = image.to(self.trainer.device)
                acts = self.model.get_hidden_activations(image)
                activations.append(acts.cpu().squeeze().numpy())
        
        activations = np.array(activations)  # [10, hidden_size]
        
        # Heatmap
        ax = fig.add_subplot(3, 1, 2)
        im = ax.imshow(activations, cmap='hot', aspect='auto')
        ax.set_xlabel('Neuron Index', fontsize=11)
        ax.set_ylabel('Digit', fontsize=11)
        ax.set_yticks(range(10))
        ax.set_title('Neuron Activation Patterns\n(Yellow = Strong Activation)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Activation')
        
        # Average activation
        ax = fig.add_subplot(3, 1, 3)
        mean_act = activations.mean(axis=0)
        ax.bar(range(len(mean_act)), mean_act, color='steelblue', alpha=0.7)
        ax.set_xlabel('Neuron Index', fontsize=11)
        ax.set_ylabel('Average Activation', fontsize=11)
        ax.set_title('Neuron Utilization', fontsize=12)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_confusion_matrix(self, filename: str = "07_confusion_matrix.png"):
        """Show confusion matrices for MNIST and EMNIST."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MNIST confusion matrix
        cm_mnist = self.trainer.get_confusion_matrix(self.data.mnist_test_loader)
        im1 = axes[0].imshow(cm_mnist.numpy(), cmap='Blues')
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('True', fontsize=11)
        axes[0].set_title('Confusion Matrix (MNIST Test)', fontsize=12)
        axes[0].set_xticks(range(10))
        axes[0].set_yticks(range(10))
        plt.colorbar(im1, ax=axes[0], label='Count')
        
        # EMNIST confusion matrix
        if self.data.emnist_test_loader is not None:
            cm_emnist = self.trainer.get_confusion_matrix(self.data.emnist_test_loader)
            im2 = axes[1].imshow(cm_emnist.numpy(), cmap='Oranges')
            axes[1].set_xlabel('Predicted', fontsize=11)
            axes[1].set_ylabel('True', fontsize=11)
            axes[1].set_title('Confusion Matrix (EMNIST Test)', fontsize=12)
            axes[1].set_xticks(range(10))
            axes[1].set_yticks(range(10))
            plt.colorbar(im2, ax=axes[1], label='Count')
        else:
            axes[1].text(0.5, 0.5, 'EMNIST Not Available', 
                        ha='center', va='center', fontsize=14)
            axes[1].axis('off')
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_confidence_analysis(self, filename: str = "08_confidence_analysis.png"):
        """Analyze relationship between confidence and accuracy."""
        all_probs = []
        all_correct = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.data.mnist_test_loader:
                images = images.to(self.trainer.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(dim=1)
                
                max_probs = probs.max(dim=1)[0].cpu().numpy()
                correct = (preds.cpu() == labels).numpy()
                
                all_probs.extend(max_probs)
                all_correct.extend(correct)
        
        all_probs = np.array(all_probs)
        all_correct = np.array(all_correct)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confidence distribution
        axes[0].hist(all_probs[all_correct], bins=50, alpha=0.7, 
                    label='Correct', color='green')
        axes[0].hist(all_probs[~all_correct], bins=50, alpha=0.7,
                    label='Incorrect', color='red')
        axes[0].set_xlabel('Confidence', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Confidence Distribution\n(Good: Green Right, Red Left)', fontsize=12)
        axes[0].legend(fontsize=11)
        
        # Accuracy by confidence bin
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(all_probs, bins) - 1
        accuracies = []
        bin_centers = []
        
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                acc = all_correct[mask].mean() * 100
                accuracies.append(acc)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[1].bar(bin_centers, accuracies, width=0.08, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Confidence Level', fontsize=11)
        axes[1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[1].set_title('Accuracy vs Confidence\n(Should Increase)', fontsize=12)
        axes[1].set_ylim(0, 105)
        
        plt.tight_layout()
        self._save(filename)
    
    def generate_all(self):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")
        print("-" * 50)
        
        self.plot_training_history()
        self.plot_sample_data()
        self.plot_predictions()
        self.plot_cross_dataset_comparison()
        self.plot_neuron_weights()
        self.plot_activations_by_digit()
        self.plot_confusion_matrix()
        self.plot_confidence_analysis()
        
        print("-" * 50)
        print(f"All visualizations saved to {self.output_dir}/")


if __name__ == "__main__":
    print("Testing Visualizer...")
    from .model import DigitRecognizer
    from .data import DataManager
    from .trainer import Trainer
    
    model = DigitRecognizer(hidden_size=128)
    data = DataManager(batch_size=64)
    trainer = Trainer(model, data)
    trainer.train(epochs=1)
    
    viz = Visualizer(trainer, output_dir="outputs")
    viz.generate_all()
    
    print("\n✓ Visualizer test complete!")
