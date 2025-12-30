"""
Visualization Module for CNN Analysis

Creates professional, portfolio-ready visualizations showing:
- Training progress (loss and accuracy curves)
- Sample data from both MNIST and EMNIST datasets
- Model predictions with confidence scores
- Cross-dataset performance comparison
- Convolutional filter patterns
- Feature maps visualization
- Confusion matrices with detailed annotations
- Confidence calibration analysis
- Stability test results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .trainer import Trainer


# Professional style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class Visualizer:
    """
    Creates professional visualizations for CNN analysis.
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
        
        # Color palette
        self.colors = {
            'primary': '#2563eb',      # Blue
            'secondary': '#dc2626',    # Red
            'success': '#16a34a',      # Green
            'warning': '#ea580c',      # Orange
            'mnist': '#3b82f6',        # Light blue
            'emnist': '#f97316',       # Orange
        }
    
    def _save(self, filename: str, dpi: int = 150):
        """Save figure and close it."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.close()
        print(f"  Saved: {path}")
    
    def plot_training_history(self, filename: str = "01_training_history.png"):
        """Plot training loss and accuracy over epochs."""
        h = self.trainer.history
        epochs = h['epoch']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(epochs, h['train_loss'], color=self.colors['primary'], 
                    linewidth=2.5, marker='o', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_xlim(0, max(epochs) + 1)
        
        # Accuracy curves
        axes[1].plot(epochs, h['train_accuracy'], color=self.colors['success'], 
                    linewidth=2.5, marker='o', markersize=4, label='Train (MNIST)')
        axes[1].plot(epochs, h['mnist_test_accuracy'], color=self.colors['mnist'],
                    linewidth=2.5, marker='s', markersize=4, label='Test (MNIST)')
        if h['emnist_test_accuracy'] and h['emnist_test_accuracy'][0] > 0:
            axes[1].plot(epochs, h['emnist_test_accuracy'], color=self.colors['emnist'],
                        linewidth=2.5, marker='^', markersize=4, label='Test (EMNIST)')
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Accuracy Over Training', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xlim(0, max(epochs) + 1)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_sample_data(self, filename: str = "02_sample_data.png"):
        """Show sample images from MNIST and EMNIST datasets."""
        fig, axes = plt.subplots(2, 10, figsize=(16, 4))
        
        # MNIST samples (top row)
        mnist_imgs, mnist_lbls = self.data.get_sample_batch("mnist_train", 10)
        for i in range(10):
            axes[0, i].imshow(mnist_imgs[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'{mnist_lbls[i].item()}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
        
        # Row label
        axes[0, 0].text(-0.5, 0.5, 'MNIST\n(Train)', transform=axes[0, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right',
                       color=self.colors['mnist'])
        
        # EMNIST samples (bottom row)
        if self.data.emnist_test_loader is not None:
            emnist_imgs, emnist_lbls = self.data.get_sample_batch("emnist_test", 10)
            for i in range(10):
                axes[1, i].imshow(emnist_imgs[i].squeeze(), cmap='gray')
                axes[1, i].set_title(f'{emnist_lbls[i].item()}', fontsize=12, fontweight='bold')
                axes[1, i].axis('off')
            
            axes[1, 0].text(-0.5, 0.5, 'EMNIST\n(Test)', transform=axes[1, 0].transAxes,
                           fontsize=12, fontweight='bold', va='center', ha='right',
                           color=self.colors['emnist'])
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_predictions(self, filename: str = "03_predictions.png"):
        """Show model predictions on test set with confidence scores."""
        images, labels, preds, probs = self.trainer.get_predictions(
            self.data.mnist_test_loader, 48
        )
        
        fig, axes = plt.subplots(6, 8, figsize=(16, 12))
        
        for i in range(48):
            ax = axes[i // 8, i % 8]
            image = images[i].squeeze()
            true_label = labels[i].item()
            pred_label = preds[i].item()
            confidence = probs[i][pred_label].item() * 100
            
            ax.imshow(image, cmap='gray')
            
            if true_label == pred_label:
                color = self.colors['success']
                title = f'{pred_label} ✓\n{confidence:.0f}%'
            else:
                color = self.colors['secondary']
                title = f'{pred_label} ✗ ({true_label})\n{confidence:.0f}%'
            
            ax.set_title(title, color=color, fontsize=9, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_cross_dataset_comparison(self, filename: str = "04_cross_dataset_evaluation.png"):
        """Compare performance on MNIST vs EMNIST with professional styling."""
        h = self.trainer.history
        
        if not h['emnist_test_accuracy'] or h['emnist_test_accuracy'][0] == 0:
            print("  Skipped: EMNIST not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        mnist_acc = h['mnist_test_accuracy'][-1]
        emnist_acc = h['emnist_test_accuracy'][-1]
        
        bars = axes[0].bar(
            ['MNIST\n(Same Distribution)', 'EMNIST\n(Different Writers)'],
            [mnist_acc, emnist_acc],
            color=[self.colors['mnist'], self.colors['emnist']],
            width=0.5,
            edgecolor='black',
            linewidth=1.5
        )
        
        axes[0].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        axes[0].set_title('Final Test Accuracy', fontsize=14, fontweight='bold', pad=40)
        axes[0].set_ylim(0, 115)  # More room for labels
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Add value labels on bars - inside the bars for cleaner look
        for bar, acc in zip(bars, [mnist_acc, emnist_acc]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                        f'{acc:.1f}%', ha='center', va='top', 
                        fontsize=20, fontweight='bold', color='white')
        
        # Hide y-axis above 100
        axes[0].set_yticks([0, 20, 40, 60, 80, 100])
        
        # Training curve
        epochs = h['epoch']
        axes[1].plot(epochs, h['mnist_test_accuracy'], color=self.colors['mnist'],
                    linewidth=2.5, marker='s', markersize=5, label=f"MNIST Test ({mnist_acc:.1f}%)")
        axes[1].plot(epochs, h['emnist_test_accuracy'], color=self.colors['emnist'],
                    linewidth=2.5, marker='^', markersize=5, label=f"EMNIST Test ({emnist_acc:.1f}%)")
        
        axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        axes[1].set_title('Accuracy Progression', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xlim(0, max(epochs) + 1)
        
        plt.tight_layout()
        self._save(filename)
    
    def plot_conv_filters(self, filename: str = "05_conv_filters.png"):
        """Visualize convolutional filters with professional styling."""
        filters = self.model.get_conv_filters(layer=1)
        num_filters = filters.shape[0]
        
        cols = 8
        rows = (num_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
        
        vmax = filters.abs().max().item()
        
        for i in range(num_filters):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            filt = filters[i, 0].numpy()
            im = ax.imshow(filt, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'F{i}', fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Weight Value', fontsize=11)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        self._save(filename)
    
    def plot_feature_maps(self, filename: str = "06_feature_maps.png"):
        """Show feature maps with improved layout."""
        images, labels = self.data.get_sample_batch("mnist_test", 1)
        image = images[0:1].to(self.trainer.device)
        label = labels[0].item()
        
        self.model.eval()
        with torch.no_grad():
            fm1, fm2 = self.model.get_feature_maps(image)
        
        fig = plt.figure(figsize=(16, 10))
        
        # Original image
        ax_input = fig.add_axes([0.02, 0.72, 0.12, 0.22])
        ax_input.imshow(images[0].squeeze(), cmap='gray')
        ax_input.set_title(f'Input: {label}', fontsize=12, fontweight='bold')
        ax_input.axis('off')
        
        # Layer 1 title
        ax_label1 = fig.add_axes([0.2, 0.90, 0.6, 0.05])
        ax_label1.text(0.5, 0.5, 'Layer 1 Feature Maps (16 of 32)',
                      ha='center', va='center', fontsize=13, fontweight='bold')
        ax_label1.axis('off')
        
        # Layer 1 feature maps
        fm1_grid = fm1[0, :16].numpy()
        grid_img1 = np.zeros((2 * 28, 8 * 28))
        for i in range(16):
            r, c = i // 8, i % 8
            grid_img1[r*28:(r+1)*28, c*28:(c+1)*28] = fm1_grid[i]
        
        ax_fm1 = fig.add_axes([0.1, 0.50, 0.8, 0.38])
        ax_fm1.imshow(grid_img1, cmap='hot')
        ax_fm1.axis('off')
        
        # Layer 2 title
        ax_label2 = fig.add_axes([0.2, 0.42, 0.6, 0.05])
        ax_label2.text(0.5, 0.5, 'Layer 2 Feature Maps (16 of 64)',
                      ha='center', va='center', fontsize=13, fontweight='bold')
        ax_label2.axis('off')
        
        # Layer 2 feature maps
        fm2_grid = fm2[0, :16].numpy()
        grid_img2 = np.zeros((2 * 14, 8 * 14))
        for i in range(16):
            r, c = i // 8, i % 8
            grid_img2[r*14:(r+1)*14, c*14:(c+1)*14] = fm2_grid[i]
        
        ax_fm2 = fig.add_axes([0.1, 0.05, 0.8, 0.35])
        ax_fm2.imshow(grid_img2, cmap='hot')
        ax_fm2.axis('off')
        
        self._save(filename)
    
    def plot_confusion_matrix(self, filename: str = "07_confusion_matrix.png"):
        """Show confusion matrices with numbers and better styling."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # MNIST confusion matrix
        cm_mnist = self.trainer.get_confusion_matrix(self.data.mnist_test_loader).numpy()
        self._plot_cm(axes[0], cm_mnist, 'MNIST Test', 'Blues')
        
        # EMNIST confusion matrix
        if self.data.emnist_test_loader is not None:
            cm_emnist = self.trainer.get_confusion_matrix(self.data.emnist_test_loader).numpy()
            self._plot_cm(axes[1], cm_emnist, 'EMNIST Test', 'Oranges')
        else:
            axes[1].text(0.5, 0.5, 'EMNIST Not Available', 
                        ha='center', va='center', fontsize=14)
            axes[1].axis('off')
        
        plt.tight_layout()
        self._save(filename)
    
    def _plot_cm(self, ax, cm, title, cmap):
        """Helper to plot a single confusion matrix with annotations."""
        # Normalize for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax.imshow(cm_normalized, cmap=cmap, vmin=0, vmax=100)
        
        # Add text annotations
        thresh = 50
        for i in range(10):
            for j in range(10):
                value = cm_normalized[i, j]
                count = cm[i, j]
                color = 'white' if value > thresh else 'black'
                if count > 0:
                    ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy %', fontsize=10)
    
    def plot_confidence_analysis(self, filename: str = "08_confidence_analysis.png"):
        """Analyze confidence distribution with professional styling."""
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
                    label='Correct', color=self.colors['success'], edgecolor='darkgreen')
        axes[0].hist(all_probs[~all_correct], bins=50, alpha=0.7,
                    label='Incorrect', color=self.colors['secondary'], edgecolor='darkred')
        axes[0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11, frameon=True, fancybox=True)
        axes[0].set_xlim(0.3, 1.02)
        
        # Calibration curve
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
        
        axes[1].bar(bin_centers, accuracies, width=0.08, color=self.colors['primary'], 
                   alpha=0.8, edgecolor='darkblue', linewidth=1.5)
        axes[1].plot([0, 1], [0, 100], 'k--', alpha=0.5, label='Perfect Calibration')
        axes[1].set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Confidence Calibration', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0.3, 1.05)
        axes[1].set_ylim(0, 105)
        axes[1].legend(fontsize=10, loc='lower right')
        
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
        self.plot_conv_filters()
        self.plot_feature_maps()
        self.plot_confusion_matrix()
        self.plot_confidence_analysis()
        
        print("-" * 50)
        print(f"All visualizations saved to {self.output_dir}/")
    
    def plot_stability_results(self, stability_results: dict):
        """Generate professional stability visualization."""
        accuracies = stability_results['accuracies']
        if not accuracies:
            print("  Skipped: No stability data")
            return
        
        import numpy as np
        accuracies = np.array(accuracies)
        num_runs = len(accuracies)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Line graph
        runs = np.arange(1, num_runs + 1)
        axes[0].plot(runs, accuracies, color=self.colors['primary'], linewidth=2,
                    marker='o', markersize=5, alpha=0.8)
        axes[0].axhline(y=stability_results['mean'], color=self.colors['secondary'],
                       linestyle='--', linewidth=2.5, label=f"Mean: {stability_results['mean']:.2f}%")
        axes[0].fill_between(runs, 
                            stability_results['mean'] - stability_results['std'],
                            stability_results['mean'] + stability_results['std'],
                            alpha=0.2, color=self.colors['secondary'],
                            label=f"±1 Std: {stability_results['std']:.2f}%")
        
        axes[0].set_xlabel('Test Run', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('EMNIST Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Stability Test ({num_runs} Runs)', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=11, frameon=True, fancybox=True)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_xlim(0, num_runs + 1)
        
        # Right: Box plot
        box = axes[1].boxplot(accuracies, patch_artist=True, widths=0.5)
        box['boxes'][0].set_facecolor(self.colors['primary'])
        box['boxes'][0].set_alpha(0.7)
        box['medians'][0].set_color(self.colors['secondary'])
        box['medians'][0].set_linewidth(3)
        
        # Scatter points
        x_jitter = np.random.normal(1, 0.04, size=len(accuracies))
        axes[1].scatter(x_jitter, accuracies, alpha=0.6, color='navy', s=30, zorder=3)
        
        # Stats box - positioned inside plot area
        stats_text = (
            f"Mean:  {stability_results['mean']:.2f}%\n"
            f"Std:   {stability_results['std']:.2f}%\n"
            f"Min:   {stability_results['min']:.2f}%\n"
            f"Max:   {stability_results['max']:.2f}%"
        )
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                             edgecolor='orange', linewidth=2, alpha=0.95))
        
        axes[1].set_ylabel('EMNIST Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(['EMNIST Test'], fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        self._save("09_stability_test.png")
        
        # Print interpretation
        std = stability_results['std']
        if std < 0.1:
            stability = "Excellent"
        elif std < 0.5:
            stability = "Very Good"
        elif std < 1.0:
            stability = "Good"
        else:
            stability = "Moderate"
        
        print(f"  Stability: {stability} (std={std:.3f}%)")


if __name__ == "__main__":
    print("Visualizer module - run main.py to generate visualizations")
