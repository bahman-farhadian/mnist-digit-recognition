"""
Training Module for Neural Network Digit Recognition

This module handles:
- Training on MNIST dataset (60,000 images)
- Testing on MNIST test set (same-distribution evaluation)
- Testing on EMNIST-Digits (cross-dataset generalization test)

The training loop uses:
- Cross-Entropy Loss: Standard loss for classification
- Adam Optimizer: Adaptive learning rate optimizer
- Mini-batch Gradient Descent: Process batches of 64 images at a time
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path


class Trainer:
    """
    Trains the neural network and tracks performance metrics.
    
    Key metrics tracked:
    - Training loss and accuracy (how well we fit the training data)
    - MNIST test accuracy (same-distribution generalization)
    - EMNIST test accuracy (cross-dataset generalization)
    
    The gap between MNIST and EMNIST accuracy indicates how well
    the model learned general digit features vs. MNIST-specific patterns.
    """
    
    def __init__(self, model, data, learning_rate: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            model: Neural network to train
            data: DataManager with training and test data
            learning_rate: Step size for optimizer (default 0.001 works well for Adam)
        """
        self.model = model
        self.data = data
        self.learning_rate = learning_rate
        
        # Use GPU if available for faster training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Cross-Entropy Loss: Perfect for multi-class classification
        # Combines softmax + negative log likelihood
        self.criterion = nn.CrossEntropyLoss()
        
        # Adam Optimizer: Adapts learning rate per-parameter
        # Generally works better than plain SGD for deep learning
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # History for plotting training curves
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'mnist_test_accuracy': [],
            'emnist_test_accuracy': []
        }
        
        print(f"Training on: {self.device}")
    
    def train(self, epochs: int = 5):
        """
        Train the model for specified number of epochs.
        
        Each epoch:
        1. Processes all training batches (forward + backward pass)
        2. Evaluates on MNIST test set
        3. Evaluates on EMNIST test set (cross-dataset)
        
        Args:
            epochs: Number of complete passes through training data
        """
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            # =================================================================
            # Training Phase
            # =================================================================
            self.model.train()  # Enable dropout, batch norm training mode
            total_loss = 0
            correct = 0
            total = 0
            
            # Progress bar for this epoch
            pbar = tqdm(self.data.train_loader, desc=f"Epoch {epoch}/{epochs}")
            
            for images, labels in pbar:
                # Move data to GPU if available
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass: compute predictions
                self.optimizer.zero_grad()  # Clear gradients from last batch
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass: compute gradients
                loss.backward()
                
                # Update weights using gradients
                self.optimizer.step()
                
                # Track metrics for this batch
                total_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
            
            # =================================================================
            # Evaluation Phase
            # =================================================================
            train_loss = total_loss / len(self.data.train_loader)
            train_acc = 100. * correct / total
            
            # Test on MNIST (same distribution as training)
            mnist_acc = self.evaluate(self.data.mnist_test_loader)
            
            # Test on EMNIST (cross-dataset evaluation)
            if self.data.emnist_test_loader is not None:
                emnist_acc = self.evaluate(self.data.emnist_test_loader)
            else:
                emnist_acc = 0.0
            
            # =================================================================
            # Record History
            # =================================================================
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['mnist_test_accuracy'].append(mnist_acc)
            self.history['emnist_test_accuracy'].append(emnist_acc)
            
            # Print epoch summary
            if emnist_acc > 0:
                print(f"  → Train: {train_acc:.2f}% | MNIST Test: {mnist_acc:.2f}% | EMNIST Test: {emnist_acc:.2f}%")
            else:
                print(f"  → Train: {train_acc:.2f}% | MNIST Test: {mnist_acc:.2f}%")
        
        # Training complete summary
        print("=" * 60)
        print("Training complete!")
        print(f"  Final MNIST accuracy:  {self.history['mnist_test_accuracy'][-1]:.2f}%")
        if self.history['emnist_test_accuracy'][-1] > 0:
            print(f"  Final EMNIST accuracy: {self.history['emnist_test_accuracy'][-1]:.2f}%")
    
    def evaluate(self, loader) -> float:
        """
        Evaluate model accuracy on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Accuracy as percentage (0-100)
        """
        self.model.eval()  # Disable dropout, use running stats for batch norm
        correct = 0
        total = 0
        
        with torch.no_grad():  # Don't compute gradients during evaluation
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(dim=1)
                
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100. * correct / total
    
    def get_predictions(self, loader, n: int = 48):
        """
        Get model predictions for visualization.
        
        Args:
            loader: DataLoader to get predictions from
            n: Number of samples to return
            
        Returns:
            Tuple of (images, labels, predictions, probabilities)
        """
        self.model.eval()
        all_images = []
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(dim=1)
                
                all_images.extend(images.cpu())
                all_labels.extend(labels)
                all_preds.extend(preds.cpu())
                all_probs.extend(probs.cpu())
                
                if len(all_images) >= n:
                    break
        
        return (
            torch.stack(all_images[:n]),
            torch.tensor(all_labels[:n]),
            torch.tensor(all_preds[:n]),
            torch.stack(all_probs[:n])
        )
    
    def get_confusion_matrix(self, loader) -> torch.Tensor:
        """
        Compute confusion matrix for error analysis.
        
        Returns:
            10x10 tensor where [i,j] = count of true class i predicted as j
        """
        self.model.eval()
        confusion = torch.zeros(10, 10, dtype=torch.long)
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(dim=1)
                
                for true_label, pred_label in zip(labels, preds.cpu()):
                    confusion[true_label.item(), pred_label.item()] += 1
        
        return confusion
    
    def stability_test(self, num_runs: int = 32) -> dict:
        """
        Run stability test on EMNIST with dropout enabled.
        
        This tests model stability by running inference multiple times
        with dropout active (model.train() mode). Variance in results
        indicates how sensitive the model is to dropout randomness.
        
        A stable model should have low variance across runs.
        
        Args:
            num_runs: Number of test runs (default: 32)
            
        Returns:
            Dictionary with accuracies list and statistics
        """
        if self.data.emnist_test_loader is None:
            print("  EMNIST not available for stability test")
            return {'accuracies': [], 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        print(f"\nRunning stability test ({num_runs} runs with dropout enabled)...")
        
        accuracies = []
        
        for run in range(num_runs):
            # Enable training mode to activate dropout
            self.model.train()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.data.emnist_test_loader:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(dim=1)
                    correct += predicted.eq(labels.to(self.device)).sum().item()
                    total += labels.size(0)
            
            accuracy = 100. * correct / total
            accuracies.append(accuracy)
            
            # Progress indicator
            if (run + 1) % 10 == 0 or run == 0:
                print(f"  Run {run + 1}/{num_runs}: {accuracy:.2f}%")
        
        # Back to eval mode
        self.model.eval()
        
        # Calculate statistics
        import numpy as np
        accuracies_np = np.array(accuracies)
        
        results = {
            'accuracies': accuracies,
            'mean': float(accuracies_np.mean()),
            'std': float(accuracies_np.std()),
            'min': float(accuracies_np.min()),
            'max': float(accuracies_np.max())
        }
        
        print(f"\nStability Test Results:")
        print(f"  Mean:  {results['mean']:.2f}%")
        print(f"  Std:   {results['std']:.2f}%")
        print(f"  Range: {results['min']:.2f}% - {results['max']:.2f}%")
        
        return results
    
    def save_model(self, path: str):
        """
        Save trained model weights.
        
        Args:
            path: File path to save to (typically .pt or .pth)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load pre-trained model weights.
        
        Args:
            path: File path to load from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


# =============================================================================
# Module-level test
# =============================================================================
if __name__ == "__main__":
    print("Testing Trainer...")
    print()
    
    from .model import DigitRecognizer
    from .data import DataManager
    
    # Quick test with CNN and 1 epoch
    model = DigitRecognizer()
    data = DataManager(batch_size=64)
    trainer = Trainer(model, data)
    trainer.train(epochs=1)
    
    print(f"\nHistory keys: {trainer.history.keys()}")
    print("\n✓ Trainer test complete!")
