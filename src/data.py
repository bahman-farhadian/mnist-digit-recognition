"""
Data Loading Module for MNIST and EMNIST Cross-Dataset Evaluation

This module demonstrates cross-dataset generalization:
- TRAIN on MNIST (60,000 images)
- TEST on MNIST test set (10,000 images) - same distribution
- TEST on EMNIST-Digits test set (40,000 images) - cross-dataset evaluation

Why EMNIST?
- EMNIST-Digits comes from the same NIST database as MNIST
- Same visual style (28×28, centered, grayscale)
- BUT different writers than MNIST
- This tests whether model learned general digit features vs. memorizing MNIST patterns

IMPORTANT: EMNIST images are transposed compared to MNIST!
We apply PIL's TRANSPOSE to fix the orientation.

Expected results:
- MNIST Test: 97-98% (trained on this distribution)
- EMNIST Test: 85-95% (same style, different writers)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image


class FixEMNISTOrientation:
    """
    Fix EMNIST orientation at PIL level before ToTensor.
    
    EMNIST images are transposed compared to MNIST.
    Using PIL's transpose is the correct fix.
    """
    def __call__(self, img):
        return img.transpose(Image.TRANSPOSE)


class DataManager:
    """
    Manages MNIST (training) and EMNIST-Digits (cross-dataset testing).
    
    Dataset sizes:
        MNIST Train:  60,000 images (what we train on)
        MNIST Test:   10,000 images (same-distribution test)
        EMNIST Test:  40,000 images (cross-dataset evaluation)
    
    The model never sees EMNIST during training - good EMNIST accuracy
    proves the model learned generalizable digit recognition.
    """
    
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        """
        Initialize data manager.
        
        Args:
            data_dir: Directory to store/load datasets
            batch_size: Number of samples per batch
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Only use pin_memory if CUDA is available
        self.use_pin_memory = torch.cuda.is_available()
        
        # Training transform with data augmentation
        # Augmentation helps the model generalize to different writing styles
        self.mnist_train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # ±10 degrees rotation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # ±10% shift
                scale=(0.9, 1.1),  # ±10% scale
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Standard preprocessing for MNIST test (no augmentation)
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # EMNIST transform: fix orientation BEFORE ToTensor, then same as MNIST
        # EMNIST images are transposed compared to MNIST
        self.emnist_transform = transforms.Compose([
            FixEMNISTOrientation(),  # Fix orientation at PIL level
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Same normalization
        ])
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load MNIST for training and EMNIST for cross-dataset testing."""
        
        print("Loading datasets...")
        print("-" * 40)
        
        # =====================================================================
        # MNIST - Our training dataset
        # =====================================================================
        print("  MNIST (training dataset):")
        
        self.mnist_train = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.mnist_train_transform  # Use augmented transform
        )
        
        self.mnist_test = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.mnist_transform  # No augmentation for test
        )
        
        print(f"    Train: {len(self.mnist_train):,} images")
        print(f"    Test:  {len(self.mnist_test):,} images")
        
        # =====================================================================
        # EMNIST-Digits - Cross-dataset test (different writers)
        # =====================================================================
        print("  EMNIST-Digits (cross-dataset test):")
        
        try:
            self.emnist_test = datasets.EMNIST(
                root=self.data_dir,
                split='digits',
                train=False,
                download=True,
                transform=self.emnist_transform  # Use EMNIST-specific transform!
            )
            print(f"    Test:  {len(self.emnist_test):,} images")
            self._emnist_available = True
        except Exception as e:
            print(f"    Warning: Could not load EMNIST ({e})")
            print(f"    Cross-dataset evaluation will be skipped.")
            self.emnist_test = None
            self._emnist_available = False
        
        # =====================================================================
        # Create DataLoaders
        # =====================================================================
        
        # Training loader - shuffle for better gradient estimates
        self.train_loader = DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.use_pin_memory,
            num_workers=0
        )
        
        # MNIST test loader - no shuffle needed for evaluation
        self.mnist_test_loader = DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.use_pin_memory,
            num_workers=0
        )
        
        # EMNIST test loader - cross-dataset evaluation
        if self._emnist_available:
            self.emnist_test_loader = DataLoader(
                self.emnist_test,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=self.use_pin_memory,
                num_workers=0
            )
        else:
            self.emnist_test_loader = None
        
        print("-" * 40)
        print(f"  Training batches: {len(self.train_loader)}")
        print()
    
    def get_sample_batch(self, dataset_name: str, n: int = 10):
        """
        Get sample images for visualization.
        
        Args:
            dataset_name: One of 'mnist_train', 'mnist_test', 'emnist_test'
            n: Number of samples to return
            
        Returns:
            Tuple of (images, labels) tensors
        """
        if dataset_name == "mnist_train":
            loader = self.train_loader
        elif dataset_name == "mnist_test":
            loader = self.mnist_test_loader
        elif dataset_name == "emnist_test" and self.emnist_test_loader is not None:
            loader = self.emnist_test_loader
        else:
            loader = self.mnist_test_loader
        
        images, labels = next(iter(loader))
        return images[:n], labels[:n]
    
    def get_samples_by_digit(self, digit: int, num_samples: int = 1, dataset: str = "mnist"):
        """
        Get samples of a specific digit class.
        
        Args:
            digit: Digit class (0-9)
            num_samples: Number of samples to return
            dataset: 'mnist' or 'emnist'
            
        Returns:
            Tuple of (images, labels) tensors
        """
        if dataset == "emnist" and self._emnist_available:
            data = self.emnist_test
        else:
            data = self.mnist_train
        
        samples = []
        labels = []
        
        for img, lbl in data:
            if lbl == digit:
                samples.append(img)
                labels.append(lbl)
                if len(samples) >= num_samples:
                    break
        
        if len(samples) == 0:
            # Fallback: return any sample
            img, lbl = data[0]
            samples.append(img)
            labels.append(lbl)
        
        return torch.stack(samples), torch.tensor(labels)
    
    def __repr__(self) -> str:
        """String representation showing dataset sizes."""
        emnist_info = (f"  Cross-Dataset Test: EMNIST-Digits ({len(self.emnist_test):,} images)\n"
                      if self._emnist_available else "  Cross-Dataset Test: Not available\n")
        
        return (
            f"DataManager(\n"
            f"  Training: MNIST ({len(self.mnist_train):,} images)\n"
            f"  Same-Distribution Test: MNIST ({len(self.mnist_test):,} images)\n"
            f"{emnist_info}"
            f"  Batch size: {self.batch_size}\n"
            f")"
        )


# =============================================================================
# Module-level test
# =============================================================================
if __name__ == "__main__":
    print("Testing DataManager...")
    print()
    
    data = DataManager()
    print(data)
    
    # Test sample retrieval
    imgs, lbls = data.get_sample_batch("mnist_train", 5)
    print(f"\nMNIST sample batch: {imgs.shape}, labels: {lbls.tolist()}")
    
    if data.emnist_test_loader is not None:
        imgs, lbls = data.get_sample_batch("emnist_test", 5)
        print(f"EMNIST sample batch: {imgs.shape}, labels: {lbls.tolist()}")
    
    print("\n✓ DataManager test complete!")
