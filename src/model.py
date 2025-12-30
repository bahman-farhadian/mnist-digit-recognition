"""
Convolutional Neural Network for Digit Recognition

CNNs are superior to MLPs for image tasks because they:
- Learn LOCAL features (edges, curves) that transfer across datasets
- Have translation invariance (digit position doesn't matter as much)
- Share weights across spatial locations (fewer parameters, less overfitting)

Architecture:
    Input (1×28×28)
        ↓
    Conv2d(32 filters, 3×3) + ReLU + MaxPool(2×2) → 32×14×14
        ↓
    Conv2d(64 filters, 3×3) + ReLU + MaxPool(2×2) → 64×7×7
        ↓
    Flatten → 3136
        ↓
    Linear(256) + ReLU + Dropout(0.5)
        ↓
    Linear(10) → Class scores

This architecture typically achieves:
- MNIST: 99%+
- EMNIST: 90-95% (cross-dataset)
"""

import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    """
    Convolutional Neural Network for digit classification.
    
    Uses two convolutional layers to learn hierarchical features:
    - First conv layer: detects simple features (edges, corners)
    - Second conv layer: combines simple features into complex patterns
    
    Attributes:
        conv1: First convolutional layer (1 → 32 channels)
        conv2: Second convolutional layer (32 → 64 channels)
        fc1: First fully-connected layer (3136 → 256)
        fc2: Output layer (256 → 10 classes)
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize CNN.
        
        Args:
            num_classes: Number of output classes (10 for digits)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        # padding=1 with kernel_size=3 preserves spatial dimensions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer - reduces spatial dimensions by 2x
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv1 + pool: 28×28 → 14×14
        # After conv2 + pool: 14×14 → 7×7
        # Flattened size: 64 channels × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → class scores.
        
        Args:
            x: Input tensor of shape [batch, 1, 28, 28]
            
        Returns:
            Class scores of shape [batch, 10]
        """
        # Conv block 1: [B, 1, 28, 28] → [B, 32, 14, 14]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2: [B, 32, 14, 14] → [B, 64, 7, 7]
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten: [B, 64, 7, 7] → [B, 3136]
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_conv_filters(self, layer: int = 1) -> torch.Tensor:
        """
        Get convolutional filter weights for visualization.
        
        Args:
            layer: Which conv layer (1 or 2)
            
        Returns:
            Filter weights tensor
        """
        if layer == 1:
            # Shape: [32, 1, 3, 3] - 32 filters, 1 input channel, 3×3 kernel
            return self.conv1.weight.detach().cpu()
        else:
            # Shape: [64, 32, 3, 3] - 64 filters, 32 input channels, 3×3 kernel
            return self.conv2.weight.detach().cpu()
    
    def get_feature_maps(self, x: torch.Tensor) -> tuple:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input image tensor
            
        Returns:
            Tuple of (conv1_output, conv2_output)
        """
        # Conv block 1
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        p1 = self.pool(c1)
        
        # Conv block 2
        c2 = self.conv2(p1)
        c2 = self.bn2(c2)
        c2 = self.relu(c2)
        
        return c1.detach().cpu(), c2.detach().cpu()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation showing architecture."""
        params = self.count_parameters()
        return (
            f"DigitRecognizer (CNN)\n"
            f"  Architecture:\n"
            f"    Input:  1×28×28\n"
            f"    Conv1:  32 filters (3×3) + BatchNorm + ReLU + MaxPool → 32×14×14\n"
            f"    Conv2:  64 filters (3×3) + BatchNorm + ReLU + MaxPool → 64×7×7\n"
            f"    Flatten: 3136\n"
            f"    FC1:    256 + ReLU + Dropout(0.5)\n"
            f"    FC2:    {self.num_classes} (output)\n"
            f"  Total parameters: {params:,}\n"
        )


# =============================================================================
# Module-level test
# =============================================================================
if __name__ == "__main__":
    print("Testing DigitRecognizer (CNN)...")
    
    model = DigitRecognizer()
    print(model)
    
    # Test forward pass
    batch = torch.randn(32, 1, 28, 28)
    output = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test filter extraction
    filters = model.get_conv_filters(layer=1)
    print(f"Conv1 filters shape: {filters.shape}")
    
    # Test feature maps
    single_img = torch.randn(1, 1, 28, 28)
    fm1, fm2 = model.get_feature_maps(single_img)
    print(f"Feature map 1 shape: {fm1.shape}")
    print(f"Feature map 2 shape: {fm2.shape}")
    
    print("\n✓ CNN test complete!")
