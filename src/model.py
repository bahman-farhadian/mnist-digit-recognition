"""
Neural Network Model for Digit Recognition

This module defines a simple Multi-Layer Perceptron (MLP):
    Input (784) → Hidden (1024) → Output (10)

Architecture explanation:
- Input: 28×28 image flattened to 784 values
- Hidden: 1024 neurons with ReLU activation
- Output: 10 neurons (one per digit class 0-9)

Why this architecture?
- Simple enough to understand completely
- Large enough (1024 neurons) to learn complex patterns
- Single hidden layer demonstrates core neural network concepts
- Every weight and activation can be visualized
"""

import torch
import torch.nn as nn


class DigitRecognizer(nn.Module):
    """
    Simple feedforward neural network for digit classification.
    
    Architecture: 784 → hidden_size → 10
    
    Attributes:
        input_size: 784 (28×28 flattened image)
        hidden_size: Number of hidden neurons (default 1024)
        output_size: 10 (digits 0-9)
    
    Example:
        >>> model = DigitRecognizer(hidden_size=1024)
        >>> x = torch.randn(32, 1, 28, 28)  # batch of 32 images
        >>> output = model(x)  # shape: [32, 10]
    """
    
    def __init__(self, hidden_size: int = 1024):
        """
        Initialize the network.
        
        Args:
            hidden_size: Number of neurons in hidden layer
        """
        super().__init__()
        
        self.input_size = 784    # 28 × 28 = 784 pixels
        self.hidden_size = hidden_size
        self.output_size = 10    # 10 digit classes
        
        # Hidden layer: transforms 784 inputs → hidden_size features
        # Each neuron learns to detect a specific pattern in the input
        self.hidden = nn.Linear(self.input_size, self.hidden_size)
        
        # Output layer: transforms hidden_size features → 10 class scores
        # Each output neuron corresponds to one digit (0-9)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        
        # ReLU activation: introduces non-linearity
        # f(x) = max(0, x) - passes positive values, blocks negative
        self.relu = nn.ReLU()
        
        # Initialize weights using Kaiming/He initialization
        # Better than random for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization.
        
        For ReLU networks, Kaiming initialization helps maintain
        appropriate variance of activations across layers.
        """
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        
        nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → class scores.
        
        Args:
            x: Input tensor of shape [batch, 1, 28, 28] or [batch, 784]
            
        Returns:
            Class scores of shape [batch, 10]
        """
        # Flatten image to vector: [batch, 1, 28, 28] → [batch, 784]
        x = x.view(x.size(0), -1)
        
        # Hidden layer with ReLU activation
        x = self.hidden(x)
        x = self.relu(x)
        
        # Output layer (no activation - CrossEntropyLoss applies softmax)
        x = self.output(x)
        
        return x
    
    def get_hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hidden layer activations (after ReLU).
        
        Useful for visualization and understanding what each neuron detects.
        
        Args:
            x: Input tensor
            
        Returns:
            Hidden activations of shape [batch, hidden_size]
        """
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.relu(x)
        return x
    
    def get_hidden_weights(self) -> torch.Tensor:
        """
        Get hidden layer weights.
        
        Each row is one neuron's weights (784 values).
        Can be reshaped to 28×28 to visualize what pattern
        that neuron responds to.
        
        Returns:
            Weight matrix of shape [hidden_size, 784]
        """
        return self.hidden.weight.detach().cpu()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation showing architecture."""
        params = self.count_parameters()
        return (
            f"DigitRecognizer(\n"
            f"  Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}\n"
            f"  Parameters: {params:,}\n"
            f"    Hidden layer: {self.input_size} × {self.hidden_size} + {self.hidden_size} = "
            f"{self.input_size * self.hidden_size + self.hidden_size:,}\n"
            f"    Output layer: {self.hidden_size} × {self.output_size} + {self.output_size} = "
            f"{self.hidden_size * self.output_size + self.output_size:,}\n"
            f")"
        )


# =============================================================================
# Module-level test
# =============================================================================
if __name__ == "__main__":
    print("Testing DigitRecognizer...")
    
    # Create model
    model = DigitRecognizer(hidden_size=1024)
    print(model)
    
    # Test forward pass
    batch = torch.randn(32, 1, 28, 28)
    output = model(batch)
    print(f"\nInput shape:  {batch.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test helper methods
    activations = model.get_hidden_activations(batch)
    print(f"Hidden activations shape: {activations.shape}")
    
    weights = model.get_hidden_weights()
    print(f"Hidden weights shape: {weights.shape}")
    
    print("\n✓ Model test complete!")
