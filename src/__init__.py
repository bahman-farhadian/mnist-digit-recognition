"""
Neural Network Digit Recognition Package

Train on MNIST, Test on EMNIST (different writers)
Demonstrates cross-dataset generalization.
"""

from .model import DigitRecognizer
from .data import DataManager
from .trainer import Trainer
from .visualizer import Visualizer

__all__ = ['DigitRecognizer', 'DataManager', 'Trainer', 'Visualizer']
