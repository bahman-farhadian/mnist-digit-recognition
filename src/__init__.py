"""
CNN Digit Recognition Package

Train CNN on MNIST, test on EMNIST (different writers)
Demonstrates cross-dataset generalization with convolutional networks.
"""

from .model import DigitRecognizer
from .data import DataManager
from .trainer import Trainer
from .visualizer import Visualizer

__all__ = ['DigitRecognizer', 'DataManager', 'Trainer', 'Visualizer']
