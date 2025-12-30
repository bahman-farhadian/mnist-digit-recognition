# Neural Network Digit Recognition

A portfolio project demonstrating deep learning fundamentals with PyTorch: training a neural network to recognize handwritten digits and evaluating cross-dataset generalization.

## Highlights

- **Train on MNIST** → **Test on EMNIST**: Cross-dataset evaluation proves generalization
- **1024 hidden neurons**: Single hidden layer with 814,090 learnable parameters
- **~98% MNIST accuracy**, **~85-95% EMNIST accuracy**: Strong cross-dataset performance
- **Every line commented**: Understand exactly what each piece of code does
- **8 visualizations**: See what the network actually learns

## Cross-Dataset Evaluation

This project trains on **MNIST** (60,000 images) and tests on both:
- **MNIST Test** (10,000 images): Same distribution as training
- **EMNIST-Digits** (40,000 images): Different writers from the same NIST database

EMNIST uses the same visual format as MNIST (28×28, centered, grayscale) but comes from different writers. High EMNIST accuracy without ever training on it proves the model learned **general digit features**, not just MNIST-specific patterns.

## Quick Start

```bash
# Clone and enter directory
cd mnist-digit-recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run training with default settings (5 epochs, 1024 neurons)
python main.py

# Or customize
python main.py --epochs 10 --hidden-size 2048
```

## Expected Results

After 5 epochs with 1024 hidden neurons:

| Dataset | Accuracy | Meaning |
|---------|----------|---------|
| MNIST Test | 97-98% | Model learned from training data |
| EMNIST Test | 85-95% | Model generalizes to new writers |

The gap between MNIST and EMNIST is expected and indicates the model isn't overfitting to MNIST-specific patterns.

## Project Structure

```
mnist-digit-recognition/
├── main.py              # Single entry point - run this!
├── README.md            # This file
├── requirements.txt     # Dependencies
├── src/
│   ├── __init__.py      # Package init
│   ├── model.py         # Neural network architecture
│   ├── data.py          # Dataset loading (MNIST + EMNIST)
│   ├── trainer.py       # Training loop
│   └── visualizer.py    # All visualizations
└── outputs/             # Generated files go here
    ├── model.pt         # Saved model weights
    └── *.png            # Visualization images
```

## Network Architecture

```
Input Layer:    784 neurons (28×28 flattened image)
                    ↓
Hidden Layer:   1024 neurons + ReLU activation
                    ↓
Output Layer:   10 neurons (one per digit 0-9)
                    ↓
                Softmax → Probabilities
```

**Total parameters**: 814,090 (784×1024 + 1024 + 1024×10 + 10)

## Generated Visualizations

| File | Description |
|------|-------------|
| `01_training_history.png` | Loss and accuracy curves over epochs |
| `02_sample_data.png` | Side-by-side MNIST vs EMNIST samples |
| `03_predictions.png` | Model predictions with confidence scores |
| `04_cross_dataset_eval.png` | MNIST vs EMNIST accuracy comparison |
| `05_neuron_weights.png` | What patterns each neuron detects |
| `06_activations_by_digit.png` | How different digits activate neurons |
| `07_confusion_matrix.png` | Which digits get confused |
| `08_confidence_analysis.png` | Confidence calibration |

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --epochs INT        Training epochs (default: 5)
  --batch-size INT    Batch size (default: 64)
  --learning-rate FLOAT  Adam learning rate (default: 0.001)
  --hidden-size INT   Hidden layer neurons (default: 1024)
  --output-dir PATH   Output directory (default: outputs)
  --no-viz           Skip visualization generation
  --no-save          Skip saving model
```

## Understanding the Code

Each file is heavily commented to explain:

- **model.py**: Neural network architecture, weight initialization, forward pass
- **data.py**: How datasets are loaded and preprocessed
- **trainer.py**: The training loop, backpropagation, evaluation
- **visualizer.py**: How each visualization is created

## Key Concepts Demonstrated

1. **Forward Propagation**: Input → Hidden → Output with ReLU activation
2. **Backpropagation**: Computing gradients to update weights
3. **Cross-Entropy Loss**: Standard loss for multi-class classification
4. **Adam Optimizer**: Adaptive learning rate optimization
5. **Mini-batch Training**: Processing 64 images at a time for efficiency
6. **Cross-Dataset Generalization**: Testing on unseen data distribution

## Dependencies

- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Dataset utilities
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `tqdm>=4.65.0` - Progress bars

## License

MIT License - Feel free to use this code for learning and portfolio projects.

## Acknowledgments

- MNIST dataset: Yann LeCun et al.
- EMNIST dataset: Cohen et al. (2017)
- PyTorch: Meta AI Research
