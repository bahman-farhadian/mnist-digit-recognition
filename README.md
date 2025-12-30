# CNN Digit Recognition

A PyTorch convolutional neural network for handwritten digit recognition with cross-dataset evaluation. Trains on MNIST and evaluates generalization on EMNIST (different writers).

## Highlights

- **CNN Architecture**: 2 conv layers + 2 FC layers with batch normalization
- **Cross-Dataset Evaluation**: Train on MNIST → Test on EMNIST
- **99%+ MNIST accuracy**, **90%+ EMNIST accuracy**: Strong generalization
- **8 visualizations**: Training curves, conv filters, feature maps, confusion matrices

## Why CNN?

CNNs outperform MLPs on image tasks because they:
- Learn **local features** (edges, curves) that transfer across datasets
- Have **translation invariance** - digit position matters less
- **Share weights** spatially - fewer parameters, less overfitting

## Architecture

```
Input (1×28×28)
    ↓
Conv2d(32, 3×3) + BatchNorm + ReLU + MaxPool → 32×14×14
    ↓
Conv2d(64, 3×3) + BatchNorm + ReLU + MaxPool → 64×7×7
    ↓
Flatten → 3136
    ↓
Linear(256) + ReLU + Dropout(0.5)
    ↓
Linear(10) → Class scores
```

**Total parameters**: ~850K

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib tqdm

# Run training (10 epochs default)
python main.py

# Train longer for best results
python main.py --epochs 20
```

## Expected Results

| Dataset | Accuracy | Meaning |
|---------|----------|---------|
| MNIST Test | 99%+ | Learned from training data |
| EMNIST Test | 90-95% | Generalizes to new writers |

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU support)
- Linux (tested on Ubuntu)

### GPU Support

The code automatically detects CUDA GPUs on Linux. For GPU training:

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

## Project Structure

```
mnist-digit-recognition/
├── main.py              # Entry point
├── README.md            
├── requirements.txt     
├── src/
│   ├── __init__.py
│   ├── model.py         # CNN architecture
│   ├── data.py          # MNIST + EMNIST loading
│   ├── trainer.py       # Training loop
│   └── visualizer.py    # All visualizations
└── outputs/
    ├── model.pt         # Saved weights
    └── *.png            # Visualizations
```

## Visualizations

| File | Description |
|------|-------------|
| `01_training_history.png` | Loss and accuracy curves |
| `02_sample_data.png` | MNIST vs EMNIST samples |
| `03_predictions.png` | Predictions with confidence |
| `04_cross_dataset_evaluation.png` | MNIST vs EMNIST comparison |
| `05_conv_filters.png` | Learned 3×3 filters |
| `06_feature_maps.png` | How CNN sees digits |
| `07_confusion_matrix.png` | Error analysis |
| `08_confidence_analysis.png` | Confidence calibration |

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --epochs INT          Training epochs (default: 10)
  --batch-size INT      Batch size (default: 64)
  --learning-rate FLOAT Learning rate (default: 0.001)
  --output-dir PATH     Output directory (default: outputs)
  --no-viz              Skip visualizations
  --no-save             Skip saving model
```

## License

MIT License
