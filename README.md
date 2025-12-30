# CNN Digit Recognition

A PyTorch convolutional neural network for handwritten digit recognition with cross-dataset evaluation and stability analysis. Trains on MNIST and evaluates generalization on EMNIST (different writers).

## Highlights

- **CNN Architecture**: ~1,024,000 parameters with batch normalization
- **Cross-Dataset Evaluation**: Train on MNIST → Test on EMNIST
- **Stability Analysis**: Tests model consistency with dropout variance
- **99%+ MNIST accuracy**, **95%+ EMNIST accuracy**: Strong generalization
- **9 visualizations**: Training curves, conv filters, feature maps, stability analysis

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
Linear(320) + ReLU + Dropout(0.5)
    ↓
Linear(10) → Class scores
```

**Total parameters**: ~1,024,000

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib tqdm

# Run training (32 epochs default + stability test)
python main.py

# Quick test with fewer epochs
python main.py --epochs 10
```

## Expected Results

| Metric | Value |
|--------|-------|
| MNIST Test | 99%+ |
| EMNIST Test | 95%+ |
| Stability (std) | < 1% |

## Stability Test

After training, the model runs N test passes (N = epochs) with **dropout enabled**. This measures prediction consistency:

- **Low variance** (std < 1%): Model is stable and confident
- **High variance** (std > 2%): Model relies heavily on specific neurons

The stability test produces a line graph and box plot showing accuracy distribution.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU support)
- Linux (tested on Ubuntu)

### GPU Support

The code automatically detects CUDA GPUs. For GPU training on Linux:

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Project Structure

```
mnist-digit-recognition/
├── main.py              # Entry point
├── README.md            
├── requirements.txt     
├── src/
│   ├── __init__.py
│   ├── model.py         # CNN architecture (~1M params)
│   ├── data.py          # MNIST + EMNIST loading
│   ├── trainer.py       # Training loop + stability test
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
| `09_stability_test.png` | Stability line graph + box plot |

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --epochs INT          Training epochs (default: 32)
                        Also sets number of stability test runs
  --batch-size INT      Batch size (default: 64)
  --learning-rate FLOAT Learning rate (default: 0.001)
  --output-dir PATH     Output directory (default: outputs)
  --no-viz              Skip visualizations
  --no-save             Skip saving model
  --no-stability        Skip stability test
```

## License

MIT License
