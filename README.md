# PET Image Reconstruction with UNETR

This repository contains code for incomplete-ring PET (Positron Emission Tomography) image reconstruction using a 3D UNETR (UNet Transformer) model.

## Project Structure

- `main.py`: Main script for training and evaluation
- `model.py`: UNETR model implementation
- `dataset.py`: Data loading utilities
- `train.py`: Training and validation functions
- `utils.py`: Helper functions for visualization and metrics calculation

## Setup

### Requirements

```
torch>=1.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-image>=0.18.0
tqdm>=4.50.0
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Data Preparation

The code expects 3D PET images in NumPy (.npy) format with shape (128, 128, 80).

Data should be organized in the following directory structure:
```
/path/to/data/
├── reconstruction_npy_full_train/
│   ├── 2000000000/                      # Complete PET images 
│   └── 2000000000/listmode_i/reconstruction_incomplete/  # Incomplete PET images
└── reconstruction_npy_full_test/
    ├── 2000000000/                      # Complete PET images
    └── 2000000000/listmode_i/reconstruction_incomplete/  # Incomplete PET images
```

## Usage

### Training

```bash
python main.py \
  --train-incomplete-dir /path/to/train/incomplete \
  --train-complete-dir /path/to/train/complete \
  --test-incomplete-dir /path/to/test/incomplete \
  --test-complete-dir /path/to/test/complete \
  --batch-size 2 \
  --epochs 100 \
  --lr 1e-4
```

### Resuming Training

To resume from a checkpoint:

```bash
python main.py \
  --resume /path/to/checkpoint.pth \
  --epochs 150  # Total number of epochs to train
```

### Key Arguments

- `--batch-size`: Batch size for training (default: 2)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--embed-dim`: Embedding dimension for transformer (default: 768)
- `--patch-size`: Patch size for transformer (default: 16)
- `--num-heads`: Number of attention heads (default: 12)
- `--loss`: Loss function: 'mse', 'l1', or 'smoothl1' (default: 'mse')
- `--optimizer`: 'adam' or 'sgd' (default: 'adam')
- `--scheduler`: Learning rate scheduler: 'step', 'cosine', 'plateau', or '' (default: '')

Run `python main.py --help` for a complete list of options.

## Results

During training, the following are saved in a timestamped log directory:

- Checkpoint after each epoch
- Best model based on validation loss
- Visualization of input, output, and target slices along X, Y, Z directions
- Training and validation metrics (loss, PSNR, SSIM)

### Log Structure

```
log/
└── 20251114321/  # Timestamp
    ├── args.json
    ├── best_model.pth
    ├── checkpoint_epoch_001.pth
    ├── epoch_001_sample_0_input.png
    ├── epoch_001_sample_0_output.png
    ├── epoch_001_sample_0_target.png
    ├── training_log.txt
    └── val_metrics.csv
```

## Model Architecture

The model uses a UNETR architecture, which combines a ViT (Vision Transformer) encoder with a 3D UNet-like decoder. The transformer processes 3D patches of the input volume and extracts features at multiple depths that are used by the decoder for spatial reconstruction.

## Citation

If you use this code, please cite the relevant papers:
- UNETR: Transformers for 3D Medical Image Segmentation
- [Your research paper]