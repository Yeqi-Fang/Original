import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def create_log_dir():
    """Create a timestamped log directory"""
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join('log', current_time)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, current_time

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        return model, optimizer, 0, float('inf')
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return model, optimizer, epoch, loss

def save_slices(volume, save_path, title=None):
    """
    Save X, Y, Z slices of a 3D volume as a single image
    Args:
        volume: 3D tensor or numpy array [C, X, Y, Z]
        save_path: Path to save the image
        title: Optional title for the plot
        vmin, vmax: Min/max values for visualization normalization
    """
    # Convert to numpy if tensor
    if torch.is_tensor(volume):
        volume = volume.squeeze().cpu().numpy()
    else:
        volume = volume.squeeze()
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # X-direction slice (sagittal)
    x_center = volume.shape[0] // 2
    axs[0].imshow(volume[x_center, :, :], cmap='magma', interpolation='nearest')
    axs[0].set_title(f'Sagittal Slice (X={x_center})')
    axs[0].axis('off')
    
    # Y-direction slice (coronal)
    y_center = volume.shape[1] // 2
    axs[1].imshow(volume[:, y_center, :], cmap='magma', interpolation='nearest')
    axs[1].set_title(f'Coronal Slice (Y={y_center})')
    axs[1].axis('off')
    
    # Z-direction slice (axial)
    z_center = volume.shape[2] // 2
    axs[2].imshow(volume[:, :, z_center], cmap='magma', interpolation='nearest')
    axs[2].set_title(f'Axial Slice (Z={z_center})')
    axs[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def calculate_metrics(prediction, target):
    """
    Calculate PSNR and SSIM between prediction and target
    Args:
        prediction: Predicted image
        target: Target image
    Returns:
        Dictionary of metrics
    """
    if torch.is_tensor(prediction):
        prediction = prediction.squeeze().cpu().numpy()
    if torch.is_tensor(target):
        target = target.squeeze().cpu().numpy()
    
    # Ensure data range is appropriate
    pred = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
    targ = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    # Calculate PSNR
    try:
        psnr = peak_signal_noise_ratio(targ, pred, data_range=1.0)
    except Exception:
        psnr = 0.0
    
    # Calculate SSIM
    try:
        ssim = structural_similarity(targ, pred, data_range=1.0)
    except Exception:
        ssim = 0.0
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }

def log_metrics(metrics, log_dir, epoch, split='val'):
    """Log metrics to file"""
    log_file = os.path.join(log_dir, f'{split}_metrics.csv')
    
    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch,psnr,ssim\n')
    
    # Append metrics
    with open(log_file, 'a') as f:
        f.write(f"{epoch},{metrics['psnr']:.4f},{metrics['ssim']:.4f}\n")