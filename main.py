import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
import json

from dataset import PETDataset
from model import UNETR
from train import train_one_epoch, validate
from utils import create_log_dir, save_checkpoint, load_checkpoint

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    log_dir, timestamp = create_log_dir()
    print(f"Logs will be saved to: {log_dir}")
    
    # Save arguments
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create datasets
    train_dataset = PETDataset(
        incomplete_dir=args.train_incomplete_dir,
        complete_dir=args.train_complete_dir
    )
    
    test_dataset = PETDataset(
        incomplete_dir=args.test_incomplete_dir,
        complete_dir=args.test_complete_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Use batch size 1 for testing for easier visualization
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples, Test dataset: {len(test_dataset)} samples")
    
    # Create model
    model = UNETR(
        img_shape=(128, 128, 80),  # Match dataset dimensions
        input_dim=1,               # Single channel input
        output_dim=1,              # Single channel output
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Define loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    elif args.loss == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, patience=5)
    else:
        scheduler = None
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.resume)
            
            if scheduler is not None:
                for _ in range(start_epoch):
                    scheduler.step()
        else:
            print(f"No checkpoint found at '{args.resume}', starting from scratch")
    
    # Training loop
    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    
    total_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}\nEpoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        
        # Validate
        val_loss, val_metrics = validate(model, test_loader, criterion, device, log_dir, epoch+1)
        
        # Update learning rate if using scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch+1,
            loss=val_loss,
            path=os.path.join(log_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
        )
        
        if is_best:
            best_path = os.path.join(log_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, val_loss, best_path)
            print(f"Saved best model with validation loss: {val_loss:.6f}")
        
        # Log losses to a file
        with open(os.path.join(log_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                   f"PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}\n")
    
    # Training completed
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PET Reconstruction using UNETR')
    
    # Data paths
    parser.add_argument('--train-incomplete-dir', type=str, 
                        default='/mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/listmode_i/reconstruction_incomplete',
                        help='Directory with incomplete training PET images')
    parser.add_argument('--train-complete-dir', type=str, 
                        default='/mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000',
                        help='Directory with complete training PET images')
    parser.add_argument('--test-incomplete-dir', type=str, 
                        default='/mnt/d/fyq/sinogram/reconstruction_npy_full_test/2000000000/listmode_i/reconstruction_incomplete',
                        help='Directory with incomplete test PET images')
    parser.add_argument('--test-complete-dir', type=str, 
                        default='/mnt/d/fyq/sinogram/reconstruction_npy_full_test/2000000000',
                        help='Directory with complete test PET images')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    
    # Model parameters
    parser.add_argument('--embed-dim', type=int, default=768, help='Embedding dimension for transformer')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for transformer')
    parser.add_argument('--num-heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'l1', 'smoothl1'], help='Loss function')
    parser.add_argument('--scheduler', type=str, default='', choices=['step', 'cosine', 'plateau', ''], help='LR scheduler')
    parser.add_argument('--lr-step', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Gamma for schedulers')
    
    args = parser.parse_args()
    main(args)