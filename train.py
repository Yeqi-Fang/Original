import os
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import save_slices, calculate_metrics, log_metrics

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        # Get data
        incomplete = batch['incomplete'].to(device)
        complete = batch['complete'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(incomplete)
        loss = criterion(outputs, complete)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        batch_loss = loss.item()
        epoch_loss += batch_loss
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{batch_loss:.6f}")
    
    # Calculate average loss
    epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f}s. Avg loss: {epoch_loss:.6f}")
    return epoch_loss

def validate(model, val_loader, criterion, device, log_dir, epoch):
    """Validate model and save visualizations"""
    model.eval()
    val_loss = 0.0
    batch_count = 0
    all_metrics = {'psnr': 0.0, 'ssim': 0.0}
    metrics_count = 0
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Get data
            incomplete = batch['incomplete'].to(device)
            complete = batch['complete'].to(device)
            filename = batch['filename'][0]  # Batch size 1 for test
            
            # Forward pass
            outputs = model(incomplete)
            loss = criterion(outputs, complete)
            
            # Update statistics
            batch_loss = loss.item()
            val_loss += batch_loss
            batch_count += 1
            
            # Calculate metrics
            metrics = calculate_metrics(outputs.cpu(), complete.cpu())
            for key in all_metrics:
                all_metrics[key] += metrics[key]
            metrics_count += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{batch_loss:.6f}", psnr=f"{metrics['psnr']:.2f}")
            
            # Save visualizations for the first few samples
            if i < 3:  # Save first 3 test samples
                # Determine min/max for consistent visualization
                vmin = min(incomplete.min().item(), complete.min().item(), outputs.min().item())
                vmax = max(incomplete.max().item(), complete.max().item(), outputs.max().item())
                
                # Save input image
                save_slices(
                    incomplete,
                    os.path.join(log_dir, f'epoch_{epoch:03d}_sample_{i}_input.png'),
                    title=f'Input (Incomplete) - {filename}',
                    vmin=vmin, vmax=vmax
                )
                
                # Save output (prediction) image
                save_slices(
                    outputs,
                    os.path.join(log_dir, f'epoch_{epoch:03d}_sample_{i}_output.png'),
                    title=f'Output (Predicted) - PSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.4f}',
                    vmin=vmin, vmax=vmax
                )
                
                # Save target (ground truth) image
                save_slices(
                    complete,
                    os.path.join(log_dir, f'epoch_{epoch:03d}_sample_{i}_target.png'),
                    title=f'Target (Complete) - {filename}',
                    vmin=vmin, vmax=vmax
                )
    
    # Calculate average metrics
    val_loss = val_loss / batch_count if batch_count > 0 else float('inf')
    for key in all_metrics:
        all_metrics[key] = all_metrics[key] / metrics_count if metrics_count > 0 else 0.0
    
    # Print results
    print(f"Validation - Loss: {val_loss:.6f}, PSNR: {all_metrics['psnr']:.2f}, SSIM: {all_metrics['ssim']:.4f}")
    
    # Log metrics
    log_metrics(all_metrics, log_dir, epoch)
    
    return val_loss, all_metrics