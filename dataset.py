import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PETDataset(Dataset):
    """Dataset for incomplete and complete PET image pairs"""
    def __init__(self, incomplete_dir, complete_dir, transform=None):
        """
        Args:
            incomplete_dir: Directory containing incomplete PET reconstructions
            complete_dir: Directory containing complete PET reconstructions
            transform: Optional transforms to be applied
        """
        self.incomplete_dir = incomplete_dir
        self.complete_dir = complete_dir
        self.transform = transform
        
        # Get all file names in the incomplete directory
        self.files = [f for f in os.listdir(incomplete_dir) if f.endswith('.npy')]
        print(f"Found {len(self.files)} files in dataset")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get file name
        file_name = self.files[idx]
        
        # Load incomplete and complete PET images
        incomplete_path = os.path.join(self.incomplete_dir, file_name)
        complete_path = os.path.join(self.complete_dir, file_name)
        
        # Load numpy arrays
        try:
            incomplete_img = np.load(incomplete_path)
            complete_img = np.load(complete_path)
            
            # Convert to tensors and add channel dimension
            incomplete_img = torch.from_numpy(incomplete_img).float().unsqueeze(0)  # [1, 128, 128, 80]
            complete_img = torch.from_numpy(complete_img).float().unsqueeze(0)      # [1, 128, 128, 80]
            
            # Normalize if needed (optional - comment out if data is already normalized)
            # incomplete_img = (incomplete_img - incomplete_img.min()) / (incomplete_img.max() - incomplete_img.min() + 1e-8)
            # complete_img = (complete_img - complete_img.min()) / (complete_img.max() - complete_img.min() + 1e-8)
            
            sample = {
                'incomplete': incomplete_img, 
                'complete': complete_img, 
                'filename': file_name
            }
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
            
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            # Return a dummy sample in case of error
            dummy = torch.zeros((1, 128, 128, 80))
            return {'incomplete': dummy, 'complete': dummy, 'filename': file_name}