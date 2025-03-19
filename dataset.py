import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PETDataset(Dataset):
    """Dataset for incomplete and complete PET image pairs with preloading"""
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
        
        # Pre-load all data into memory as float16
        self.incomplete_data = {}
        self.complete_data = {}
        
        print("Preloading dataset into memory (float16)...")
        for i, file_name in enumerate(self.files):
            if i % 20 == 0:
                print(f"  Loading {i}/{len(self.files)} files...")
            
            incomplete_path = os.path.join(incomplete_dir, file_name)
            complete_path = os.path.join(complete_dir, file_name)
            
            try:
                # Load and convert to float16 to save memory
                incomplete_img = np.load(incomplete_path).astype(np.float16)
                complete_img = np.load(complete_path).astype(np.float16)
                
                # Store in dictionaries
                self.incomplete_data[file_name] = incomplete_img
                self.complete_data[file_name] = complete_img
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
                # Create a dummy sample in case of error
                dummy = np.zeros((128, 128, 80), dtype=np.float16)
                self.incomplete_data[file_name] = dummy
                self.complete_data[file_name] = dummy
        
        print("Dataset preloaded into memory")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get file name
        file_name = self.files[idx]
        
        # Get pre-loaded data and convert to float32 (from float16)
        incomplete_img = self.incomplete_data[file_name]
        complete_img = self.complete_data[file_name]
        
        # Convert numpy arrays to torch tensors (float32) and add channel dimension
        incomplete_img = torch.from_numpy(incomplete_img.astype(np.float32)).unsqueeze(0)  # [1, 128, 128, 80]
        complete_img = torch.from_numpy(complete_img.astype(np.float32)).unsqueeze(0)      # [1, 128, 128, 80]
        
        # Prepare sample
        sample = {
            'incomplete': incomplete_img, 
            'complete': complete_img, 
            'filename': file_name
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample