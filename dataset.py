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
        self.incomplete_files = [f for f in os.listdir(incomplete_dir) if f.endswith('.npy')]
        
        # Create mapping between incomplete and complete filenames
        self.incomplete_to_complete = {}
        for inc_file in self.incomplete_files:
            # Assuming format: "reconstructed_incomplete_index{N}_num{X}.npy" -> "reconstructed_index{N}_num{X}.npy"
            comp_file = inc_file.replace("_incomplete", "")
            
            # Verify that the complete file exists
            if os.path.exists(os.path.join(complete_dir, comp_file)):
                self.incomplete_to_complete[inc_file] = comp_file
            else:
                print(f"Warning: No matching complete file found for {inc_file}")
        
        # Update file list to only include files with valid mappings
        self.files = list(self.incomplete_to_complete.keys())
        print(f"Found {len(self.files)} valid file pairs in dataset")
        
        # Pre-load all data into memory as float32
        self.incomplete_data = {}
        self.complete_data = {}
        
        print("Preloading dataset into memory (float32)...")
        for i, inc_file in enumerate(self.files):
            if i % 20 == 0:
                print(f"  Loading {i}/{len(self.files)} file pairs...")
            
            comp_file = self.incomplete_to_complete[inc_file]
            incomplete_path = os.path.join(incomplete_dir, inc_file)
            complete_path = os.path.join(complete_dir, comp_file)
            
            try:
                # Load and convert to float32 to save memory
                incomplete_img = np.load(incomplete_path).astype(np.float32)
                complete_img = np.load(complete_path).astype(np.float32)
                
                # Store in dictionaries
                self.incomplete_data[inc_file] = incomplete_img
                self.complete_data[inc_file] = complete_img
            except Exception as e:
                print(f"Error loading file pair {inc_file}/{comp_file}: {e}")
                # Remove this file from the list
                self.files.remove(inc_file)
        
        print(f"Dataset preloaded into memory with {len(self.files)} valid samples")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get file name (incomplete version)
        inc_file = self.files[idx]
        
        # Get pre-loaded data and convert to float32 
        incomplete_img = self.incomplete_data[inc_file]
        complete_img = self.complete_data[inc_file]
        
        # Convert numpy arrays to torch tensors (float32) and add channel dimension
        incomplete_img = torch.from_numpy(incomplete_img.astype(np.float32)).unsqueeze(0)  # [1, 128, 128, 80]
        complete_img = torch.from_numpy(complete_img.astype(np.float32)).unsqueeze(0)      # [1, 128, 128, 80]
        
        # Prepare sample
        sample = {
            'incomplete': incomplete_img, 
            'complete': complete_img, 
            'filename': inc_file  # We use the incomplete filename as reference
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample