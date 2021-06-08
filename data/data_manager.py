import os
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import numpy as np
import torch

from pathlib import Path

from data.aneurysm_dataset_2d import AneurysmDataset2D
from data.z_dim_transform import ZDimTransform

class DataManager:

    def __init__(self, config):
        self.config = config
        
        self.data_path = Path(config['data_path'])
        self.img_suffix = config['img_suffix']
        self.mask_suffix = config['mask_suffix']
        self.data_format = config['data_format']
        self.compression_file_format = config['compression_file_format']
        
        self.use_transform = config['use_transform']
        self.frac_test = config['frac_test']
        self.width_and_height = config['width_and_height']
        self.z_dim = config['z_dim']
        
        self.suffix_complete = '.' + self.data_format + '.' + self.compression_file_format
        self.mask_suffix_complete = self.mask_suffix + self.suffix_complete
        self.img_suffix_complete = self.img_suffix + self.suffix_complete
        
        

    def get_full_cases(self):
        """ Returns all curated cases and permutates with seed. """
        
        all_img_suffix_complete = '*' + self.img_suffix_complete

        all_cases = list(map(lambda file: file.parts[-1].split(self.img_suffix)[0], self.data_path.glob(all_img_suffix_complete)))  # List all cases from dir
        num_cases = len(all_cases)
        
        print("%d cases detected." % num_cases)
        
        all_cases = np.random.RandomState(seed=42).permutation(all_cases)  # Permute the cases (With consistency)

        return all_cases
    
            
            
    def get_train_test_split_cases(self):
        """ Returns train and test split cases. """

        all_cases = self.get_full_cases()
        all_cases_count = len(all_cases)

        test_count = round(self.frac_test * all_cases_count)
        train_count = all_cases_count - test_count
        
        print("Split into %d train- and %d test-cases." % (train_count, test_count))

        test_cases = all_cases[:test_count]
        train_cases = all_cases[test_count:]

        return train_cases, test_cases

    
    
    def get_dataset_2d(self, cases):
        """ Returns dataset optionally with transforms. """
        
        # Init custom transforms
        z_dim_transform = ZDimTransform(self.z_dim)

        if self.use_transform:
            transform = transforms.Compose([
                z_dim_transform
            ])

            dataset = AneurysmDataset2D(cases, self.config, transform)
        else:
            dataset = AneurysmDataset2D(cases, self.config)

        return dataset
    
    
    
    def prepare_image_batch(self, batch):
        """ [batch_size, height, width, z_dim] to [batch_size * z_dim, 1, height, width]. """
        
        _, h, w, _ = batch.shape
        image_slices = batch.permute(0, 3, 1, 2).reshape(-1, h, w).float()
        image_slices = torch.unsqueeze(image_slices, axis=1)
        
        return image_slices
    
    
    
    def prepare_mask_batch(self, batch):
        """ [batch_size, height, width, z_dim] to [batch_size * z_dim, height, width]. """
        
        _, h, w, _ = batch.shape
        mask_slices = batch.permute(0, 3, 1, 2).reshape(-1, h, w).float()
        
        return mask_slices