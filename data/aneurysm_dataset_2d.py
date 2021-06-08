import os
from pathlib import Path

import nibabel as nib
import numpy as np

from torch.utils.data import Dataset

class AneurysmDataset2D(Dataset):
    """ Aneurysm dataset. """

    def __init__(self, cases, config, transform=None):
        self.cases = cases
        self.transform = transform

        self.data_path = config['data_path']
        self.data_format = config['data_format']
        self.compression_file_format = config['compression_file_format']
        self.img_suffix = config['img_suffix']
        self.mask_suffix = config['mask_suffix']

        self.img_suffix = self.img_suffix + '.' + self.data_format + '.' + self.compression_file_format
        self.mask_suffix = self.mask_suffix + '.' + self.data_format + '.' + self.compression_file_format

    def __len__(self):
        return len(self.cases)

    def __load_image(self, case):
        # load NIFTI image
        img_name = case + self.img_suffix
        img_path = os.path.join(self.data_path, img_name)
        image_nib = nib.load(img_path)

        return image_nib.get_fdata() # To numpy

    def __load_mask(self, case):
        # load NIFTI mask
        mask_name = case + self.mask_suffix
        mask_path = os.path.join(self.data_path, mask_name)
        mask_nib = nib.load(mask_path)

        return mask_nib.get_fdata() # To numpy

    def __getitem__(self, idx):
        # load image and ground-truth
        case = self.cases[idx]
        
        image = self.__load_image(case)
        mask = self.__load_mask(case)

        # data augmentation
        sample = {'image': image, 'mask': mask}
        # Transformations can be integrated here
        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['mask']