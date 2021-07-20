import os
from pathlib import Path

import nibabel as nib
import numpy as np

from torch.utils.data import Dataset

class AneurysmDataset2D(Dataset):
    """
    Aneurysm dataset class

    :param cases: which cases should be used to create the dataset
    :param config: the config dict
    :param transform: the transformations for the images
    """
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
        """
        Function to load images from cases in numpy array.

        :param cases: which cases should be used for the dataset
        :return: image in numpy array
        """
        # load NIFTI image
        img_name = case + self.img_suffix
        img_path = os.path.join(self.data_path, img_name)
        image_nib = nib.load(img_path)

        return image_nib.get_fdata() # To numpy

    def __load_mask(self, case):
        """
        Function to load masks from cases in numpy array.

        :param cases: which cases should be used for the dataset
        :return: mask in numpy array
        """
        # load NIFTI mask
        mask_name = case + self.mask_suffix
        mask_path = os.path.join(self.data_path, mask_name)
        mask_nib = nib.load(mask_path)

        return mask_nib.get_fdata() # To numpy

    def __getitem__(self, idx):
        """
        Function that returns an image and the corresponding mask in a numpy array.

        :param idx: which cases should be loaded
        :return: image and mask as numpy array
        """
        # load image and ground-truth
        case = self.cases[idx]

        image = self.__load_image(case)
        mask = self.__load_mask(case)

        # data augmentation
        sample = {'image': image, 'mask': mask}
        # Transformations can be integrated here
        sample = self.transform(sample)

        return sample['image'], sample['mask']
