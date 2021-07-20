import os
import torchvision.transforms as transforms
import numpy as np
import nibabel as nib
import numpy as np
import torch

from pathlib import Path

from data.aneurysm_dataset_2d import AneurysmDataset2D
from data.z_dim_transform import ZDimTransform
from data.rescale_transform import RescaleTransform
from data.reshape_transform import ReshapeTransform
from data.clip_and_normalize_transform import ClipValuesAndNormalize

class DataManager:
    """
    Data manager class
    """
    def __init__(self, config):
        """
        Init class and construct path to image and mask.

        :param config: configfile in configs/ folder
        """
        self.config = config

        self.data_path = Path(config['data_path'])
        self.img_suffix = config['img_suffix']
        self.mask_suffix = config['mask_suffix']
        self.data_format = config['data_format']
        self.compression_file_format = config['compression_file_format']

        self.width_and_height = config['width_and_height']
        self.width_and_height_to_model = config['width_and_height_to_model']
        self.z_dim = config['z_dim']

        self.activation = config['activation']

        self.suffix_complete = '.' + self.data_format + '.' + self.compression_file_format
        self.mask_suffix_complete = self.mask_suffix + self.suffix_complete
        self.img_suffix_complete = self.img_suffix + self.suffix_complete


    def get_full_cases(self):
        """
        Returns all curated cases and permutates with seed.
        :return: all cases from data folder
        """

        all_img_suffix_complete = '*' + self.img_suffix_complete

        all_cases = list(map(lambda file: file.parts[-1].split(self.img_suffix)[0], self.data_path.glob(all_img_suffix_complete)))  # List all cases from dir
        num_cases = len(all_cases)

        print("%d cases detected." % num_cases)

        all_cases = np.random.RandomState(seed=42).permutation(all_cases)  # Permute the cases (With consistency)

        return all_cases


    def get_dataset_2d(self, cases):
        """
        Returns dataset optionally with transforms.

        :param cases: all the cases which are supposed to be in the dataset
        :return: the dataset
        """

        # Init custom transforms
        z_dim_transform = ZDimTransform(self.z_dim)
        reshape_transform = ReshapeTransform()
        rescale_transform = RescaleTransform(self.width_and_height_to_model)
        clip_values_and_normalize_transform = ClipValuesAndNormalize(min_percentile=680.,max_percentile=6400.,mean=2819.1563145249925, std=1745.2284865700638, activation=self.activation)

        transform = transforms.Compose([
            z_dim_transform,
            clip_values_and_normalize_transform,
            reshape_transform,
            rescale_transform
        ])

        dataset = AneurysmDataset2D(cases, self.config, transform)

        return dataset
