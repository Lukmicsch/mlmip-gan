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
from data.color_channel_transform import ColorChannelTransform

class DataManager:

    def __init__(self, config):
        self.config = config

        self.data_path = Path(config['data_path'])
        self.img_suffix = config['img_suffix']
        self.mask_suffix = config['mask_suffix']
        self.data_format = config['data_format']
        self.compression_file_format = config['compression_file_format']

        self.width_and_height = config['width_and_height']
        self.width_and_height_to_model = config['width_and_height_to_model']
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


    def get_dataset_2d(self, cases):
        """ Returns dataset optionally with transforms. """

        # Init custom transforms
        z_dim_transform = ZDimTransform(self.z_dim)
        reshape_transform = ReshapeTransform()
        rescale_transform = RescaleTransform(self.width_and_height_to_model)

        transform = transforms.Compose([
            z_dim_transform,
            reshape_transform,
            rescale_transform,
        ])

        dataset = AneurysmDataset2D(cases, self.config, transform)

        return dataset


    def prepare_image_batch(self, batch):
        """ [batch_size, height, width, z_dim] to [batch_size * z_dim, 1, height, width]. """

        # Restructuring
        _, h, w, _ = batch.shape
        image_slices = batch.permute(0, 3, 1, 2).reshape(-1, h, w).float()
        image_slices = torch.unsqueeze(image_slices, axis=1)

        batch_size_slices = image_slices.shape[0]

        # Throw out all zero padded slices
        indexing = torch.sum(image_slices, (2,3)).nonzero(as_tuple=True)
        image_slices = torch.unsqueeze(image_slices[indexing], 1)
        if batch_size_slices != image_slices.shape[0]:
            print("Padding was removed.")

        return image_slices, indexing


    def prepare_mask_batch(self, batch, indexing):
        """ [batch_size, height, width, z_dim] to [batch_size * z_dim, height, width]. """

        # Restructuring
        _, h, w, _ = batch.shape
        mask_slices = batch.permute(0, 3, 1, 2).reshape(-1, h, w).float()
        mask_slices = torch.unsqueeze(mask_slices, axis=1)
        batch_size_slices = mask_slices.shape[0]

        # Now remove padded images if necessary
        mask_slices = mask_slices[indexing]

        if batch_size_slices != mask_slices.shape[0]:
            print("Padding was removed.")

        return mask_slices
