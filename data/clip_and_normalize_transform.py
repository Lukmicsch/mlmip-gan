import torch
from typing import Dict
from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    """
    Abstraction class for transform.
    """
    @abstractmethod
    def transform_impl(self, image):
        pass

    def __call__(self, sample):
        if isinstance(sample, Dict):
            image, mask = sample["image"], sample["mask"]
            return {
                "image": self.transform_impl(image),
                "mask": self.transform_impl(mask),
            }
        else:
            return self.transform_impl(sample)


class ClipValuesAndNormalize(AbstractTransform):
    """
    Rescale the image in a sample to a given size.

    :param output_size (tuple or int): Desired output size. If tuple, output is
    :param matched to output_size. If int, smaller of image edges is matched
    :param to output_size keeping aspect ratio the same.
    """
    def __init__(self, min_percentile, max_percentile, mean, std, activation):
        assert isinstance(min_percentile, float)
        assert isinstance(max_percentile, float)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        self.mean = mean
        self.std = std
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.activation = activation

    def __call__(self, sample):
        """
        Call to the transformation

        :param sample: the image, mask sample
        :return: the transformed sample
        """
        if isinstance(sample, Dict):
            if "inter_mask" in sample:
                image, mask, inter_mask = (
                    sample["image"],
                    sample["mask"],
                    sample["inter_mask"],
                )
                return {
                    "image": self.transform_impl(torch.from_numpy(image)),
                    "mask": torch.squeeze(torch.from_numpy(mask)),
                    "inter_mask": torch.squeeze(torch.from_numpy(inter_mask)),
                }
            else:
                image, mask = sample["image"], sample["mask"]
                return {
                    "image": self.transform_impl(torch.from_numpy(image)),
                    "mask": torch.squeeze(torch.from_numpy(mask)),
                }
        else:
            return self.transform_impl(sample)


    def transform_impl(self, image):
        """
        The actual transformation implementation.

        :param image: the image to normalize and clip
        :return: the normalized and clipped image
        """
        image = torch.clamp(image, min=self.min_percentile, max=self.max_percentile)

        if self.activation == 'tanh':
            i_range = self.max_percentile - self.min_percentile
            image -= self.min_percentile
            image /= i_range
            image = image * 2 - 1
        elif self.activation == 'sigmoid':
            image -= self.min_percentile
            image /= self.max_percentile

        return torch.squeeze(image)
