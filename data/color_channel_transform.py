import torch

class ColorChannelTransform(object):
    """Reshape the image and mask in a sample from (batch_size, height, width,
    1) to
    shape (batch_size, 1, height, width) and (batch_size, height, width).
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_slices = torch.tensor(image).permute(0, 3, 1, 2)
        mask_slices = torch.squeeze(torch.tensor(mask))

        return {'image': image_slices, 'mask': mask_slices}
