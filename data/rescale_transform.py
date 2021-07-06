import numpy as np
from skimage.transform import resize

class RescaleTransform(object):
    """Rescale the image in a sample to a given size with shape (batch_size,
    height, width, 1).

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        z, _, h, w = image.shape

        new_h = new_w = self.output_size

        img_slices_out = np.zeros((z, new_h, new_w, 1))
        mask_slices_out = np.zeros((z, new_h, new_w, 1))

        for n,i in enumerate(image):
            img_slices_out[n,:,:,:] = resize(image[n,:,:,:], img_slices_out.shape[1:], anti_aliasing=True)
            mask_slices_out[n,:,:,:] = resize(mask[n,:,:,:], mask_slices_out.shape[1:], anti_aliasing=True)

        return {'image': img_slices_out, 'mask': mask_slices_out}
