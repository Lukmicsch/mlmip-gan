import numpy as np

class ZDimTransform(object):
    """
    Returns the 3d scan with given z_dim.
    """
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def __call__(self, sample):
        """
        Check for false z_dim and transform images.

        :param sample: the image, mask sample
        :return: the transformed sample
        """

        img = sample['image']
        mask = sample['mask']

        z_dim_img = img.shape[-1]
        z_dim_mask = mask.shape[-1]

        if z_dim_img != self.z_dim:
            img = self.__transform_z_dim__(img, z_dim_img)

        if z_dim_mask != self.z_dim:
            mask = self.__transform_z_dim__(mask, z_dim_mask)

        sample = {'image': img, 'mask': mask}

        return sample

    def __transform_z_dim__(self, scan, z_dim):
        """
        Transforms given scan, cutting/padding from above and below.

        :param scan: the given 3d scan
        :param z_dim: where to trim the scan
        :return: the transformed scan
        """

        excess = z_dim - self.z_dim

        upper_cut = excess // 2
        lower_cut = excess - upper_cut

        if excess < 0:
            # Padding with zeros
            upper_cut = -upper_cut
            lower_cut = -lower_cut
            padded = np.zeros((scan.shape[0], scan.shape[1], self.z_dim))
            padded[:,:,upper_cut:-lower_cut] = scan
            scan = padded
        else:
            # Cutting
            scan = scan[:,:,upper_cut:-lower_cut]

        return scan

    def __repr__(self):
        return self.__class__.__name__ + '()'
