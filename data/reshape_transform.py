import torch

class ReshapeTransform(object):
    """
    Reshape the image and mask in a sample from (width, height, z_dim) to
    shape (batch_size, width, height).
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Call to the transformation

        :param sample: image and mask sample
        :return: reshaped image and mask
        """
        image, mask = sample['image'], sample['mask']

        z_dim  = image.shape[-1]

        img_slices, indexing = self.__prepare_image_batch(image)
        mask_slices = self.__prepare_mask_batch(mask, indexing)

        return {'image': img_slices, 'mask': mask_slices}

    def __prepare_image_batch(self, image):
        """
        Transforms images from [height, width, z_dim] to [z_dim, height, width, 1].

        :param batch: the image batch to be transformed
        :return: the the image slices
        """
        # Restructuring
        h, w, z = image.shape
        batch = image.clone().detach()
        image_slices = batch.permute(2, 0, 1).reshape(-1, h, w).float()
        image_slices = torch.unsqueeze(image_slices, axis=1)

        # Throw out all zero padded slices
        indexing = torch.sum(image_slices, (2,3)).nonzero(as_tuple=True)
        # Relocate the color channel for resizing via skimage
        image_slices = torch.unsqueeze(image_slices[indexing], 3)

        return image_slices.numpy(), indexing

    def __prepare_mask_batch(self, mask, indexing):
        """
        Transforms masks from [height, width, z_dim] to [z_dim, height, width, 1].

        :param batch: the mask batch to be transformed
        :return: the mask slices
        """

        # Restructuring
        h, w, z = mask.shape
        batch = mask.clone().detach()
        mask_slices = batch.permute(2, 0, 1).reshape(-1, h, w).float()
        mask_slices = torch.unsqueeze(mask_slices, axis=1)

        # Now remove padded images if necessary and add color channel for
        # resizing step via skimage
        mask_slices = torch.unsqueeze(mask_slices[indexing], 3)

        return mask_slices.numpy()
