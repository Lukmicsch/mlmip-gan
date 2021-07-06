import torch

class ReshapeTransform(object):
    """Reshape the image and mask in a sample from (width, height, z_dim) to
    shape (batch_size, width, height).
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        z_dim  = image.shape[-1]

        img_slices, indexing = self.__prepare_image_batch(image)
        mask_slices = self.__prepare_mask_batch(mask, indexing)

        return {'image': img_slices, 'mask': mask_slices}

    def __prepare_image_batch(self, batch):
        """ [height, width, z_dim] to [z_dim, height, width, 1]. """

        # Restructuring
        h, w, z = batch.shape
        batch = torch.tensor(batch)
        image_slices = batch.permute(2, 0, 1).reshape(-1, h, w).float()
        image_slices = torch.unsqueeze(image_slices, axis=1)

        # Throw out all zero padded slices
        indexing = torch.sum(image_slices, (2,3)).nonzero(as_tuple=True)
        # Relocate the color channel for resizing via skimage
        image_slices = torch.unsqueeze(image_slices[indexing], 3)

        return image_slices.numpy(), indexing

    def __prepare_mask_batch(self, batch, indexing):
        """ [height, width, z_dim] to [z_dim, height, width, 1]. """

        # Restructuring
        h, w, z = batch.shape
        batch = torch.tensor(batch)
        mask_slices = batch.permute(2, 0, 1).reshape(-1, h, w).float()
        mask_slices = torch.unsqueeze(mask_slices, axis=1)

        # Now remove padded images if necessary and add color channel for
        # resizing step via skimage
        mask_slices = torch.unsqueeze(mask_slices[indexing], 3)

        return mask_slices.numpy()
