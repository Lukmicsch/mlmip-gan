import numpy as np

class ZDimTransform(object):
    """ Returns image with given z_dim. """
    
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def __call__(self, sample):
        """ Check for false z_dim and transform images. """
        
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
    
    def __transform_z_dim__(self, pic, z_dim):
        """ Transforms given image. """
        
        excess = z_dim - self.z_dim
            
        upper_cut = excess // 2
        lower_cut = excess - upper_cut

        if excess < 0:
            # Padding with zeros
            upper_cut = -upper_cut
            lower_cut = -lower_cut
            padded = np.zeros((pic.shape[0], pic.shape[1], self.z_dim))
            padded[:,:,upper_cut:-lower_cut] = pic
            pic = padded
        else:
            # Cutting
            pic = pic[:,:,upper_cut:-lower_cut]
            
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'