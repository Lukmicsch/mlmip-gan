import torch
from torch import nn


def crop(image, new_shape):
    """
    Center-crop the image

    :param image: image tensor of shape (batch size, channels, height, width)
    :param new_shape: a torch.Size object with the shape you want x to have
    :return: cropped image
    """
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - round(new_shape[2] / 2)
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - round(new_shape[3] / 2)
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]

    return cropped_image



def weights_init(m):
    """
    Init weights used by discriminator and generator.

    :param m: given nn.module
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.

    :param n_samples: the number of samples to generate, a scalar
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    :return: noise tensor
    '''
    return torch.randn(n_samples, z_dim, device=device)
