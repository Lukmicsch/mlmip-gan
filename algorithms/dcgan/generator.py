import torch
from torch import nn

class Generator(nn.Module):
    '''
    Generator Class

    :param z_dim: the dimension of the noise vector, a scalar
    :param im_chan: the number of channels in the images
    :param hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3,
                       stride=2, first_layer=False, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.

        :param input_channels: how many channels the input feature representation has
        :param output_channels: how many channels the output feature representation should have
        :param kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        :param stride: the stride of the convolution
        :param final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        :return: a generator block
        '''
        # Build the neural block
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.

        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :return: noise tensor
        '''

        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :return network output
        '''

        x = self.unsqueeze_noise(noise)
        x = self.gen(x)

        return x
