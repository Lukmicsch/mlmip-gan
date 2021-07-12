import torch
from torch import nn
from utils.torch_utils import Conv2dSamePadding


class Generator256(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=100, im_chan=1):
        super(Generator256, self).__init__()
        self.z_dim = z_dim

        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block_1(input_size=z_dim, output_size=4096),
            self.make_gen_block_2(input_channels=4096),
            self.make_gen_block_3(input_channels=256, output_channels=256,
                                  kernel_size=4, stride=1),
            self.make_gen_block_3(input_channels=256, output_channels=128,
                                  kernel_size=4, stride=1),
            self.make_gen_block_3(input_channels=128, output_channels=64,
                                  kernel_size=3, stride=1),
            self.make_gen_block_3(input_channels=64, output_channels=32,
                                  kernel_size=3, stride=1),
            self.make_gen_block_3(input_channels=32, output_channels=16,
                                  kernel_size=3, stride=1),
            self.make_gen_block_3(input_channels=16, output_channels=8,
                                  kernel_size=3, stride=1, batch_norm=False),
            self.make_gen_block_3(input_channels=8, output_channels=1,
                                  kernel_size=3, stride=1, final_layer=True)
        )


    def make_gen_block_1(self, input_size, output_size):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Unflatten(1, (output_size, 1, 1))
        )


    def make_gen_block_2(self, input_channels, output_channels=256, kernel_size=4, stride=1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.ReLU()
        )


    def make_gen_block_3(self, input_channels, output_channels,
                         kernel_size, stride, batch_norm=True, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        if not final_layer:
            if batch_norm:
                return nn.Sequential(
                    Conv2dSamePadding(input_channels, output_channels,
                                      kernel_size, stride, 0, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
            else:
                return nn.Sequential(
                    Conv2dSamePadding(input_channels, output_channels, kernel_size, stride, 0, bias=False),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
        else: # Final Layer
            return nn.Sequential(
                Conv2dSamePadding(input_channels, output_channels, kernel_size, stride, 0, bias=False),
                nn.Sigmoid()
            )


    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        x = self.gen(x)

        return x
