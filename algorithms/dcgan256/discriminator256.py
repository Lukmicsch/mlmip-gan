import torch
from torch import nn
from utils.torch_utils import Conv2dSamePadding, GaussianNoise

class Discriminator256(nn.Module):
    '''
    Discriminator Class

    :param im_chan: the number of channels in the images
    :param hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator256, self).__init__()
        self.disc = nn.Sequential(
            GaussianNoise(sigma=0.2),
            self.make_disc_block(input_channels=1, output_channels=8,
                                 kernel_size=3,
                            stride=1, batch_norm=False),
            self.make_disc_block(input_channels=8, output_channels=16,
                                 kernel_size=3, stride=1, batch_norm=True),
            self.make_disc_block(input_channels=16, output_channels=32,
                                 kernel_size=3, stride=1, batch_norm=True),
            self.make_disc_block(input_channels=32, output_channels=64,
                                 kernel_size=3, stride=1, batch_norm=True),
            self.make_disc_block(input_channels=64, output_channels=128,
                                 kernel_size=3, stride=1, batch_norm=True),
            self.make_disc_block(input_channels=128, output_channels=256,
                                 kernel_size=3, stride=1, batch_norm=True),
            self.make_disc_block(final_layer=True)
        )


    def make_disc_block(self, input_channels=1, output_channels=8,
                        kernel_size=3,
                        stride=1, batch_norm=True, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN.

        :param input_channels: how many channels the input feature representation has
        :param output_channels: how many channels the output feature representation should have
        :param kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        :param stride: the stride of the convolution
        :param final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        :return: discriminator block
        '''
        # Build the neural block
        if not final_layer:
            if batch_norm:
                return nn.Sequential(
                    Conv2dSamePadding(input_channels, output_channels,
                                         kernel_size, stride, 0, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.25),
                    nn.AvgPool2d(2)
                )
            else:
                return nn.Sequential(
                    Conv2dSamePadding(input_channels, output_channels,
                                         kernel_size, stride, 0, bias=False),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.25),
                    nn.AvgPool2d(2)
                )

        else: # Final Layer
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(4096, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.

        :param image: a flattened image tensor with dimension (im_dim)
        :return: discriminator output
        '''
        disc_pred = self.disc(image)

        return disc_pred
