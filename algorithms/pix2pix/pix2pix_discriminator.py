import torch
from torch import nn

from algorithms.pix2pix.pix2pix_blocks import FeatureMapBlock, ContractingBlock



class Discriminator(nn.Module):
    """"
    PatchGAN discriminator, structured like contracting part of UNet. Output: matrix of real-fake values.  
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    """

    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)

        self.final = nn.Conv2d(hidden_channels * 16, hidden_channels, kernel_size=1)



    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn