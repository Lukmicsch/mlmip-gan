import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    """
    Convolutional layer with same padding. Equivalent to keras padding=same.
    https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    """
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer for discriminator.
    https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/3
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


def get_loss_fn(loss_fn):
    """
    Return loss function specified in config.

    :param loss_fn: specifies the loss function
    :return: the loss function
    """

    criterion = None

    if loss_fn == "L1":
        criterion = nn.L1Loss
    elif loss_fn == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def plot_tensor_images(images, num_images=9):
    """
    Plots images in gray colormap, by default 9 with 3 per row and column.

    :param images: the images to be plotted
    :param num_images: how many images in total
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = range(images.shape[0])
    indices = torch.tensor(np.random.choice(batch_size,
                                                 num_images)).to(device)
    images_selected = torch.index_select(images, 0,
                                         indices).to('cpu').detach().numpy()

    plot_size = 3

    w = images.shape[2]
    h = images.shape[3]

    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns*rows):
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        im = images_selected[i,0,:,:]#.astype(np.uint8)
        plt.imshow(im, cmap="gray")

    plt.show()  # finally, render the plot

