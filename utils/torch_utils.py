import numpy as np
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_loss_fn(loss_fn):
    """ Return loss function specified in config. """

    criterion = None

    if loss_fn == "L1":
        criterion = nn.L1Loss
    elif loss_fn == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def plot_tensor_images(images, num_images, size):
    indicies = range(images.shape[0])
    image_choices = np.random.choice(indicies, num_images)

    figure, ax = plt.subplots(1, 1)

    # convert to numpy array
    image_np = self.image_nib.get_fdata()
    mask_np = self.mask_nib.get_fdata()

    # get number of slices
    _, _, slices = image_np.shape
    ind = slices // 2

    # plot image with mask overlay
    image_plt = ax.imshow(self.overlay)


def show_tensor_images_dcgan(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



def show_tensor_images_pix2pix(image_tensor, num_images=25, size=(1, 28, 28)):
    """ Visualize images from tensor. """

    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
