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



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """ Visualize images from tensor. """

    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()