import torch
from torch import nn


def crop(image, new_shape):
    """
    Center-crop the image. 
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    """

    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - round(new_shape[2] / 2)
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - round(new_shape[3] / 2)
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]

    return cropped_image


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    """
    Return the loss of the generator given inputs. 
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    """

    fakes = gen(condition)

    disc_fakes = disc(fakes, condition)

    labels = torch.ones_like(disc_fakes)

    ad_loss = adv_criterion(disc_fakes, labels)
    rec_loss = recon_criterion(fakes, real)

    gen_loss = ad_loss + lambda_recon * rec_loss

    return gen_loss



def weights_init(m):
    """ Init weights used by disc and gen. """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)