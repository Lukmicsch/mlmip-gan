from torch import nn



def crop(image, new_shape):
    """ Center-crop the image. """

    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - round(new_shape[2] / 2)
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - round(new_shape[3] / 2)
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image



def get_loss(loss_fn):
    """ Return loss function specified in config. """

    criterion = None

    if loss_fn == "L1":
        criterion = nn.L1Loss
    elif loss_fn == "BCE":
        criterion = nn.BCEWithLogitsLoss()

    return criterion