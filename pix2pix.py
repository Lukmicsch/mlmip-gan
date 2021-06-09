import torch
import torch.nn.functional as F
from torch import nn

from algorithms.algorithm_utils import get_loss



def run_pix2pix_train(config):
    # Init
    adv_criterion = get_loss(config['recon_criterion'])
    recon_criterion = get_loss(config['recon_criterion'])
    lambda_recon = config['lambda_recon']

    n_epochs = config['n_epochs']
    input_dim = config['channels'] # TODO: is this correct? was 3
    real_dim = config['channels'] # TODO: is this correct? was 3
    display_step = config['display_step']
    batch_size = config['batch_size']
    lr = config['lr']
    target_shape = config['width_and_height']
    device = config['device']