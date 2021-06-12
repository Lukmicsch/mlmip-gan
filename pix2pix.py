import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.torch_utils import get_loss_fn
from data.data_manager import DataManager
from algorithms.pix2pix.pix2pix_utils import get_gen_loss
from algorithms.algorithm_utils import weights_init
from algorithms.pix2pix.unet import UNet
from algorithms.pix2pix.pix2pix_discriminator import Discriminator
from utils.torch_utils import show_tensor_images_pix2pix


def run_pix2pix_train(config):
    # Init
    adv_criterion = get_loss_fn(config['recon_criterion'])
    recon_criterion = get_loss_fn(config['recon_criterion'])
    lambda_recon = config['lambda_recon']

    n_epochs = config['n_epochs']
    input_dim = config['channels'] # TODO: is this correct? was 3
    real_dim = config['channels'] # TODO: is this correct? was 3
    display_step = config['display_step']
    batch_size = config['batch_size']
    lr = config['lr']
    target_shape = config['width_and_height']
    device = config['device']

    save_model = config['save_model']

    data_manager = DataManager(config)
    full_cases = data_manager.get_full_cases()

    dataset = data_manager.get_dataset_2d(full_cases)

    gen = UNet(input_dim, real_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator(input_dim + real_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(dataloader):
            
            # ME!
            image = data_manager.prepare_image_batch(image).to(device)
            print(image.shape)
            exit(1)
            
            """
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            condition = nn.functional.interpolate(condition, size=target_shape)
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=target_shape)
            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device) """

            ### Update discriminator ###
            disc_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition)  # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)  # Update gradients
            disc_opt.step()  # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward()  # Update gradients
            gen_opt.step()  # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(
                        f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({'gen': gen.state_dict(),
                                'gen_opt': gen_opt.state_dict(),
                                'disc': disc.state_dict(),
                                'disc_opt': disc_opt.state_dict()
                                }, f"pix2pix_{cur_step}.pth")
            cur_step += 1