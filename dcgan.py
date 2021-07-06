import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from algorithms.dcgan.discriminator import Discriminator
from algorithms.dcgan.generator import Generator
from algorithms.dcgan256.discriminator import Discriminator256
from algorithms.dcgan256.generator import Generator256
from algorithms.algorithm_utils import get_noise
from utils.torch_utils import show_tensor_images_dcgan
from data.data_manager import DataManager
from algorithms.algorithm_utils import weights_init
from utils.torch_utils import get_loss_fn
from utils.torch_utils import plot_tensor_images

def run_dcgan_train(config):
    # Init
    data_manager = DataManager(config)
    full_cases = data_manager.get_full_cases()

    dataset = data_manager.get_dataset_2d(full_cases)

    criterion = get_loss_fn('BCE')
    z_dim = config['z_dim']
    display_step = config['display_step']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    # A learning rate of 0.0002 works well on DCGAN
    lr = config['lr']

    # These parameters control the optimizer's momentum, which you can read more about here:
    # https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']

    device = config['device']

    save_model = config['save_model']

    gen = Generator256(z_dim).to(device) if config['dcgan256'] else Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator256().to(device) if config['dcgan256'] else Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):

            real = torch.unsqueeze(real.squeeze(), 1).float()

            #real, _ = data_manager.prepare_image_batch(real)

            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()

            ## Update generator ##
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fake_2)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ## Visualization code ##
            #if cur_step % display_step == 0 and cur_step > 0:
            if True:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}, epoch: {epoch}")
                print("real: ", real.shape)
                print("fake: ", fake.shape)
                exit(1)
                show_tensor_images_dcgan(fake)
                show_tensor_images_dcgan(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
