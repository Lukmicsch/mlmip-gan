import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.data_manager import DataManager
from utils.torch_utils import get_loss_fn
from algorithms.wgan.generator import Generator
from algorithms.wgan.critic import Critic
from algorithms.algorithms_utils import weights_init
from algorithms.wgan.wgan_utils import get_gradient
from algorithms.wgan.wgan_utils import gradient_penalty
from algorithms.wgan.wgan_utils import get_gen_loss
from algorithms.wgan.wgan_utils import get_crit_loss

def run_wgan_train(config):
    # Init
    data_manager = DataManager(config)
    full_cases = data_manager.get_full_cases()

    dataset = data_manager.get_dataset_2d(full_cases)
    
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
    
    c_lambda = config['c_lambda']
    crit_repeats = config['crit_repeats']
    
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    crit = Critic().to(device)
    crit_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    cur_step = 0
    generator_losses = []
    critic_losses = []
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):

            # ME!!
            real = data_manager.prepare_image_batch(real).to(device)

            cur_batch_size = len(real)
            real = real.to(device)

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()

            cur_step += 1