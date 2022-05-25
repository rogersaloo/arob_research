import torch
from dataset import NormalPneumonia
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import fid_config

writer_real= SummaryWriter(f'runs/GAN/test_real')
writer_fake= SummaryWriter(f'runs/GAN/test_fake')

dataset = NormalPneumonia(
        root_pneumonia=config.TRAIN_DIR+"/normal", root_normal=config.TRAIN_DIR+"/pneumonia", transform=config.transforms
    )
    # val_dataset = PneumonianormalDataset(
    #    root_pneumonia="cyclegan_test/pneumonias", root_normal="cyclegan_test/normals", transform=config.transforms
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
    )

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    global fake_pneumonia

    for idx, (normal, pneumonia) in enumerate(loop):
        normal = normal.to(config.DEVICE)
        pneumonia = pneumonia.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_pneumonia = gen_H(normal)
            D_H_real = disc_H(pneumonia)
            D_H_fake = disc_H(fake_pneumonia.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_normal = gen_Z(pneumonia)
            D_Z_real = disc_Z(normal)
            D_Z_fake = disc_Z(fake_normal.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_pneumonia)
            D_Z_fake = disc_Z(fake_normal)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_normal = gen_Z(fake_pneumonia)
            cycle_pneumonia = gen_H(fake_normal)
            cycle_normal_loss = l1(normal, cycle_normal)
            cycle_pneumonia_loss = l1(pneumonia, cycle_pneumonia)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_normal = gen_Z(normal)
            identity_pneumonia = gen_H(pneumonia)
            identity_normal_loss = l1(normal, identity_normal)
            identity_pneumonia_loss = l1(pneumonia, identity_pneumonia)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_normal_loss * config.LAMBDA_CYCLE
                + cycle_pneumonia_loss * config.LAMBDA_CYCLE
                + identity_pneumonia_loss * config.LAMBDA_IDENTITY
                + identity_normal_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 2 == 0:
            save_image(fake_pneumonia*0.5+0.5, f"saved_images/{fid_config.NORMAL_IMAGE_LABEL}_{idx}.png")
            save_image(fake_normal*0.5+0.5, f"saved_images/{fid_config.PNEUMONIA_IMAGE_LABEL}_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))

# img_grid = torchvision.utils.make_grid(fake_pneumonia train_fn.__globals__)
# writer_real.add_image('normal', img_grid)
# writer_real.close()
def main():
    global loader
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
    #     )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()

