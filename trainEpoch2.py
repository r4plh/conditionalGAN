# train2.py - Continue training from epoch 1
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import wandb
import os

from dataset import get_celeba_dataloader
from models import Generator, Discriminator
from encoder import FaceNetEncoder

config = {
    "project_name": "conditional-face-gan-pytorch",
    "run_name": "run-facenet-epoch2-continue",
    "num_epochs": 1,      # Train 1 more epoch
    "batch_size": 64,
    "image_size": 128,
    "noise_dim": 100,
    "embedding_dim": 512,
    "g_lr": 0.0002,
    "d_lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "log_interval": 25,
    "sample_interval": 100,
    "checkpoint_dir": "./checkpoints",
}

def train():
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    dataloader = get_celeba_dataloader(batch_size=config["batch_size"], image_size=config["image_size"])

    encoder = FaceNetEncoder(device=device)
    generator = Generator(noise_dim=config["noise_dim"], embedding_dim=config["embedding_dim"]).to(device)
    discriminator = Discriminator(embedding_dim=config["embedding_dim"]).to(device)

    # ===== LOAD CHECKPOINT FROM EPOCH 1 =====
    print("--- Loading checkpoint from epoch 1 ---")
    generator.load_state_dict(torch.load(f"{config['checkpoint_dir']}/generator_epoch_1.pth", map_location=device))
    # Load discriminator if exists, otherwise use fresh one
    try:
        discriminator.load_state_dict(torch.load(f"{config['checkpoint_dir']}/discriminator_epoch_1.pth", map_location=device))
        print("✓ Both models loaded from epoch 1")
    except:
        print("✓ Generator loaded, using fresh discriminator")

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config["g_lr"], betas=(config["beta1"], config["beta2"]))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["d_lr"], betas=(config["beta1"], config["beta2"]))
    
    real_label = 0.9 
    fake_label = 0.0
    
    # Start from epoch 2
    start_epoch = 1
    
    print("--- Starting Training (Epoch 2) ---")
    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        for i, batch in enumerate(dataloader):
            real_images = batch['image'].to(device)
            current_batch_size = real_images.size(0)
            
            with torch.no_grad():
                real_embeddings = encoder(real_images)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images, real_embeddings)
            labels_real = torch.full((current_batch_size,), real_label, device=device)
            loss_d_real = criterion(output_real, labels_real)
            loss_d_real.backward()

            noise = torch.randn(current_batch_size, config["noise_dim"], device=device)
            fake_images = generator(noise, real_embeddings)
            output_fake = discriminator(fake_images.detach(), real_embeddings)
            labels_fake = torch.full((current_batch_size,), fake_label, device=device)
            loss_d_fake = criterion(output_fake, labels_fake)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()

            # Train Generator
            generator.zero_grad()
            output_g = discriminator(fake_images, real_embeddings)
            loss_g = criterion(output_g, labels_real)
            loss_g.backward()
            optimizer_g.step()

            if i % config["log_interval"] == 0:
                print(f"[Epoch {epoch+1}] [Batch {i}] [D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
                wandb.log({"Generator Loss": loss_g.item(), "Discriminator Loss": loss_d.item()})

            if i % config["sample_interval"] == 0:
                with torch.no_grad():
                    fake_sample = generator(noise[:16], real_embeddings[:16])
                    wandb.log({"Generated Images": wandb.Image(make_grid(fake_sample, normalize=True))})

        # Save checkpoint at end of epoch
        print(f"\n--- End of Epoch {epoch+1}. Saving models... ---")
        torch.save(generator.state_dict(), os.path.join(config["checkpoint_dir"], f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(config["checkpoint_dir"], f"discriminator_epoch_{epoch+1}.pth"))
        print("Models saved successfully.\n")

    print("--- Training Finished (Epoch 2 Complete) ---")
    wandb.finish()

if __name__ == '__main__':
    train()