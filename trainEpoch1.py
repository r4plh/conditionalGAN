# train.py
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
    "run_name": "run-facenet-epoch1-save",
    "num_epochs": 1,      
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train():
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    dataloader = get_celeba_dataloader(batch_size=config["batch_size"], image_size=config["image_size"])

    encoder = FaceNetEncoder(device=device)
    generator = Generator(noise_dim=config["noise_dim"], embedding_dim=config["embedding_dim"]).to(device)
    discriminator = Discriminator(embedding_dim=config["embedding_dim"]).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config["g_lr"], betas=(config["beta1"], config["beta2"]))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["d_lr"], betas=(config["beta1"], config["beta2"]))
    

    real_label = 0.9 
    fake_label = 0.0

    fixed_noise = torch.randn(config["batch_size"], config["noise_dim"], device=device)
    fixed_embeddings = None 

    print("--- Starting Training Run (to save at batch 1000 and end of epoch 1) ---")
    for epoch in range(config["num_epochs"]):
        for i, batch in enumerate(dataloader):
            real_images = batch['image'].to(device)
            current_batch_size = real_images.size(0)
            
            with torch.no_grad():
                real_embeddings = encoder(real_images)

            if fixed_embeddings is None:
                fixed_embeddings = real_embeddings
                wandb.log({"Real Images": wandb.Image(make_grid(real_images, normalize=True))})


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


            generator.zero_grad()
            output_g = discriminator(fake_images, real_embeddings)
            loss_g = criterion(output_g, labels_real)
            loss_g.backward()
            optimizer_g.step()

            if i % config["log_interval"] == 0:
                print(f"[Epoch {epoch+1}/{config['num_epochs']}] [Batch {i}] [D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
                wandb.log({ "Generator Loss": loss_g.item(), "Discriminator Loss": loss_d.item() })

            if i % config["sample_interval"] == 0:
                # ... wandb image logging ...
                pass

            # =================================================================
            # NEW: SAVE CHECKPOINT AT BATCH 1000
            # =================================================================
            if i == 1000:
                print(f"\n--- Reached batch {i}. Saving mid-epoch checkpoint... ---")
                torch.save(
                    generator.state_dict(),
                    os.path.join(config["checkpoint_dir"], f"generator_batch_{i}.pth")
                )
                print("Mid-epoch model saved successfully.\n")
            # =================================================================

        # --- SAVE CHECKPOINT AT THE END OF THE EPOCH ---
        print(f"\n--- End of Epoch {epoch+1}. Saving models... ---")
        torch.save(
            generator.state_dict(), 
            os.path.join(config["checkpoint_dir"], f"generator_epoch_{epoch+1}.pth")
        )
        print("End-of-epoch models saved successfully.\n")

    print("--- Training Finished. ---")
    wandb.finish()

if __name__ == '__main__':
    train()