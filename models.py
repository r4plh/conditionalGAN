import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, embedding_dim=512, channels=3):
        super(Generator, self).__init__()
        input_dim = noise_dim + embedding_dim # This will now be 100 + 512

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, embedding):
        combined_input = torch.cat([noise, embedding], dim=1)
        reshaped_input = combined_input.view(-1, combined_input.size(1), 1, 1)
        return self.main(reshaped_input)

class Discriminator(nn.Module):
    def __init__(self, embedding_dim=512, channels=3):
        super(Discriminator, self).__init__()
        
        self.image_path = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # This layer now takes 512 (from image) + 512 (from embedding)
        self.combined_path = nn.Sequential(
            nn.Conv2d(512 + embedding_dim, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, embedding):
        image_features = self.image_path(image)
        embedding_reshaped = embedding.view(-1, embedding.size(1), 1, 1)
        embedding_expanded = embedding_reshaped.expand(-1, -1, image_features.size(2), image_features.size(3))
        combined = torch.cat([image_features, embedding_expanded], dim=1)
        output = self.combined_path(combined)
        return output.view(-1, 1).squeeze(1)