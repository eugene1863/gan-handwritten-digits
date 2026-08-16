"""Conditional DCGAN for MNIST handwritten digits.

The generator takes a latent noise vector plus a digit label (0-9) and
produces a 28x28 grayscale image. Conditioning lets the API request a
specific digit instead of a random one.
"""
import torch
import torch.nn as nn

LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SIZE = 28


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = IMG_SIZE // 4  # 7
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 14x14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gen_input = torch.cat((self.label_emb(labels), noise), dim=1)
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, IMG_SIZE * IMG_SIZE)

        def block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                layers.insert(1, nn.BatchNorm2d(out_ch, 0.8))
            return layers

        self.conv = nn.Sequential(
            *block(2, 16, bn=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        ds_size = IMG_SIZE
        for _ in range(4):  # 4 stride-2 convs, kernel=3, padding=1
            ds_size = (ds_size + 2 * 1 - 3) // 2 + 1
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_emb(labels).view(labels.size(0), 1, IMG_SIZE, IMG_SIZE)
        d_in = torch.cat((img, label_map), dim=1)
        out = self.conv(d_in)
        out = out.view(out.size(0), -1)
        return self.adv_layer(out)
