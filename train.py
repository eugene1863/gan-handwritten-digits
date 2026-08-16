"""Train the conditional GAN on MNIST and save the generator weights.

Usage:
    python train.py --epochs 50 --batch-size 128
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Discriminator, Generator, LATENT_DIM, NUM_CLASSES

CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"


def get_dataloader(batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def save_sample_grid(generator: Generator, device: torch.device, epoch: int) -> None:
    generator.eval()
    with torch.no_grad():
        n_per_digit = 8
        labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_digit)
        noise = torch.randn(labels.size(0), LATENT_DIM, device=device)
        imgs = generator(noise, labels)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    save_image(imgs, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", nrow=n_per_digit, normalize=True)
    generator.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = get_dataloader(args.batch_size)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    adversarial_loss = nn.BCELoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # --- Train Generator ---
            opt_g.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            gen_imgs = generator(noise, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_g.step()

            # --- Train Discriminator ---
            opt_d.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        save_sample_grid(generator, device, epoch)
        torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator.pth")
        torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator.pth")
        print(f"Saved checkpoint after epoch {epoch}")

    print("Training complete. Generator weights at checkpoints/generator.pth")


if __name__ == "__main__":
    main()
