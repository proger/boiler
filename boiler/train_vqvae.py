import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils

from tqdm import tqdm

from .dataset import WavFile
from .mel import Audio2Mel
from .import vqvae

steps = 0

def train(args, epoch, loader, model, optimizer, scheduler, fft, writer):
    loader = tqdm(loader)

    criterion = nn.L1Loss()

    sample_size = 25

    for i, waveform in enumerate(loader):
        model.zero_grad()

        waveform = waveform.cuda()
        img = fft(waveform).detach().to(args.device).unsqueeze(1)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        recon_loss = recon_loss.mean()
        latent_loss = latent_loss.mean()
        loss = recon_loss + args.latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; recon: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; "
                f"lr: {lr:.5f}"
            )
        )

        global steps
        writer.add_scalar("loss/latent", latent_loss.item(), steps)
        writer.add_scalar("loss/recon", recon_loss.item(), steps)
        writer.add_scalar("lr", lr, steps)

        if i % 1000 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            vertical_bar = torch.zeros(sample_size, 1, args.n_mel_channels, 5)
            demo = torch.cat([sample.cpu(), vertical_bar, out.cpu()], dim=-1)
            demo = torchvision.utils.make_grid(demo, nrow=1, normalize=True)

            writer.add_image('sample', demo, steps)

            model.train()

        steps += 1


def make_dataloader(wav_dir: Path, shuffle: bool = False, batch_size: int = 128) -> DataLoader:
    dataset = ConcatDataset([WavFile(wav) for wav in wav_dir.glob('*.wav')])
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle)


def main(args):
    root = args.save_path
    root.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(root))

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(args.device)
    loader = make_dataloader(args.wav_dir, shuffle=True)
    model = vqvae.VQVAE2(in_channel=1).to(args.device)

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        total_steps=len(loader) * args.epoch
    )

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, fft, writer)

        torch.save(model.state_dict(), root / f"vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--wav_dir", type=Path, required=True)
    parser.add_argument("--load_path", type=Path)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--latent_loss_weight", type=float, default=0.5, help="""
        Quote from https://sunniesuhyoung.github.io/files/vqvae.pdf:

        [Wu and Flierl, 2018] gives the following intuition: If we increase the value of λ, the vector quantizer becomes
        “more powerful.” The quantization error is minimized, and the codewords are pushed far away from each other.
        This may decrease the reconstruction error, but leads to a bad generalization ability. On the other hand, a
        smaller value of λ creates a “weaker” vector quantizer and the quantization error increases. This creates similar
        effects as the low rate setting of the vector quantizer, where the locality of the data space is better preserved.
        In short, with λ < 1, we can learn features that better preserve the similarity relations of the data space.
    """)


    args = parser.parse_args()
    print(args)
    main(args)