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
from .quantize import Quantize
from .vqvae import VQVAE

steps = 0

def train(epoch, loader, model, optimizer, scheduler, device, fft, root, writer):
    loader = tqdm(loader)

    criterion = nn.L1Loss()

    latent_loss_weight = 2
    sample_size = 25

    for i, waveform in enumerate(loader):
        model.zero_grad()

        waveform = waveform.cuda()
        #img = waveform
        img = fft(waveform).detach().to(device).unsqueeze(1)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        recon_loss = recon_loss.mean()
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
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

            demo = torch.cat([sample, out], 0).cpu()
            demo = torchvision.utils.make_grid(demo, nrow=2, normalize=True)

            writer.add_image('sample', demo, steps)

            model.train()

        steps += 1


def main(args):
    device = "cuda"

    root = args.save_path
    root.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(root))

    dataset = ConcatDataset([WavFile(wav) for wav in args.data_path.glob('*.wav')])
    loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda()

    model = VQVAE(in_channel=1).to(device)

    if args.load_path:
        last = list(args.load_path.glob('*.pt'))[-1]
        model.load_state_dict(torch.load(last, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        total_steps=len(loader) * args.epoch
    )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, fft, root, writer)

        torch.save(model.state_dict(), root / f"vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str, default='default')
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--load_path", type=Path)
    parser.add_argument("--save_path", type=Path, required=True)

    args = parser.parse_args()
    print(args)

    main(args)