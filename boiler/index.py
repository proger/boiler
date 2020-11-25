import argparse
import contextlib
import gc
import io
from pathlib import Path
import pickle

import annoy
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# https://stackoverflow.com/a/61624923
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from torch.utils.tensorboard import SummaryWriter

import boiler.vqvae as vqvae
from boiler.mel import Audio2Mel
import boiler.dataset
from boiler.dataset import WavFile
from boiler.train_vqvae import make_dataloader
from torch.utils.data import DataLoader, ConcatDataset

import musicnn.extractor


class MeanFreqTopVQVAE(nn.Module):
    def __init__(self, pt_path, device):
        super().__init__()
        self.fft = Audio2Mel(n_mel_channels=80).to(device)
        self.model = vqvae.VQVAE2(in_channel=1).to(device)
        self.model.load_state_dict(torch.load(pt_path, map_location=args.device))

    def forward(self, batch):
        zs = self.model.encode(self.fft(batch).unsqueeze(1))
        quant_t = zs[0]
        quant_t = quant_t.mean(dim=-2, keepdim=True).view(batch.size(0), -1)
        return quant_t


class BagTopVQVAE(nn.Module):
    def __init__(self, pt_path, device):
        super().__init__()

        self.fft = Audio2Mel(n_mel_channels=80).to(device)
        self.model = vqvae.VQVAE2(in_channel=1).to(device)
        self.model.load_state_dict(torch.load(pt_path, map_location=args.device))

    def forward(self, batch):
        return self.model.encode_bag_t(self.fft(batch).unsqueeze(1), normalize=True)[0]


class BagBottomVQVAE(nn.Module):
    def __init__(self, pt_path, device):
        super().__init__()

        self.fft = Audio2Mel(n_mel_channels=80).to(device)
        self.model = vqvae.VQVAE2(in_channel=1).to(device)
        self.model.load_state_dict(torch.load(pt_path, map_location=args.device))

    def forward(self, batch):
        return self.model.encode_bag(self.fft(batch).unsqueeze(1), normalize=True)[0]


def main(args):
    coubs = sorted(args.wav_dir.glob('*.wav'))
    coub_dataset = ConcatDataset([boiler.dataset.WavFile(wav) for wav in coubs])
    coub_loader = DataLoader(coub_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = globals()[args.encoder](args.pt_path, args.device)

    embeddings = []
    for coub in tqdm(coub_loader):
        embeddings.append(model(coub.to(args.device)).detach().cpu())
        gc.collect()
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings)

    writer = SummaryWriter(log_dir='exp_emb/')
    writer.add_embedding(embeddings, global_step=1, metadata=[f'https://coub.com/view/{p.stem}' for p in coubs])

    index = annoy.AnnoyIndex(embeddings.size(1), 'angular')
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)

    index.build(100)
    save_path = args.pt_path.with_suffix('.annoy')
    print('annoy index saved to', save_path)
    index.save(str(save_path))


def musicnn_penultimate(args):
    index = annoy.AnnoyIndex(200, 'euclidean')

    def slow_embed(z):
        i, wav = z
        try:
            x = musicnn.extractor.extractor(str(wav))
            with open(f'musicnn/{wav.stem}.pickle', 'wb') as f:
                pickle.dump(x, f)
            x = x[2]['penultimate'].mean(axis=0)
            x = x/np.linalg.norm(x)
            return i, x
        except UnboundLocalError:
            return i, None

    results = Parallel(n_jobs=6)(delayed(slow_embed)(z) for z in tqdm(enumerate(sorted(args.wav_dir.glob('*.wav')))))

    for i, x in results:
        if x is None:
            continue
        index.add_item(i, x)

    index.build(100)
    index.save('musicnn.annoy')
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--wav_dir", type=Path, required=True)
    parser.add_argument("--pt_path", type=Path, required=False, help="pytorch checkpoint")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('encoder', default='BagTopVQVAE', choices=['MeanFreqTopVQVAE', 'BagTopVQVAE', 'BagBottomVQVAE', 'musicnn'])

    args = parser.parse_args()
    print(args)

    if args.encoder == 'musicnn':
        musicnn_penultimate(args)
    else:
        assert args.pt_path is not None
        main(args)