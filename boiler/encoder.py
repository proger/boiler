import argparse
from pathlib import Path

import torch
import torch.nn as nn

from boiler.mel import Audio2Mel
from boiler import vqvae


class VQVAEEncoder(nn.Module):
    def __init__(self, pt_path, device):
        super().__init__()
        self.fft = Audio2Mel(n_mel_channels=80).to(device)
        self.model = vqvae.VQVAE2(in_channel=1, n_embed=(64,512)).to(device)
        self.model.load_state_dict(torch.load(pt_path, map_location=device))


class MeanFreqTopVQVAE(VQVAEEncoder):
    def forward(self, batch):
        zs = self.model.encode(self.fft(batch).unsqueeze(1))
        quant_t = zs[0]
        quant_t = quant_t.mean(dim=-2, keepdim=True).view(batch.size(0), -1)
        return quant_t


class BagTopVQVAE(VQVAEEncoder):
    def forward(self, batch):
        return self.model.encode_bag_t(self.fft(batch).unsqueeze(1), normalize=True)[0]


class BagBottomVQVAE(VQVAEEncoder):
    def forward(self, batch):
        return self.model.encode_bag(self.fft(batch).unsqueeze(1), normalize=True)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=Path, required=False, help="VQVAE checkpoint")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_path", type=Path, help="jit script location")
    parser.add_argument('encoder', default='BagTopVQVAE', choices=['MeanFreqTopVQVAE', 'BagTopVQVAE', 'BagBottomVQVAE'])

    args = parser.parse_args()

    class_ = globals()[args.encoder]
    module = class_(args.pt_path, args.device)

    jit_module = torch.jit.script(module)
    jit_module.save(args.save_path)