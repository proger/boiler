import argparse
from pathlib import Path

import annoy
import torch
from tqdm import tqdm

import boiler.vqvae as vqvae
from boiler.mel import Audio2Mel
from boiler.dataset import WavFile
from boiler.train_vqvae import make_dataloader


def main(args):
    if not args.save_path.parent.exists():
        raise Exception('save_path parent must exist. try mkdir?', args.save_path)

    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).to(args.device)
    loader = make_dataloader(args.wav_dir, shuffle=False, batch_size=args.batch_size)
    model = vqvae.VQVAE2(in_channel=1).to(args.device)

    model.load_state_dict(torch.load(args.load_path, map_location=args.device))

    bags = []
    for batch_number, batch in enumerate(tqdm(loader)):
        zs = model.encode(fft(batch.cuda()).unsqueeze(1))

        quant_t = zs[0]
        quant_t = quant_t.mean(dim=-2,keepdim=True).view(args.batch_size,-1).detach().cpu()
        bags.append(quant_t)

        del zs
        torch.cuda.empty_cache()

    bags = torch.cat(bags).cpu()

    index = annoy.AnnoyIndex(512, 'euclidean')

    for i, vec in enumerate(bags):
        #index.add_item(i, torch.tensor(vec).view(-1))
        index.add_item(i, vec)

    index.build(100)
    index.save(str(args.save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--wav_dir", type=Path, required=True)
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True, help="annoy index path")
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()
    print(args)
    main(args)