import argparse
import gc
from pathlib import Path
import pickle

import annoy
from joblib import Parallel, delayed
import torch
import torch.jit
from tqdm import tqdm
import numpy as np

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile # https://stackoverflow.com/a/61624923

from torch.utils.tensorboard import SummaryWriter

import boiler.dataset
from torch.utils.data import DataLoader, ConcatDataset

import boiler.encoder

def main(args):
    coubs = sorted(args.wav_dir.glob('*.wav'))
    coub_dataset = ConcatDataset([boiler.dataset.WavFile(wav) for wav in coubs])
    coub_loader = DataLoader(coub_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = getattr(boiler.encoder, args.encoder)(args.pt_path, args.device)

    embeddings = []
    for coub in tqdm(coub_loader):
        embeddings.append(model(coub.to(args.device)).detach().cpu())
        gc.collect()
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings)

    index_dir = args.pt_path.with_suffix('') / str(args.encoder)
    torch.jit.script(model).save(str(index_dir / 'encoder.pt'))

    writer = SummaryWriter(log_dir=index_dir)
    writer.add_embedding(embeddings, global_step=None, metadata=[f'https://coub.com/view/{p.stem}' for p in coubs])

    index = annoy.AnnoyIndex(embeddings.size(1), 'angular')
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)

    index.build(100)
    save_path = index_dir / 'annoy'
    print('annoy index saved to', save_path)
    index.save(str(save_path))


def musicnn_penultimate(args):
    import musicnn.extractor

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