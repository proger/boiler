import argparse
import gc
from pathlib import Path
import pickle
from typing import Tuple

import annoy
from joblib import Parallel, delayed
import torch
import torch.jit
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # https://stackoverflow.com/a/61624923
except ImportError:  # tensorflow is not installed, it's ok
    pass

import boiler.dataset
import boiler.encoder


def make_embeddings(args):
    """
    Make embeddings from 'args.wav_dir', save them into Annoy and TF Projector and JIT-compile their model.
    """
    coubs = sorted(args.wav_dir.glob('*.wav'))
    coub_dataset = ConcatDataset([boiler.dataset.WavFile(wav) for wav in coubs])
    coub_loader = DataLoader(coub_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = getattr(boiler.encoder, args.encoder)(args.pt_path, args.device)

    embeddings = []
    for batch in tqdm(coub_loader):
        embeddings.append(model(batch.to(args.device)).detach().cpu())
        gc.collect()
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings)

    model_dir = args.pt_path.with_suffix('') / str(args.encoder)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(model_dir / 'encoder.pt'))

    writer = SummaryWriter(log_dir=model_dir)
    writer.add_embedding(embeddings, global_step=None, metadata=[f'https://coub.com/view/{p.stem}' for p in coubs])

    return model_dir, embeddings


def make_index(embeddings: torch.FloatTensor, model_dir: Path, n_trees: int) -> Tuple[annoy.AnnoyIndex, Path]:
    index = annoy.AnnoyIndex(embeddings.size(1), 'angular')
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)
    index.build(n_trees)
    save_path = model_dir / 'annoy'
    index.save(str(save_path))
    return index, save_path


def musicnn_penultimate(args):
    import musicnn.extractor
    import numpy as np

    index = annoy.AnnoyIndex(200, 'euclidean')

    def slow_embed(z):
        i, wav = z
        try:
            x = musicnn.extractor.extractor(str(wav))
            with open(f'musicnn/{wav.stem}.pickle', 'wb') as f:
                pickle.dump(x, f)
            x = x[2]['penultimate'].mean(axis=0)
            x = x / np.linalg.norm(x)
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
    parser.add_argument("--n_trees", type=int, default=100, help="number of annoy trees (see https://github.com/spotify/annoy#tradeoffs)")
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
        model_dir, embeddings = make_embeddings(args)
        make_index(embeddings, model_dir, args.n_trees)
        print(f'export BOILER_MODEL_DIR={model_dir}')