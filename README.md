# boiler

Boiler is a content-based recommender for coubs based on input audio clips. It consists of a VQ-VAE2 encoder that projects short audio clips and Annoy index
that performs nearest-neighbor lookups.

```console
pip3 install git+https://github.com/proger/boiler.git
```

## ~~Craving~~Crawling for Coubs

I used a slightly patched version of https://github.com/flute/coub-crawler to download almost 24 hours of coubs in one run.

I've discovered that downloaded audios are truncated by the video length
however it's common that coub audio tracks are longer than video clips themselves.
I'm going to ignore that issue for now.

I used ffmpeg to convert mp4s to wavs:

```bash
parallel -j6 -n1  ffmpeg -nostdin -i {} -vn -ar 16000 -ac 1 wav/{/.}.wav ::: video/*.mp4
```

Based on the [distribution](https://github.com/glamp/bashplotlib) of audio lengths I've decided to pad each audio clip to 2**17 samples and repeating
shorter clips and truncating longer ones:

```console
proger@rt:~/coub-crawler/monthlyLog$ soxi -D wav/*.wav | hist -b 20 -p 🍄

 5644|           🍄
 5347|           🍄
 5050|           🍄
 4753|           🍄
 4456|           🍄
 4159|           🍄
 3862|           🍄
 3565|           🍄
 3268|           🍄
 2971|           🍄
 2674|           🍄
 2377|           🍄
 2080|           🍄
 1783|           🍄
 1486|           🍄
 1189|           🍄
  892|           🍄
  595|         🍄🍄🍄
  298|       🍄🍄🍄🍄🍄
    1| 🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄🍄

----------------------------------
|            Summary             |
----------------------------------
|       observations: 9344       |
|      min value: 0.162562       |
|        mean : 9.168637         |
|      max value: 20.247812      |
----------------------------------
```

## Training

The model is a [VQ-VAE2](https://github.com/rosinality/vq-vae-2-pytorch) with large 2D convolution kernels over mel-spectrograms.
To train you need a directory with 16-bit signed wav files of length 2**17 samples at rate of 16k samples per second.

```console
% python3 -m boiler.train_vqvae --wav_dir /home/proger/coub-crawler/monthlyLog/wav --save_path exp/p_t64_b512 [--load_path exp/o_vectorquant_bigger_a/vqvae_084.pt]
```

To produce wav vectors I use a bag of the codebook words produced by the top level embedding.
There are some other encoders in [`boiler.encoder`](boiler/encoder.py). Every encoder is an instance of `torch.nn.Module` and expects
a normalized (see `torchaudio.load`) float WAV input of shape `(N, 1, 2**17)` that outputs a vector of `(N, D)` where D is encoder-specific (currently 64).

The encoder is currently around 10M parameters due to large convolution kernels but could probably be made smaller.

The quality of the model is currently checked manually by looking at Tensorflow Projector embeddings, potentially it may be possible to test embeddings
using the GTZAN or MagnaTagATune datasets by defining successful recall as having embedded neighbors of query audios be of the same genre/having the same tags.

## Compiling the Encoder and Indexing the Dataset

Indexing produces a list of url mappings (`metadata.tsv`), a list of tensors in the same order (`tensors.tsv`), a JIT-ready TorchScript module (`encoder.pt`) that
converts single-channel wavs to embeddings.

```console
% python3 -m boiler.index --wav_dir /home/proger/coub-crawler/monthlyLog/wav --pt_path exp/p_t64_b512/vqvae_223.pt BagTopVQVAE

% find exp/p_t64_b512/vqvae_223 -type f
exp/p_t64_b512/vqvae_223/BagTopVQVAE/events.out.tfevents.1606420670.rt.1393854.0
exp/p_t64_b512/vqvae_223/BagTopVQVAE/00000/default/metadata.tsv
exp/p_t64_b512/vqvae_223/BagTopVQVAE/00000/default/tensors.tsv
exp/p_t64_b512/vqvae_223/BagTopVQVAE/projector_config.pbtxt
exp/p_t64_b512/vqvae_223/BagTopVQVAE/encoder.pt
exp/p_t64_b512/vqvae_223/BagTopVQVAE/annoy
exp/p_t64_b512/vqvae_223/BagTopVQVAE/events.out.tfevents.1606498269.rt.2782441.0
```

[`boiler.index`](boiler/index.py) outputs a *model directory* based on the vqvae checkpoint (argument `--pt_path`) and encoder name, so from `--pt_path exp/p_t64_b512/vqvae_223.pt BagTopVQVAE` the model directory will be called `exp/p_t64_b512/vqvae_223/BagTopVQVAE`.

[`boiler.api.nearest.Nearest`](boiler/api/nearest.py) is the module that wraps Annoy and currently assumes all of the metadata (coub urls included in `metadata.tsv`) fits in RAM.

Basic usage is:

```python
from pathlib import Path
import torch
import boiler.api.nearest

model_dir = Path('exp/p_t64_b512/vqvae_223/BagTopVQVAE')

encoder = torch.jit.load(str(model_dir / 'encoder.pt'), map_location='cpu')

x = encoder(torch.randn(1,1,2**17)).squeeze()

index = boiler.api.nearest.Nearest(model_dir)
print(index.search(x))
```

Encoder and index benchmarks are included in [benchmark-annoy.ipynb](benchmark-annoy.ipynb) and [benchmark-encoder.ipynb](benchmark-encoder.ipynb)

## Web API

[`boiler.api.web`](boiler/api/web.py) defineds a basic FastAPI-based frontend to nearest-neighbor search.

```
pip3 install uvicorn # you need this installed separately
BOILER_MODEL_DIR=exp/p_t64_b512/vqvae_223/BagTopVQVAE uvicorn boiler.api.web:app --host 0.0.0.0 --port 8000 --workers 6
```

Increasing the number of workers does not affect memory usage of the index. Batched APIs are currently out of scope.

There is a single API call right now:
- `POST /nearest` accepts a single embedding and returns a list of coubs (the task of producing an embedding is left to the caller)

Visit http://localhost:8000/docs for details on API usage, type information and examples.