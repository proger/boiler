# boiler

## ~~Craving~~Crawling for Coubs

I used a slightly patched version of https://github.com/flute/coub-crawler to download almost 24 hours of coubs in one run.

I've discovered that downloaded audios are truncated by the video length
however it's common that coub audio tracks are longer than video clips themselves.
I'm going to ignore that issue for now.

I used ffmpeg to convert mp4s to wavs:

```bash
parallel -j6 -n1  ffmpeg -nostdin -i {} -vn -ar 16000 -ac 1 wav/{/.}.wav ::: video/*.mp4
```

Based on the [distribution](https://github.com/glamp/bashplotlib) of audio lengths I've decided to pad each audio clip to 9s by repeating
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
% python3 -m boiler.train_vqvae --wav_dir /home/proger/coub-crawler/monthlyLog/wav --latent_loss_weight 0.5 --save_path exp/p_t64_b512 [--load_path exp/o_vectorquant_bigger_a/vqvae_084.pt]
```

To produce wav vectors I use a bag of the codebook words produced by the top level embedding.
There are some other encoders in [boiler/encoder.py](boiler/encoder.py).

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

## API

```
BOILER_INDEX_FILE=exp/p_t64_b512/vqvae_223/BagTopVQVAE/annoy uvicorn boiler.api.web:app --host 0.0.0.0 --port 8000 --workers 6
```

Visit http://localhost:8000/docs for the API overview.