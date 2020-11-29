import itertools as it
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import streamlit as st
import torch

import boiler.dataset
import boiler.api.nearest

logger.add("boiler.log", enqueue=True, serialize=True)
#st.set_page_config(layout='wide')

model_dir = st.text_input('BOILER_MODEL_DIR', value='exp/p_t64_b512/vqvae_223/BagTopVQVAE')
model_dir = Path(model_dir)

wav_dir = st.text_input('wav_dir', value='/home/proger/coub-crawler/monthlyLog/wav')
wav_dir = Path(wav_dir)

index = boiler.api.nearest.Nearest(model_dir)
encoder = torch.jit.load(str(model_dir / 'encoder.pt'))

coub_id = st.text_input('query coub id', value='2b5fyy')
logger = logger.bind(query_coub_id=coub_id)

wav_filename = wav_dir / f'{coub_id}.wav'

st.audio(str(wav_filename))
wav_tensor = boiler.dataset.WavFile(wav_filename)[0]
batch = wav_tensor.unsqueeze(0)
batch = batch.to(next(encoder.parameters()).device)

query = encoder(batch)
fig = plt.figure(figsize=(10,1))
fig.gca().matshow(query.detach().cpu(), aspect=1)
st.pyplot(fig)
query = query.squeeze()

st.markdown('## results')

columns = st.beta_columns(3)
columns = it.cycle(columns)

results = index.search(query)

for i, item in enumerate(results):
    dist = item['distance']
    coub_id = item['coub_id']
    column = next(columns)

    column.write(coub_id)
    column.audio(str(wav_dir / f"{coub_id.split('/')[-1]}.wav"))

    if column.button('😍', key=f'nei-pos-{i}'):
        print(i, item, 'pos')
        logger.bind(neighbour_coub_id=coub_id, dist=dist).info('positive')
    if column.button('🤔', key=f'nei-neg-{i}'):
        logger.bind(neighbour_coub_id=coub_id, dist=dist).info('negative')
