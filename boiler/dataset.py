import math
from pathlib import Path

import torch
import torch.utils.data

import scipy.io.wavfile


def wav_index(wav_dir=Path('/home/proger/coub-crawler/monthlyLog/wav'), epoch=0):
    wavi = [WavFile(wav) for wav in wav_dir.glob('*.wav')]
    wavk = {wavf.filename.stem: i for i, wavf in enumerate(wavi)}
    return wavi, wavk


class WavFile(torch.utils.data.Dataset):
    def __init__(self, filename, sampling_rate=16000, length=2**17): # 9*16000
        self.filename = filename
        self.sampling_rate = sampling_rate
        self.length = length

    def __getitem__(self, i, augment=True):
        sr, y = scipy.io.wavfile.read(self.filename, mmap=True)
        assert sr == self.sampling_rate

        sample = torch.from_numpy(y).float()/float(1<<15)
        # repeat and/or truncate
        sample = sample.repeat(math.ceil(self.length / len(sample)))[:self.length]
        sample = sample.unsqueeze(0)

        if augment:
            amplitude = 0.3 + torch.rand(1)
            return sample * amplitude
        else:
            return sample

    def __len__(self):
        return 1