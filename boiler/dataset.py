import csv
import math
import os
from pathlib import Path
import sys
from typing import NamedTuple

import torch
import torch.utils.data
import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

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

        # amplitude normalization
        sample /= sample.max()
        return sample

    def __len__(self):
        return 1


class MTTItem(NamedTuple):
    clip_id: int
    tags: torch.LongTensor
    path: Path


class MagnaTagATune(torch.utils.data.Dataset):
    """
    http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset

    splits due to https://github.com/keunwoochoi/magnatagatune-list
    """

    def __init__(self, root: str, download: bool = False, mode: str = 'train'):
        root = self.root = Path(root)
        assert root.is_dir()

        if download:
            print("downloading and verifying", file=sys.stderr)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001', root,
                hash_value='179c91c8c2a6e9b3da3d4e69d306fd3b', hash_type='md5', resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002', root,
                hash_value='acf8265ff2e35c6ff22210e46457a824', hash_type='md5', resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003', root,
                hash_value='582dc649cabb8cd991f09e14b99349a5', hash_type='md5', resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.csv', root,
                hash_value='03ef3cb8ddcfe53fdcdb8e0cda005be2', hash_type='md5', resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv', root,
                hash_value='f04fa01752a8cc64f6e1ca142a0fef1d', hash_type='md5', resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/comparisons_final.csv', root, resume=True)

            download_url('http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3_echonest_xml.zip', root,
                hash_value='09be4ac8c682a8c182279276fadb37f9', hash_type='md5', resume=True)


            mp3s = root / 'mp3s.zip'
            if not mp3s.exists():
                print("cating mp3s", file=sys.stderr)
                os.system('cat {}/mp3.zip.* > {}'.format(root, mp3s))
            print("extracting {}".format(mp3s), file=sys.stderr)
            extract_archive(mp3s, root)
            #mp3s.unlink()
            echonest = root / 'mp3_echonest_xml.zip'
            print("extracting {}".format(echonest), file=sys.stderr)
            extract_archive(echonest, root)

        # https://github.com/jordipons/musicnn/blob/820bdeda16eecedb34f1388c4f42424238e33a48/musicnn/configuration.py#L11
        self.labels = ('guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral')
        self.num_labels = len(self.labels)
        labels_dict = {label: i for i, label in enumerate(self.labels)}

        with open(root / 'annotations_final.csv') as f:
            reader = csv.reader(f, delimiter='\t')
            _, *tag_labels, _ = next(reader)

            def make_label(tags):
                units = [torch.eye(self.num_labels)[labels_dict[synonym]] for synonym, value in zip(tag_labels, tags) if synonym in labels_dict and int(value)]
                if not units:
                    return torch.zeros(self.num_labels)
                else:
                    return torch.stack(units).sum(dim=0).float()

            items = tuple(item for item in [MTTItem(line[0], make_label(line[1:-1]), self.root / line[-1]) for line in reader]
                            if item.path.stat().st_size > 0)

        self.items = items
        self.sample_rate = 16000


    def __getitem__(self, i):
        item = self.items[i]
        wav, sr = torchaudio.load(item.path)
        assert sr == 16000
        return wav, item.tags

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    print(len(MagnaTagATune(root='/tank/coub', download=True)))