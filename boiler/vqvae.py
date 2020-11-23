import torch
from torch import nn
from torch.nn import functional as F

from .quantize import Quantize

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch




class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class VQVAE2(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=4,
        n_res_channel=32,
        embed_dim=64,
        n_embed=128,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = nn.Sequential(
            nn.Conv2d(in_channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True)
        )

        self.enc_t = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True)
        )

        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)

        self.dec_t = nn.Sequential(
            nn.Conv2d(embed_dim, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, embed_dim, kernel_size=4, stride=2, padding=1)
        )

        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )

        self.dec = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel // 2, in_channel, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def encode_bag(self, input):
        _, _, _, _, id_b = self.encode(input)

        n_embed = self.quantize_b.n_embed
        id_b_flat = id_b.view(input.size(0), -1)
        bag = torch.zeros(input.size(0), n_embed).long().cuda()
        bag.scatter_add_(dim=1, index=id_b_flat, src=torch.ones_like(id_b_flat))
        bag = F.normalize(bag.float(), p=2)
        return bag, id_b

    def encode_bag_t(self, input, normalize=True):
        _, _, _, id_t, _ = self.encode(input)

        n_embed = self.quantize_t.n_embed
        flat = id_t.view(input.size(0), -1)
        bag = torch.zeros(input.size(0), n_embed).long().cuda()
        bag.scatter_add_(dim=1, index=flat, src=torch.ones_like(flat))
        if normalize:
            bag = F.normalize(bag.float(), p=2)
        return bag, id_t


    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


