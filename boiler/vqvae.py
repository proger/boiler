import torch
from torch import nn
from torch.nn import functional as F

from .quantize import VectorQuant

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
        n_embed=(64, 512),
        decay=0.99,
    ):
        super().__init__()

        # this is only used to int8-quantize the top encoder
        # self.int8_quant = torch.quantization.QuantStub()
        # self.int8_dequant = torch.quantization.DeQuantStub()

        self.enc_b = nn.Sequential(
            nn.Conv2d(in_channel, channel // 2, kernel_size=8, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True)
        )

        self.enc_t = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=20, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True),
        )

        n_embed_t, n_embed_b = (n_embed, n_embed) if isinstance(n_embed, int) else n_embed

        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VectorQuant(1, n_embed_t, embed_dim)

        self.dec_t = nn.Sequential(
            nn.Conv2d(embed_dim, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, embed_dim, kernel_size=21, stride=2, padding=1)
        )

        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuant(1, n_embed_b, embed_dim)

        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=21, stride=2, padding=1)
        )

        self.dec = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim, channel, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel // 2, in_channel, kernel_size=8, stride=2, padding=1)
        )

    def forward(self, input):
        quant_t, quant_b, pen, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, pen

    def encode_top(self, input):
        # input = self.int8_quant(input)
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t)
        quant_t = quant_t.permute(0, 3, 2, 1)  # batch, time, mel, emb
        quant_t, id_t, vq_pen_t, encoder_pen_t, entropy_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 2, 1)
        vq_pen_t = vq_pen_t.unsqueeze(0)

        # quant_t = self.int8_dequant(quant_t)
        return enc_b, quant_t, vq_pen_t.mean(), encoder_pen_t.mean(), id_t

    def encode(self, input):
        enc_b, quant_t, vq_pen_t_mean, encoder_pen_t_mean, id_t = self.encode_top(input)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b)
        quant_b = quant_b.permute(0, 3, 2, 1)
        quant_b, id_b, vq_pen_b, encoder_pen_b, entropy_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 2, 1)
        vq_pen_b = vq_pen_b.unsqueeze(0)

        # large weight for encoder_pen causes codebook collapse early in training
        latent_loss = vq_pen_t_mean + vq_pen_b.mean() + 0.01 * (encoder_pen_t_mean + encoder_pen_b.mean())
        return quant_t, quant_b, latent_loss, id_t, id_b

    def encode_bag(self, input, normalize: bool = True):
        _, _, _, _, id_b = self.encode(input)

        n_embed = self.quantize_b.n_classes  # .n_embed
        flat = id_b.view(input.size(0), -1)
        bag = torch.zeros(input.size(0), n_embed, dtype=id_b.dtype, device=id_b.device)
        bag.scatter_add_(dim=1, index=flat, src=torch.ones_like(flat))
        if normalize:
            bag = F.normalize(bag.float(), p=2.)
        return bag, id_b

    def encode_bag_t(self, input, normalize: bool = True):
        quant_t, _, _, _, id_t = self.encode_top(input)

        n_embed = self.quantize_t.n_classes  # .n_embed
        flat = id_t.view(input.size(0), -1)
        bag = torch.zeros(input.size(0), n_embed, dtype=id_t.dtype, device=id_t.device)
        bag.scatter_add_(dim=1, index=flat, src=torch.ones_like(flat))
        if normalize:
            #bag = F.normalize(bag.type(quant_t.dtype), p=2.)
            bag = F.normalize(bag.float(), p=2.)
        return bag, id_t

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec
