import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor

        see https://github.com/pfriesch/VQVAE#deviations-from-the-papers
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=True):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.vec_len = vec_len
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * torch.sqrt(torch.tensor([x0.size(3)], dtype=torch.float, device=self.embedding0.device))
            x = target_norm * x0 / x0.norm(dim=3, p=2, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, p=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        # batch, mel, time, emb
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3, p=2).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor

        # compute the entropy
        hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
        prob = hist.masked_select(hist > 0) / len(index)
        entropy = - (prob * prob.log()).sum().detach()

        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        discrete = (output - x).detach() + x
        vq_pen = (x.detach() - output).float().norm(dim=3, p=2).pow(2)
        encoder_pen = (x - output.detach()).float().norm(dim=3, p=2).pow(2) + (x - x0).float().norm(dim=3, p=2).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (discrete, index1.view(x.size(0), x.size(1), x.size(2)), vq_pen, encoder_pen, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * torch.sqrt(torch.Tensor([self.embedding0.size(2)]))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
