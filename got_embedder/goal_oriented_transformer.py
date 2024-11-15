#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:36:28 2022

@author: oscar
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    # def forward(self, x, **kwargs):
    #     return self.fn(self.norm(x), **kwargs)

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def rearrange_qkv(self, t: torch.Tensor, heads: int):
        # Assume t has shape [b, n, h * d]
        b, n, hd = t.shape
        d = hd // heads  # Calculate the dimension per head
        return t.view(b, n, heads, d).permute(0, 2, 1, 3)  # Shape: [b, h, n, d]

    # Assuming `out` has shape [b, h, n, d]
    def rearrange_back(self, out):
        b, h, n, d = out.shape
        return out.permute(0, 2, 1, 3).reshape(b, n, h * d)  # Shape: [b, n, h * d]

    def forward(self, x):
        # return_attention = False # TODO: Remove this line
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = [self.rearrange_qkv(t, self.heads) for t in qkv]

        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self.rearrange_back(out)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # if return_attention:
        #     return out, attn
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attention_layers = nn.ModuleList(
            [PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)) for _ in range(depth)]
        )
        self.feedforward_layers = nn.ModuleList(
            [PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)) for _ in range(depth)]
        )

    def forward(self, x):
        for attn, ff in zip(self.attention_layers, self.feedforward_layers):
            x = attn(x) + x
            x = ff(x) + x
        return x

"""
Should perform the same function as:
Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width)

"""
class RearrangePatchEmbedding(nn.Module):
    def __init__(self, patch_height, patch_width):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, x):
        # Assume x has shape [b, c, h, w]
        b, c, h, w = x.shape
        p1, p2 = self.patch_height, self.patch_width
        # Reshape and permute to get shape [b, h * w, p1 * p2 * c]
        x = x.view(b, c, h // p1, p1, w // p2, p2)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // p1) * (w // p2), p1 * p2 * c)
        return x


class GoT(nn.Module):
    def __init__(
        self,
        *,
        image_size=(128, 128),
        patch_size=(16, 16),
        num_classes=2,
        dim=32,
        depth=2,
        heads=4,
        goal_size=3,
        mlp_dim=2048,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        ####### Add LayerNormalization ########
        self.layer_norm = nn.LayerNorm(dim)

        self.fc_embed = nn.Linear(goal_size, 32)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            RearrangePatchEmbedding(patch_height, patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img, goal):
        goal = self.fc_embed(goal)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = torch.unsqueeze(goal, dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        x = self.layer_norm(x)
        # return self.mlp_head(x)
        return x
