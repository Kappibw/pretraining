#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

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

        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q, k, v = [self.rearrange_qkv(t, self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.rearrange_back(out)
        out = self.to_out(out)

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
        image_size=(64, 256),
        patch_size=(16, 16),
        num_classes=2,
        dim=32,
        depth=2,
        heads=4,
        goal_size=3,
        mlp_dim=2048,
        channels=2,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        mean_pool=False
    ):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        ####### Add LayerNormalization ########
        self.layer_norm = nn.LayerNorm(dim)

        self.fc_embed = nn.Linear(goal_size, 32)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,128)

        self.pool = mean_pool

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            RearrangePatchEmbedding(patch_height, patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # Print out the model
        print("VIT Model \n\n")
        print(self)

    def forward(self, img, goal):
        goal = self.fc_embed(goal)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = torch.unsqueeze(goal, dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Get cls token or mean if pool is True
        x = x.mean(dim=1) if self.pool else x[:, 0]

        x = self.to_latent(x)
        x = self.layer_norm(x)
        # return self.mlp_head(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


##############################################
### Version with attention maps returned #####
##############################################

class AttentionWithMaps(Attention):
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [self.rearrange_qkv(t, self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self.rearrange_back(out)
        out = self.to_out(out)

        # Return both the output and attention maps
        return out, attn


class PreNormWithMaps(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        out, attn = self.fn(self.norm(x))
        return out, attn


class TransformerWithMaps(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attention_layers = nn.ModuleList(
            [PreNormWithMaps(dim, AttentionWithMaps(dim, heads=heads, dim_head=dim_head, dropout=dropout)) for _ in range(depth)]
        )
        self.feedforward_layers = nn.ModuleList(
            [PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)) for _ in range(depth)]
        )

    def forward(self, x):
        attentions = []
        for attn, ff in zip(self.attention_layers, self.feedforward_layers):
            attn_out, attn_map = attn(x)
            x = attn_out + x
            attentions.append(attn_map)
            x = ff(x) + x
        return x, attentions


class GoTWithAttentionMaps(GoT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer = TransformerWithMaps(
            dim=kwargs['dim'],
            depth=kwargs['depth'],
            heads=kwargs['heads'],
            dim_head=kwargs['dim_head'],
            mlp_dim=kwargs['mlp_dim'],
            dropout=kwargs['dropout']
        )

    def forward(self, img, goal):
        goal = self.fc_embed(goal)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = torch.unsqueeze(goal, dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x, attentions = self.transformer(x)

        # Get cls token
        x = x[:, 0]

        x = self.to_latent(x)
        x = self.layer_norm(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x, attentions
