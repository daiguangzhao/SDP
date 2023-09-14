#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""

from multiprocessing.sharedctypes import Value
from typing_extensions import Self
from einops import rearrange, repeat
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch import einsum
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.hub import load_state_dict_from_url
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from . import performer_helper
from . import orthoformer_helper
from . import nystrom_helper

default_cfgs = {
    'vit_1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_1k_large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
}


def qkv_attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class JointSpaceTimeAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DividedAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # init to zeros
        self.qkv.weight.data.fill_(0)
        self.qkv.bias.data.fill_(0)
        self.proj.weight.data.fill_(1)
        self.proj.bias.data.fill_(0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        # num of heads variable
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Scale q
        q *= self.scale

        # Take out cls_q, cls_k, cls_v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), 
            (q_, k_, v_)
        )

        # expand CLS token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = qkv_attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,TS+1,(h_dim*n_head)*3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class BaseAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# class BaseAttention_MLP_Mixer(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
        
#         self.mixer_blocks=nn.ModuleList([])
#         token_dim = int(dim/2)
#         channel_dim = int(dim * 4)
#         # depth = int(self.num_part_traj_layers*3)
#         dropout = 0
#         num_patches = 196
#         # self.num_patches = int(self.temporal_resolution*self.num_parts)
#         # for _ in range(depth):
#         #     self.mixer_blocks.append(MixerBlock(dim, num_patches, token_dim, channel_dim, dropout))
#         self.MixerBlock = MixerBlock(dim, num_patches, token_dim, channel_dim, dropout)
 

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
        
#         out = self.MixerBlock(out)

#         return out

#     # def forward_mixer(self, out):
#     #     x = MixerBlock()


class FeedForward(nn.Module):
    
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            #由此可以看出 FeedForward 的输入和输出维度是一致的
            nn.Linear(dim,hidden_dim),
            #激活函数
            nn.GELU(),
            #防止过拟合
            nn.Dropout(dropout),
            #重复上述过程
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x

from einops.layers.torch import Rearrange, Reduce

class MixerBlock(nn.Module):
    def __init__(self,dim,num_patch,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch,token_dim,dropout),
            Rearrange('b d n -> b n d')
 
         )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
    def forward(self,x):
        x = x+self.token_mixer(x)
        x = x+self.channel_mixer(x)
        return x  

class BaseAttention_V1(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class TrajectoryAttention_DFWSGAR(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q @ k.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_Part(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        
        B, N, C = x.shape
        # print("x.size()", x.size())
        
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q @ k.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_Part_cross(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        
        B, N, C = x.shape
        # print("x.size()", x.size())
        
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q @ k.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_X(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:np], t[:, np:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V2(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # print("cls_q",cls_q.size())
        # print("part_q",part_q.size())
        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])
        
        # let Part token attend to key / values of all patches across time and space
        part_out = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part_out = rearrange(part_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, part_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V2_1(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # print("cls_q",cls_q.size())
        # print("part_q",part_q.size())
        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])
        
        # let Part token attend to key / values of all patches across time and space
        part_out = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part_out = rearrange(part_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, part_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V3(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # print("cls_q",cls_q.size())
        # print("part_q",part_q.size())
        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])
        
        # let Part token attend to key / values of all patches across time and space
        part_out = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part_out = rearrange(part_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, part_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V4(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])              
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])        

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        # x = self.proj(x)
        # x = self.proj_drop(x)
        return cls_out, x

class TrajectoryAttention_VIP_V5(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # print("cls_q",cls_q.size())
        # print("part_q",part_q.size())
        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])
        
        # let Part token attend to key / values of all patches across time and space
        part_out = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part_out = rearrange(part_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        # x = torch.cat((cls_out, part_out, x), dim=1)  # [8, 785, 768]

        # x = self.proj(x)
        # x = self.proj_drop(x)
        # # x:[8, 785, 768], att:[8, 12, 784, 4]
        return cls_out, part_out, x

class TrajectoryAttention_VIP_V6(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V7(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_0(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_0_1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_3(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V8_4(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_VIP_V10(nn.Module):
    def __init__(self, nparts, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nparts = nparts
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape

        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        np=self.nparts
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        # print("q的打印尺寸",q.size())  # [((b/num_gpus)*num_heads), N, C]torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, np, 768])
        
        # let Part token attend to key / values of all patches across time and space
        part_out = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part_out = rearrange(part_out, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            # print("q_dot_k",q_dot_k.size())
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            # print("q_dot_k",q_dot_k.size())
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        x = torch.cat((cls_out, part_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn

class TrajectoryAttention_PX_DEcoder(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv_bias=True
        self.q_part = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_x = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, part, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = part.shape
        # print("x.size()", x.size())
        
        P = seq_len
        F = num_frames  # 这个不是一开始配置文件的num_frames，而是temp_resoulation：4
        h = self.num_heads
        if x is None:            
            # project x to q, k, v vaalues
            q, k, v = self.qkv(part).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

            # Reshape: 'b n (h d) -> (b h) n d'
            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]
        else:
            q = self.q_part(part)
            k,v = self.kv_x(x).chunk(2, dim = -1)
            
            # Reshape: 'b n (h d) -> (b h) n d'
            q, = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), q)  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]
            k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]
        # print("q的打印尺寸",q.size())  torch.Size([96, 785, 64])
        # remove CLS token from q, k, v
        
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))  # https://www.runoob.com/python/python-func-map.html map(lambda t:)函数解释

        # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h) # torch.Size([8, 1, 768])
        # approx == none
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q @ k.transpose(-2, -1)  # torch.Size([96, 784, 784]) 为什么想把h放在b的维度在一起，因为想让每个head都能发挥不能的作用
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # [96, 784, 4*196]-->[8, 784, 4, 196]，这时把维度换回来了，为了方便计算
            space_attn = (self.scale * q_dot_k).softmax(dim=-1) # torch.Size([96, 784, 4, 196])
            attn = self.attn_drop(space_attn)
            v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)  # [96,4*196, 64]-->[96, 4, 196, 64]
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # torch.Size([96, 784, 4, 64])

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # [96, 784, 4, 64]-->[8, 784, 4, 768]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [8, 4, 196, 4, 768]
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # torch.Size([8, 196, 768, 4])  # 选完之后为啥在最后一维？
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # torch.Size([8, 784, 768])
        q2 = self.proj_q(x_diag)  # 映射出Q矩阵 # torch.Size([8, 784, 768])
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # torch.Size([8, 784, 4, 768])
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [8, 784, 12*64]-->[8, 12, 768, 64]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))  # [8, 784, 4, 768] --> [8, 12, 784, 4, 64]
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)  # [8, 12, 784, 64] * [8, 12, 784, 4, 64] --> [8, 12, 784, 4] 看不懂einsum求和函数
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)  # [8, 784, 4, 768]-->[8, 12, 196, 4, 64]
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x) # [8, 12, 784, 4] * [8, 12, 196, 4, 64] --> [8, 12, 784, 64]
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')  # torch.Size([8, 784, 768])

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)  # [8, 785, 768]

        x = self.proj(x)
        x = self.proj_drop(x)
        # x:[8, 785, 768], att:[8, 12, 784, 4]
        return x, attn


def get_attention_module(nparts,
    attn_type='joint', dim=768, num_heads=12, qkv_bias=False, 
    attn_drop=0., proj_drop=0., use_original_code=True
):
    if attn_type == 'joint':
        attn = JointSpaceTimeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop)
    elif attn_type == 'trajectory':
        attn = TrajectoryAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'trajectory_part':
        attn = TrajectoryAttention_Part(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'trajectory_part_cross_x':
        attn = TrajectoryAttention_Part_cross(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_x_trajectory':
        attn = TrajectoryAttention_VIP_X(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory':
        attn = TrajectoryAttention_VIP(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v2':
        attn = TrajectoryAttention_VIP_V2(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v2_1':
        attn = TrajectoryAttention_VIP_V2_1(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v3':
        attn = TrajectoryAttention_VIP_V3(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v4':
        attn = TrajectoryAttention_VIP_V4(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v5':
        attn = TrajectoryAttention_VIP_V5(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)                                                        
    elif attn_type == 'vip_trajectory_v6':
        attn = TrajectoryAttention_VIP_V6(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)    
    elif attn_type == 'vip_trajectory_v7':
        attn = TrajectoryAttention_VIP_V7(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)      
    elif attn_type == 'vip_trajectory_v8':
        attn = TrajectoryAttention_VIP_V8(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)  
    elif attn_type == 'vip_trajectory_v8_0':
        attn = TrajectoryAttention_VIP_V8(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code) 
    elif attn_type == 'vip_trajectory_v8_0_1':
        attn = TrajectoryAttention_VIP_V8(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)                           
    elif attn_type == 'vip_trajectory_v8_1':
        attn = TrajectoryAttention_VIP_V8_1(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)   
    elif attn_type == 'vip_trajectory_v8_2':
        attn = TrajectoryAttention_VIP_V8_2(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)                        
    elif attn_type == 'vip_trajectory_v8_3':
        attn = TrajectoryAttention_VIP_V8_3(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)    
    elif attn_type == 'vip_trajectory_v8_4':
        attn = TrajectoryAttention_VIP_V8_4(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    elif attn_type == 'vip_trajectory_v10':
        attn = TrajectoryAttention_VIP_V10(nparts,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)            
    return attn


# def get_attention_module(
#     attn_type='joint', dim=768, num_heads=12, qkv_bias=False, 
#     attn_drop=0., proj_drop=0., use_original_code=True
# ):
#     if attn_type == 'joint':
#         attn = JointSpaceTimeAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop)
#     elif attn_type == 'trajectory':
#         attn = TrajectoryAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'trajectory_part':
#         attn = TrajectoryAttention_Part(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'trajectory_part_cross_x':
#         attn = TrajectoryAttention_Part_cross(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_x_trajectory':
#         attn = TrajectoryAttention_VIP_X(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory':
#         attn = TrajectoryAttention_VIP(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v2':
#         attn = TrajectoryAttention_VIP_V2(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v2_1':
#         attn = TrajectoryAttention_VIP_V2_1(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v3':
#         attn = TrajectoryAttention_VIP_V3(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v4':
#         attn = TrajectoryAttention_VIP_V4(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v5':
#         attn = TrajectoryAttention_VIP_V5(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)                                                        
#     elif attn_type == 'vip_trajectory_v6':
#         attn = TrajectoryAttention_VIP_V6(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)    
#     elif attn_type == 'vip_trajectory_v7':
#         attn = TrajectoryAttention_VIP_V7(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)      
#     elif attn_type == 'vip_trajectory_v8':
#         attn = TrajectoryAttention_VIP_V8(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)  
#     elif attn_type == 'vip_trajectory_v8_0':
#         attn = TrajectoryAttention_VIP_V8(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code) 
#     elif attn_type == 'vip_trajectory_v8_0_1':
#         attn = TrajectoryAttention_VIP_V8(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)                           
#     elif attn_type == 'vip_trajectory_v8_1':
#         attn = TrajectoryAttention_VIP_V8_1(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)   
#     elif attn_type == 'vip_trajectory_v8_2':
#         attn = TrajectoryAttention_VIP_V8_2(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)                        
#     elif attn_type == 'vip_trajectory_v8_3':
#         attn = TrajectoryAttention_VIP_V8_3(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)    
#     elif attn_type == 'vip_trajectory_v8_4':
#         attn = TrajectoryAttention_VIP_V8_4(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)
#     elif attn_type == 'vip_trajectory_v10':
#         attn = TrajectoryAttention_VIP_V10(nparts,
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, 
#             attn_drop=attn_drop, proj_drop=proj_drop,
#             use_original_code=use_original_code)            
#     return attn


class Block(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=None,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Base_Block(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim_head = int(dim/num_heads)
        self.attn = BaseAttention(dim=dim,heads=num_heads,dim_head=self.dim_head,dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(
            self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Base_Block_MLP_Mixer(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim_head = int(dim/num_heads)
        self.attn = BaseAttention(dim=dim,heads=num_heads,dim_head=self.dim_head,dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mixerblocak = MixerBlock(dim=dim,num_patch=785,token_dim=384,channel_dim=mlp_hidden_dim,dropout=0)

    def forward(self, x):
        x = x + self.drop_path(
            self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.mixerblocak(x) # 如果加载这里，相当于12层，计算量太大！
        x = self.mixerblocak(x)

        return x

class Base_Block_V1(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim_head = int(dim/num_heads)
        self.attn = BaseAttention_V1(dim=dim,heads=num_heads,dim_head=self.dim_head,dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_ori = x
        x, attn = self.attn(self.norm1(x))
        
        x = x_ori + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

class Block_Enc_Dec(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, num_parts=8, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        
        self.norm_part = norm_layer(dim)
        
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.cross_att = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        # encoder
        self.reason = SimpleReasoning(num_parts, dim)
        self.dropout1 = nn.Dropout(0.1)
        # decoder
        self.cross_att = CrossAttention(dim, num_heads)

    def part_encoder(self, parts, x, seq_len_p=8, num_frames=8, approx='none', num_landmarks=128):
        parts = parts + self.drop_path(self.cross_att(q = parts, k = x, v = x, kpos = None , mask= None))

        # Norm+Liner        
        parts = self.reason(parts)
        
        # parts = self.norm_part(parts)       
        # parts = parts + self.dropout1(parts)
        
        # Norm + MLP
        parts = parts + self.drop_path(self.mlp(self.norm3(parts)))
        # -----------------------------------------------------------------------------
        # Parts通过TRAJ_ATT自己跟自己玩
        # -----------------------------------------------------------------------------               
        # Norm + Traj_att
        parts = parts + self.drop_path(
            self.attn(
                self.norm1(parts),
                seq_len=seq_len_p, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0],           
        )        
        parts = parts + self.drop_path(self.mlp(self.norm2(parts)))
        # print(parts.size())        
        return parts
    def part_x_decoder(self, parts, x, seq_len_x=196, num_frames=8, approx='none', num_landmarks=128):
        # 最后输出的维度和x的维度是一样的
        out = x + self.drop_path(self.cross_att(q = x, k =parts, v = parts, kpos = None , mask= None))
        # print("x_att",x.shape)
        
        out = out + self.drop_path(self.mlp(self.norm4(out)))
        # Traj_att
        out_traj = out + self.drop_path(
            self.attn(
                self.norm5(out), 
                seq_len=seq_len_x, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        # print("x_att",x.shape)
        out_final = out_traj + self.drop_path(self.mlp(self.norm6(out_traj)))
        # print("x_fea",x.shape)        
        
        return out_final

    def forward(self, parts, x, seq_len_p=8, seq_len_x=196, num_frames=8, approx='none', num_landmarks=128):
        # Encoder
        Parts = self.part_encoder(parts, x, seq_len_p, num_frames, approx, num_landmarks)
        # Decoder
        X = self.part_x_decoder(Parts, x, seq_len_x, num_frames, approx, num_landmarks)       

        return Parts, X        

class Block_Enc_Dec_VIP(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, num_parts=8, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        
        self.norm_part = norm_layer(dim)
        
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.attn_ = get_attention_module(
            attn_type='trajectory_part', dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.cross_att = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        # encoder
        self.reason = SimpleReasoning(num_parts, dim)
        self.dropout1 = nn.Dropout(0.1)
        # decoder
        self.cross_att = CrossAttention(dim, num_heads)

    def part_encoder(self, parts, x, seq_len_p=8, num_frames=8, approx='none', num_landmarks=128):
    # -----------------------------------------------------------------------------
    # Parts通过TRAJ_ATT自己跟自己玩
    # -----------------------------------------------------------------------------               
        # Norm + Traj_att
        parts = parts + self.drop_path(
            self.attn_(
                self.norm1(parts),
                seq_len=seq_len_p, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0],           
        )
        
        # parts = parts + self.drop_path(self.mlp(self.norm2(parts)))
    # -----------------------------------------------------------------------------
    # Parts和x利用Cross_att,通过将q:parts, 而k, v:x来更新之前学习的parts.
    # -----------------------------------------------------------------------------
        
        # parts = parts + self.drop_path(self.cross_att(q = parts, k = x, v = x, kpos = None , mask= None))

        # Norm+Liner        
        parts = self.reason(parts)
        
        # parts = self.norm_part(parts)       
        # parts = parts + self.dropout1(parts)
        
        # Norm + MLP
        parts_final = parts + self.drop_path(self.mlp(self.norm3(parts)))
        # print(parts.size())        
        return parts_final


    def part_x_decoder(self, parts, x, seq_len_x=196, num_frames=8, approx='none', num_landmarks=128):
        # 最后输出的维度和x的维度是一样的
        out = x + self.drop_path(self.cross_att(q = x, k =parts, v = parts, kpos = None , mask= None))
        # print("x_att",x.shape)
        
        out = out + self.drop_path(self.mlp(self.norm4(out)))
        # Traj_att
        out_traj = out + self.drop_path(
            self.attn(
                self.norm5(out), 
                seq_len=seq_len_x, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        # print("x_att",x.shape)
        out_final = out_traj + self.drop_path(self.mlp(self.norm6(out_traj)))
        # print("x_fea",x.shape)        
        
        return out_final

    def forward(self, parts, x, seq_len_p=8, seq_len_x=196, num_frames=8, approx='none', num_landmarks=128):
        # Encoder
        Parts = self.part_encoder(parts, x, seq_len_p, num_frames, approx, num_landmarks)
        # Decoder
        X = self.part_x_decoder(Parts, x, seq_len_x, num_frames, approx, num_landmarks)       

        return Parts, X        

class Block_VIP_V1(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        # print("x_att",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x_fea",x.shape)
        return x

class Block_VIP_V2(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        # print("x_att",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x_fea",x.shape)
        return x

class Block_VIP_V2_1(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        # print("x_att",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x_fea",x.shape)
        return x

class Block_VIP_V3(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        # print("x_att",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x_fea",x.shape)
        return x

# Part Encoder
class PartEncoder(nn.Module):
    def __init__(self, num_parts=32, dim=768, drop_path=0., num_heads=8, qkv_bias=False, attn_drop=0., use_original_code=True, drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU,mlp_ratio=4.):
        super().__init__()

        self.num_parts = num_parts
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.part_trajectoryattention = get_attention_module(nparts=num_parts,
            attn_type='trajectory_part', dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5
    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks = 128):

        # Parts和x利用Cross_att,通过将q:parts, 而k, v:x来学习parts.
        h = self.num_heads
        np = self.num_parts
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)将已经patch_embedding和position的token进行映射，得到的尺寸为[b,196*4(784+1),64*12(768)x3],然后chunk成3份

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))  # 对张量尺度进行重排，且必须要确定起初的维度：[8,784+1,768]-->[8, 784+1, 12*64]-->[8*12, 784+1, 768]

        (cls_q, part_q, q_), (cls_k, part_k, k_), (cls_v, part_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:np+1], t[:, np+1:]), (q, k, v))  
             
        # let Part token attend to key / values of all patches across time and space
        part_att = qkv_attn(part_q * self.scale, k, v)  # torch.Size([96, 1, 64])
        part = rearrange(part_att, f'(b h) k d -> b k (h d)', k=np, h=h) # torch.Size([8, np, 768])
        
        # part = part + self.drop_path(
        #     self.part_trajectoryattention(
        #         self.norm1(part), 
        #         seq_len=seq_len, 
        #         num_frames=num_frames, 
        #         approx=approx,
        #         num_landmarks=num_landmarks
        #     )[0]
        # )
        part = part + self.drop_path(self.mlp(self.norm2(part)))
        
        part = part + self.drop_path(qkv_attn(part, part, part))
        
        part = part + self.drop_path(self.mlp(self.norm3(part)))
        
        return part

class Block_VIP_V4(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_parts = num_parts
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.encoder = PartEncoder(num_parts=num_parts, dim=dim, drop_path=drop_path, 
                                    num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop,
                                    use_original_code=use_original_code,drop=drop,act_layer=act_layer,mlp_ratio=mlp_ratio)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        # Encoder
        parts = self.encoder(x, seq_len=int(self.num_parts/num_frames), num_frames=num_frames, approx=approx, num_landmarks=num_landmarks)
        
        cls_out, att_x = self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )
      
        x_ = torch.cat((cls_out, parts, att_x),dim=1)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)       
        x = x + self.drop_path(x_)          
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

class Block_VIP_V5(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_parts = num_parts
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # self.encoder = PartEncoder(num_parts=num_parts, dim=dim, drop_path=drop_path, 
        #                             num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,
        #                             use_original_code=use_original_code,drop=drop)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768/12=64
        self.scale = self.head_dim ** -0.5
   
    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        # parts = self.encoder(x, seq_len=int(self.num_parts/num_frames), num_frames=num_frames, approx=approx, num_landmark=num_landmarks)
        # Encoder
        cls_out, parts, att_x = self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )
      
        x_ = torch.cat((cls_out, parts, att_x),dim=1)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)       
        x = x + self.drop_path(x_)          
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Decoder
        x = x + self.drop_path(qkv_attn(x * self.scale, parts, parts))       
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        parts = parts + self.drop_path(qkv_attn(parts * self.scale, parts, parts))       
        parts = parts + self.drop_path(self.mlp(self.norm4(parts)))

        return x 

class Block_VIP_V6(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=0, 
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.part_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, x, part_embed, seq_len=196, num_frames=8, approx='none', num_landmarks=128):       
        B = x.shape[0]
        D = self.dim      
        q = k = self.with_pos_embed(part, part_embed)
        part_out = qkv_attn(q, k, v = part)
        # part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_0 = x + self.drop_path(
            self.attn(
                self.norm2(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        x_0 = x
        x_1 = self.norm3(x_0)[:, 1:]
        
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t = num_frames, d=D)

        # part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
        #                                 key=x_1, value=x_1, attn_mask=None,
        #                                 key_padding_mask=None)
        part_ = qkv_attn(q = self.with_pos_embed(part, part_embed), k = x_1, v = x_1)
        part = part + self.dropout2(part_)
        
        part = self.norm4(part)
        
        part = part + self.drop_path(self.mlp(part))

        part = self.norm5(part)
        
        # print("part",part.size())
        # print("part.size:",part.size()) # [B, K, D]
        # x_2 = rearrange(x_1,'(b t) s d -> b (t s) d', b = B, t = num_frames, d=D)
        
        x = x_0 + self.drop_path(self.mlp(self.norm6(x_0)))
        
        return part, x

class Block_VIP_V7(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()        
        self.attn = get_attention_module(nparts=0, 
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.part_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, x, part_embed, seq_len=196, num_layer=1, num_frames=8, approx='none', num_landmarks=128):       
        B = x.shape[0]
        D = self.dim 
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        if num_layer <= 11:
            x = x + self.drop_path(self.mlp(self.norm6(x)))
            return part, x 
        else:           
            q = k = self.with_pos_embed(part, part_embed)
            # part_out = qkv_attn(q, k, v = part)
            part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
            
            part = part + self.dropout1(part_out) #残差连接
            
            part = self.norm1(part)
            
            x_0 = x + self.drop_path(
                self.attn(
                    self.norm2(x), 
                    seq_len=seq_len, 
                    num_frames=num_frames, 
                    approx=approx,
                    num_landmarks=num_landmarks
                )[0]
            )
            x_0 = x + x_0
            x_1 = self.norm3(x_0)[:, 1:]
            
            x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t = num_frames, d=D)

            part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                            key=x_1, value=x_1, attn_mask=None,
                                            key_padding_mask=None)
            # part_ = qkv_attn(q = self.with_pos_embed(part, part_embed), k = x_1, v = x_1)
            part = part + self.dropout2(part_)
            
            part = self.norm4(part)
            
            part = part + self.drop_path(self.mlp(part))

            part = self.norm5(part)
            
            # print("part",part.size())
            # print("part.size:",part.size()) # [B, K, D]
            # x_2 = rearrange(x_1,'(b t) s d -> b (t s) d', b = B, t = num_frames, d=D)
            
            x = x_0 + self.drop_path(self.mlp(self.norm6(x_0)))
            
            return part, x 

class Block_VIP_V8(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_0(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_0_1(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_1(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_2(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_3(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V8_4(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Block_VIP_V10(nn.Module):

    def __init__(
            self, num_parts=32, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(nparts=num_parts,
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        # print("x_att",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x_fea",x.shape)
        return x

class PROT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        
        """
        (1) 在最后一层的时候，(1.0 - attn_policy)表示是背景信息, 因为前景信息为1, 背景信息全部被点亮
        (2) (1.0 - attn_policy) * eye表示作者对于一个token只想保留一个背景信息
        (3) attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)表示将仅仅将前景(非常少量的背景信息)信息的分数保留
        (4) 通过前11层迭代计算rollout_attention, 将背景信息不断的过滤, 得到就仅仅只有前景信息的位置掩膜矩阵(前景为1背景为0)
        (5) 将上述的掩膜矩阵和第12层的attention矩阵相乘,输出仅有前景score的attention matrices       
        """
        B, N, _ = policy.size()
        B, H, N, N = attn.size()  # [B, Num_Heads, N, N]
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        # print(attn_policy.size()) # torch.Size([64, 1, 1, 197]): (B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N) # 创建对角矩阵，并且扩张两个维度

        attn_policy = attn_policy + (1.0 - attn_policy) * eye  # 前10次（1.0 - attn_policy）为全零矩阵，此时attn_policy = attn_policy
        max_att = torch.max(attn, dim=-1, keepdim=True)[0] # torch.Size([64, 12, 197, 1])
 
        attn = attn - max_att # 减去最大值，防止大的节点求e函数时溢出
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32) # 公式（4）
        # print("attn",attn.size()) # [B, H, N, N]
        # print("attn:", attn)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)  # torch.Size([64, 12, 197, 197])

        return attn.type_as(max_att)  # type_as --按照给定的tensor的类型转换类型

    def forward(self, x, policy):
        """
        when i is 12,
        x is final output represented the foreground information
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)，类似于chunk操作. q:[B, H, N, D]:[B, num_heads, N, head_dim] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, N, 384]
        x = self.proj(x)
        x = self.proj_drop(x) 
        return x, attn

class Block_VIP_PROTO(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PROT_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_proto(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x_ori = x
        x, attn = self.attn(self.norm1(x), policy)
        
        # 经典的FFN结构
        x = x_ori + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, attn


class Mlp_proto(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SimpleReasoning(nn.Module):
    def __init__(self, np, dim):
        super(SimpleReasoning, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Conv1d(np, np, kernel_size=1, bias=False)

    def forward(self, x):
        tokens = self.norm(x)
        tokens = self.linear(tokens)
        return x + tokens

Norm = nn.LayerNorm

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super(CrossAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v, qpos, kpos):
        # q = apply_pos(q, qpos, self.num_heads)
        # k = apply_pos(k, kpos, self.num_heads)
        # v = apply_pos(v, None, 0)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None, qpos=None, kpos=None, mask=None, rel_pos=None):
        q, k, v = self.get_qkv(q, k, v, qpos, kpos)

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # attn matrix calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        if rel_pos is not None:
            attn = rel_pos(q, attn)
        attn *= self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        return out

class DividedSpaceTimeBlock(nn.Module):

    def __init__(
        self, dim=768, num_heads=12, attn_type='divided', 
        mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

        self.norm1 = norm_layer(dim)
        
        self.attn = DividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)

        self.timeattn = DividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        time_output = self.timeattn(self.norm3(x), 
            self.einops_from_time, self.einops_to_time, n=seq_len)
        time_residual = x + time_output

        space_output = self.attn(self.norm1(time_residual), 
            self.einops_from_space, self.einops_to_space, f=num_frames)
        space_residual = time_residual + self.drop_path(space_output)

        x = space_residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_P(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_p = nn.Linear(in_features, hidden_features)
        self.act_p = act_layer()
        self.fc2_p = nn.Linear(hidden_features, out_features)
        self.drop_p = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1_p(x)
        x = self.act_p(x)
        x = self.drop_p(x)
        x = self.fc2_p(x)
        x = self.drop_p(x)
        return x

class Mlp_P_(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_p_ = nn.Linear(in_features, hidden_features)
        self.act_p_ = act_layer()
        self.fc2_p_ = nn.Linear(hidden_features, out_features)
        self.drop_p_ = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1_p_(x)
        x = self.act_p_(x)
        x = self.drop_p_(x)
        x = self.fc2_p_(x)
        x = self.drop_p_(x)
        return x

class Mlp_P_1(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_p_1 = nn.Linear(in_features, hidden_features)
        self.act_p_1 = act_layer()
        self.fc2_p_1 = nn.Linear(hidden_features, out_features)
        self.drop_p_1 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1_p_1(x)
        x = self.act_p_1(x)
        x = self.drop_p_1(x)
        x = self.fc2_p_1(x)
        x = self.drop_p_1(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size if type(img_size) is tuple else to_2tuple(img_size)
        patch_size = img_size if type(patch_size) is tuple else to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(
            self, img_size=224, temporal_resolution=4, in_chans=3, 
            patch_size=16, z_block_size=2, embed_dim=768, flatten=True
        ):
        super().__init__()
        self.height = (img_size // patch_size)
        self.width = (img_size // patch_size)
        self.frames = (temporal_resolution // z_block_size)
        self.num_patches = self.height * self.width * self.frames
        self.proj = nn.Conv3d(in_chans, embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size), 
            stride=(z_block_size, patch_size, patch_size))
        self.flatten = flatten

    def forward(self, x):
        #  输入尺寸[8, 3, 8, 224, 224]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        # print("x2.shape",x.shape)  x2.shape torch.Size([8, 784, 768])
        return x

class HeadMLP(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(HeadMLP, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=True),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )
        print(f"Dropout-NLP: {p}")

    def forward(self, x):
        return self.block_forward(x)

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def adapt_input_conv(in_chans, conv_weight, agg='sum'):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            if agg == 'sum':
                print("Summing conv1 weights")
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
            else:
                print("Averaging conv1 weights")
                conv_weight = conv_weight.mean(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            if agg == 'sum':
                print("Summing conv1 weights")
                repeat = int(math.ceil(in_chans / 3))
                conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(in_chans))
            else:
                print("Averaging conv1 weights")
                conv_weight = conv_weight.mean(dim=1, keepdim=True)
                conv_weight = conv_weight.repeat(1, in_chans, 1, 1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight

def load_pretrained(
    model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, progress=False
):
    # Load state dict
    assert(f"{cfg.VIT.PRETRAINED_WEIGHTS} not in [vit_1k, vit_1k_large]")
    state_dict = torch.hub.load_state_dict_from_url(url=default_cfgs[cfg.VIT.PRETRAINED_WEIGHTS])
    
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    input_convs = 'patch_embed.proj'
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(
                    in_chans, state_dict[weight_name], agg='avg')
                print(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                print(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = 'head'
    label_offset = cfg.get('label_offset', 0)
    pretrain_classes = 1000
    if num_classes != pretrain_classes:
        # completely discard fully connected if model num_classes doesn't match pretrained weights
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    elif label_offset > 0:
        # special case for pretrained weights with an extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    loaded_state = state_dict
    self_state = model.state_dict()
    all_names = set(self_state.keys())
    saved_names = set([])
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys() and param.shape == self_state[name].shape:
            saved_names.add(name)
            self_state[name].copy_(param)
        else:
            print(f"didnt load: {name} of shape: {param.shape}")
    print("Missing Keys:")
    print(all_names - saved_names)

class PartDecoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0.1, temporal_resolution=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.temporal_resolution = temporal_resolution
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)
        
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
        
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
 
        q = k = self.with_pos_embed(part, part_embed)
        
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_1 = self.norm2(x)[:, 1:]
        
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        
        part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                        key=x_1, value=x_1, attn_mask=None,
                                        key_padding_mask=None)

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)

        return part

class PartDecoderLayer2(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0.025):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)       
        self.norm4 = nn.LayerNorm(self.embed_dim)
        self.norm5 = nn.LayerNorm(self.embed_dim)
        self.norm6 = nn.LayerNorm(self.embed_dim)
        self.norm7 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)
        self.dropout4 = nn.Dropout(self.drop_rate)
        self.dropout5 = nn.Dropout(self.drop_rate)

        
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

        self.mlp_p_ = Mlp_P_(
        in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

        self.trajectoryatt = TrajectoryAttention_Part(
            dim=embed_dim, num_heads=nhead, qkv_bias=True, 
            attn_drop= 0, proj_drop=0,
            use_original_code=True)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x, nframes, nparts):
        
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = nframes
 
        q = k = self.with_pos_embed(part, part_embed)  # [B, T*K, D]
        
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_1 = self.norm2(x)[:, 1:] # [B, T*S, D]
        # x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        
        part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                        key=x_1, value=x_1, attn_mask=None,
                                        key_padding_mask=None)

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)
        
        part = self.with_pos_embed(part, part_embed)
        
        part = part + self.dropout4(
            self.trajectoryatt(
                self.norm5(part), 
                seq_len=nparts, 
                num_frames=nframes, 
                approx=None,
                num_landmarks=128
                )[0])

        part = part + self.dropout5(self.mlp_p_(self.norm6(part)))

        part = self.norm7(part)

        return part

class PartDecoderLayer3(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)

       
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
 
        q = k = self.with_pos_embed(part, part_embed)  # [B, T*K, D]
        
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)

        
        part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                        key=x, value=x, attn_mask=None,
                                        key_padding_mask=None)

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)

        return part

class PartDecoderLayer4(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0, temporal_resolution=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.temporal_resolution = temporal_resolution
        
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)

       
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
 
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
 
        q = k = self.with_pos_embed(part, part_embed)
        
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_1 = self.norm2(x)[:, 1:]
        
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        
        part_ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                        key=x_1, value=x_1, attn_mask=None,
                                        key_padding_mask=None)[0]

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)

        return part

class P2P_Attention(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, temporal_resolution=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = 0.1
        self.mlp_ratio = mlp_ratio
        self.part_selfattn = MulitHeadAttention(dim=self.embed_dim,num_heads=self.num_heads)
        self.p2p_attn = MulitHeadAttention(dim=self.embed_dim,num_heads=self.num_heads)
        # Implementation of Feedforward model
        self.temporal_resolution = temporal_resolution
        
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)

       
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
 
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
 
        q = k = self.with_pos_embed(part, part_embed)
        
        part_out = self.part_selfattn(q=q, k=k, v=part)
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_1 = self.norm2(x)[:, 1:]
        
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        
        part_ = self.part_selfattn(q=self.with_pos_embed(part, part_embed), k=x_1, v=x_1)

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)

        return part





class PartDecoderLayer4_Pyramid(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0,temporal_resolution=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.temporal_resolution = temporal_resolution
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.dropout3 = nn.Dropout(self.drop_rate)

       
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)
        
        # 上采样特征图
        z_block_size = self.temporal_resolution // 2
        self.up_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.ConvTranspose3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim*2),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))       
        )
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
 
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
 
        q = k = self.with_pos_embed(part, part_embed)
        
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        
        part = part + self.dropout1(part_out) #残差连接
        
        part = self.norm1(part)
        
        x_1 = self.norm2(x)[:, 1:]
        
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        # pyramid
        x_1 = x_1.permute(1, 0, 2)
        L, NT, C = x_1.shape
        N = NT // T
        H = W = int(L ** 0.5)
        x_1 = x_1.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 14, 14]
        x_1 = self.up_conv(x_1) # [12, 1536, 8, 14, 14]
        x_1 = x_1.flatten(2).transpose(1,2) # [12, 196*8, 1536]
        x_1 = rearrange(x_1,'b (t s) d -> (b t) s d', b = B, t = int(T*2), d=D) 
        part_, _ = self.multihead_attn(query=self.with_pos_embed(part, part_embed),
                                        key=x_1, value=x_1, attn_mask=None,
                                        key_padding_mask=None)

        part = part + self.dropout2(part_)
        
        part = self.norm3(part)
        
        part = part + self.dropout3(self.mlp_p(part))

        part = self.norm4(part)


        return part


class PartDecoderLayer4_divided_att(nn.Module):

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0,temporal_resolution=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.part_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        # Implementation of Feedforward model
        self.temporal_resolution = temporal_resolution
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        self.norm_s = nn.LayerNorm(self.embed_dim)
        self.norm_t = nn.LayerNorm(self.embed_dim)
        self.norm_s_1 = nn.LayerNorm(self.embed_dim)
        self.norm_s_2 = nn.LayerNorm(self.embed_dim)
        
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout_s = nn.Dropout(self.drop_rate)
        self.dropout_s_1 = nn.Dropout(self.drop_rate)

        self.temporal_fc = nn.Linear(embed_dim, embed_dim)
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part, part_embed, x):
 
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = part.size()[1]

        # spatial
        q = k = self.with_pos_embed(part, part_embed)
        part_out = self.part_attn(q, k, value = part, attn_mask=None, key_padding_mask = None)[0]
        part = part + self.dropout1(part_out) #残差连接
        
        # Times
        init_cls_token = self.norm1(x)[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(K, T, 1)
        part_t = rearrange(part,'(b t) k d -> (b k) t d', b = B, t =T, d=D)   
        res_temporal = self.multihead_attn(query=self.norm_t(part_t),
                                        key=cls_token, value = cls_token, attn_mask=None,
                                        key_padding_mask=None)[0]

        res_temporal = self.temporal_fc(res_temporal)
        part_t = res_temporal + part_t # (b k) t d
                                        
        # Spatial   
        x_s = self.norm2(x)[:, 1:]
        x_s = rearrange(x_s,'b (t s) d -> (b t) s d', b = B, t =T, d=D)
        part_s = part_t
        part_s = rearrange(part_s,'(b k) t d -> (b t) k d', b = B, t =T, d=D)
        res_spatial = self.multihead_attn(query=self.norm_s(part_s),
                                        key=x_s, 
                                        value=x_s, 
                                        attn_mask=None,
                                        key_padding_mask=None)[0] # (b t) k d 
        
        # resudal + mlp
        part_s = part_s + self.dropout_s(res_spatial)  
        part_s = part_s + self.dropout_s_1(self.mlp_p(self.norm_s_1(part_s)))

        part = self.norm_s_2(part)

        return part
        
# 用这个代替nn.multi-head attention
class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 取出这里的注意力图可视化
        attentionmap = self.attn_drop(attn)
        print("attentionmap",attentionmap.size())
        
        x = (attentionmap @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class part_slow_fast_cross_att(nn.Module):
    """
    仅仅只有时间维度的CROSS-ATT
    """

    def __init__(self, embed_dim, nhead, mlp_ratio=4, dropout=0):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = dropout
        self.mlp_ratio = mlp_ratio
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part_ud, part1, part2):
        
        part_ud = self.multihead_attn(query=part_ud, 
                                        key=part1, value=part2, 
                                        attn_mask=None, key_padding_mask=None)[0]

        part_ud = part_ud + self.dropout1(part_ud)
        
        part_ud = self.norm1(part_ud)
        
        part_ud = part_ud + self.dropout2(self.mlp_p(part_ud))

        part_ud = self.norm2(part_ud)

        return part_ud



class SFC_Attention(nn.Module):
    """
    仅仅只有时间维度的CROSS-ATT
    """

    def __init__(self, embed_dim, nhead, mlp_ratio=4):
        super().__init__()
        self.num_heads = nhead
        self.embed_dim= embed_dim
        self.drop_rate = 0.1
        self.mlp_ratio = mlp_ratio
        self.SFA = MulitHeadAttention(dim=self.embed_dim, num_heads=self.num_heads)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp_p = Mlp_P(
            in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, part_ud, part1, part2):
        
        part_ud = self.SFA(q=part_ud, k=part1, v=part2)

        part_ud = part_ud + self.dropout1(part_ud)
        
        part_ud = self.norm1(part_ud)
        
        part_ud = part_ud + self.dropout2(self.mlp_p(part_ud))

        part_ud = self.norm2(part_ud)

        return part_ud


