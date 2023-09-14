# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from https://github.com/m-bain/frozen-in-time/blob/main/model/video_transformer.py
# Modified by Yue Zhao
# The original code is under MIT License

"""
Implementations of Video Transformers in PyTorch
A PyTorch implementation of space-time transformer as described in
'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval' - https://arxiv.org/abs/2104.00650
A PyTorch implementation of timesformer as described in
'Is Space-Time Attention All You Need for Video Understanding?' - https://arxiv.org/abs/2102.05095
Acknowledgments:
- This code builds on Ross Wightman's vision_transformer code in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
- It is also inspired by lucidrains timesformer implementation:
https://github.com/lucidrains/TimeSformer-pytorch
Hacked together by Max Bain
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn
from . import vit_helper
import copy
import slowfast.models.uniformerv2_model as uniformerv2_model
import numpy as np
from typing import Tuple
import math
from operator import mul
from functools import reduce

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, ln_pre=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        # ln_pre is inserted to be compatible with CLIP-style model
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F <= self.num_frames
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        return x

# 这是Lvaila原始的
# class VarAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  initialize='random'):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#         if initialize == 'zeros':
#             self.qkv.weight.data.fill_(0)
#             self.qkv.bias.data.fill_(0)
#             # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
#             # are multiplied by 0*0, which is hard for the model to move out of.
#             self.proj.weight.data.fill_(1)
#             self.proj.bias.data.fill_(0)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, einops_from, einops_to, einops_dims):
#         h = self.num_heads
#         # project x to q, k, v vaalues
#         q, k, v = self.qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         q *= self.scale

#         # splice out CLS token at index 1
#         (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

#         # let CLS token attend to key / values of all patches across time and space
#         cls_out = attn(cls_q, k, v)

#         # rearrange across time or space
#         q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_)) # 时间是变成了[b*n,f,d]，空间则变成了[b*f,s,d]

#         # expand cls token keys and values across time or space and concat
#         r = q_.shape[0] // cls_k.shape[0] # 时间att是f, 空间att是n
#         cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v)) # 时间[b*f, 1, d]；空间[b*s, 1, d] 

#         k_ = torch.cat((cls_k, k_), dim=1)
#         v_ = torch.cat((cls_v, v_), dim=1)

#         # attention
#         out = attn(q_, k_, v_)

#         # merge back time or space
#         out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

#         # concat back the cls token
#         out = torch.cat((cls_out, out), dim=1)

#         # merge back the heads
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         # to out
#         x = self.proj(out)
#         x = self.proj_drop(x)
#         return x


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, einops_from, einops_to, einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_)) # 时间是变成了[b*n,f,d]，空间则变成了[b*f,s,d]

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0] # 时间att是f, 空间att是n
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v)) # 时间[b*f, 1, d]；空间[b*s, 1, d] 

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


# # lavila原始的 + space_prompot
# class SpaceTimeBlock(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
#                  attention_style='frozen-in-time', is_tanh_gating=False,
#                  use_space_prompt=False,
#                  num_frames=16):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = VarAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.timeattn = VarAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
#             initialize=time_init)

#         if is_tanh_gating:
#             self.alpha_timeattn = nn.Parameter(torch.zeros([]))

#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.norm3 = norm_layer(dim)

#         self.attention_style = attention_style
#         self.use_space_prompt = use_space_prompt
#         if self.use_space_prompt:
#             self.Space_Prompt_Attention = Space_Prompt_Attention(temporal_resolution=num_frames,num_part_layers=1,num_parts=9,embed_dim=self.embed_dim,nheads=num_heads,mlp_ratio=mlp_ratio)

#     def forward(self, x, 
#                 space_prompt,
#                 space_prompt_position,
#                 einops_from_space, einops_to_space, 
#                 einops_from_time, einops_to_time,
#                 time_n, space_f, use_checkpoint=False):
#         if use_checkpoint:
#             time_output = checkpoint.checkpoint(
#                 self.timeattn, self.norm3(x), einops_from_time, einops_to_time, {"n": time_n}
#             )
#         else:
#             time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, {"n": time_n})
        
#         if hasattr(self, "alpha_timeattn"):
#             time_output = torch.tanh(self.alpha_timeattn) * time_output
#         time_residual = x + time_output
#         if use_checkpoint:
#             space_output = checkpoint.checkpoint(
#                 self.attn, self.norm1(time_residual), einops_from_space, einops_to_space, {"f": space_f}
#             )
#         else:
#             space_output = self.attn(self.norm1(time_residual),
#                                      einops_from_space,
#                                      einops_to_space, {"f": space_f})
#         if self.attention_style == 'frozen-in-time': # 默认是的，执行
#             space_residual = x + self.drop_path(space_output)
#         else:
#             raise NotImplementedError
#         x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))
    
#         if self.use_space_prompt:
#             space_prompt = self.Space_Prompt_Attention(space_prompt,space_prompt_position,x) # [b*t, 1, d]
#             return x, space_prompt
#         else:
#             return x

class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', is_tanh_gating=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.timeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            initialize=time_init)

        if is_tanh_gating:
            self.alpha_timeattn = nn.Parameter(torch.zeros([]))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style

    def forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
                time_n, space_f, use_checkpoint=False):
        if use_checkpoint:
            time_output = checkpoint.checkpoint(
                self.timeattn, self.norm3(x), einops_from_time, einops_to_time, {"n": time_n}
            )
        else:
            time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, {"n": time_n})
        if hasattr(self, "alpha_timeattn"):
            time_output = torch.tanh(self.alpha_timeattn) * time_output
        time_residual = x + time_output
        if use_checkpoint:
            space_output = checkpoint.checkpoint(
                self.attn, self.norm1(time_residual), einops_from_space, einops_to_space, {"f": space_f}
            )
        else:
            space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                     einops_to_space, {"f": space_f})
        if self.attention_style == 'frozen-in-time':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        return x


# 这个是两层FC加一个激活函数的mlp block
# 因为有两个mixing,进出的维度都不变，只是中间全连接层的神经元数量不同
# 定义多层感知机
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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 模块1
class Part_Token_Attention_V13_Cross(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, mlp_ratio=4.):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        # mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        # self.mlp_p_ = vit_helper.Mlp_P_(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        
        self.PartLayers = _get_clones(partlayer, self.num_part_layers)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    # def forward_features(self, x):
    def forward(self, x):
        
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # 维度是 [B*T, K, D]，这里的K代表了每一帧设置K个可学习的tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
        
        part = rearrange(part,'(b t) k d -> b (t k) d', b = B, t =T, d=D)
   
        part = torch.mean(part, dim=1)

        return part



# 模块1 + 2 + 3
class Part_Token_Attention_V13_Cross_Uniformerv2_Local_MLP_MIXER_SlowFast_T(nn.Module):
    """ 
    这里的仅仅改变时间分辨率, 特征通道保持不变！
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
                mlp_ratio=4., mlp_miexer_depth=1, uniformer_layers=2, fusion='divided_cat_1'):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        
        self.PartLayers = _get_clones(partlayer, self.num_part_layers)
       
        # mlp-mixer
        self.mixer_blocks=nn.ModuleList([])
        token_dim = int(self.embed_dim/2)
        channel_dim = int(self.embed_dim * 4)
        depth = mlp_miexer_depth
        dropout = 0
        self.num_patches = int(self.temporal_resolution*self.num_parts) + self.temporal_resolution
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(self.embed_dim, self.num_patches, token_dim, channel_dim, dropout))
        
        self.mixer_blocks_down=nn.ModuleList([])
        token_dim_down = int(self.embed_dim/2)
        channel_dim_down = int(self.embed_dim * 4)
        self.num_patches_down = int(int(self.temporal_resolution /2) *self.num_parts) + int(self.temporal_resolution/2)        
        for _ in range(depth):
            self.mixer_blocks_down.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_down, 
                                                    token_dim=token_dim_down, channel_dim=channel_dim_down, dropout=dropout))
        
        self.mixer_blocks_up=nn.ModuleList([])
        token_dim_up = int(self.embed_dim/2)
        channel_dim_up = int(self.embed_dim * 4)
        self.num_patches_up = int(int(self.temporal_resolution * 2)*self.num_parts) + int(self.temporal_resolution*2)         
        for _ in range(depth):
            self.mixer_blocks_up.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_up, 
                                                    token_dim=token_dim_up, channel_dim=channel_dim_up, dropout=dropout))
        
        #------uniformerv2------
        layers = uniformer_layers
        self.resblocks = nn.ModuleList([
            uniformerv2_model.ResidualAttentionBlock(
                d_model=self.embed_dim, n_head=self.nheads, attn_mask=None, 
                drop_path=0.01,
                dw_reduction=1.5,
                no_lmhra=False,
                double_lmhra=True,
            ) for i in range(layers)
        ])
        # down
        self.resblocks_down = nn.ModuleList([
            uniformerv2_model.ResidualAttentionBlock(
                d_model=int(self.embed_dim), n_head=self.nheads, attn_mask=None, 
                drop_path=0.01,
                dw_reduction=1.5,
                no_lmhra=False,
                double_lmhra=True,
            ) for i in range(layers)
        ]) 
        # up
        self.resblocks_up = nn.ModuleList([
            uniformerv2_model.ResidualAttentionBlock(
                d_model=int(self.embed_dim), n_head=self.nheads, attn_mask=None, 
                drop_path=0.01,
                dw_reduction=1.5,
                no_lmhra=False,
                double_lmhra=True,
            ) for i in range(layers)
        ])                  
        # slowfast
        # 下采样特征图
        z_block_size = 2       
        self.down_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.Conv3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))
        )
        # 上采样特征图
        self.up_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.ConvTranspose3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))       
        )
          
        self.fusion = fusion
           
        # Initialize weights
        self.apply(self._init_weights)

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

    # def forward_features(self, x):
    def forward(self, x):
        
        B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = self.num_parts
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
        
        # papre up-down conv
        part_up_down = part.clone() 
        part_up_down = part_up_down.permute(1, 0, 2) # [B*T, K, D] --> [K, B*T, D]
        L, NT, C = part_up_down.shape
        N = NT // T
        H = W = int(L ** 0.5)
        part_up_down = part_up_down.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 3, 3]
        
        # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
        part_down = part_up_down.clone()
        part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
        part_down = part_down.flatten(2).transpose(1,2) # [13, 2*9, 768]
        part_down = rearrange(part_down,'b (t k) d -> (b t) k d',  b= B, t=int(T/2), d=int(D))

        # up_conv
        part_up = part_up_down.clone()
        part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
        part_up = part_up.flatten(2).transpose(1,2) # [13, 8*9, 1536]
        part_up = rearrange(part_up,'b (t k) d -> (b t) k d',  b= B, t=int(T*2), d=int(D))

        # take care of cls token
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
        # maping cls_token_dowm/up, up or down cls_token channel dim
        init_cls_token_down = x[:, 0, :].unsqueeze(1)
        cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
        cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
        init_cls_token_up = x[:, 0, :].unsqueeze(1)
        cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
        cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

        # cat cls_token and part_token 
        part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
        part = part.permute(1, 0, 2) # [k+1, b*t, d]

        part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
        part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

        part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
        part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
        
        # uniformerv2
        for i, resblock in enumerate(self.resblocks):
            part = resblock(part, T)
        for i, resblock in enumerate(self.resblocks_down):
            part_down = resblock(part_down, int(T/2))
        for i, resblock in enumerate(self.resblocks_up):
            part_up = resblock(part_up, int(T*2))                         
        
        part = rearrange(part,'k (b t) d -> b (t k) d', b = B, t = T, d = D)
        part_down = rearrange(part_down,'k (b t) d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        part_up = rearrange(part_up,'k (b t) d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
        
        # mlp-mix
        for mixer_block in self.mixer_blocks:
            part = mixer_block(part)
        for mixer_block in self.mixer_blocks_down:
            part_down = mixer_block(part_down)
        for mixer_block in self.mixer_blocks_up:
            part_up = mixer_block(part_up)
        
        # 输出特征
        if self.fusion == 'mean_proj_cat':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            # part_down = self.part_up_linear(part_down)
            part_up = torch.mean(part_up, dim=1)
            # part_up = self.part_down_linear(part_up)
            part_fusion = part_down + part + part_up
            return part_fusion           
        elif self.fusion == 'divided_cat_1':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            part_up = torch.mean(part_up, dim=1)
            return part_down, part, part_up 


# 模块1 + 2
class Part_Token_Attention_V13_Cross_SlowFast_T(nn.Module):
    """ 
    这里的仅仅改变时间分辨率, 特征通道保持不变！
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
                mlp_ratio=4., fusion='divided_cat_1'):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        
        self.PartLayers = _get_clones(partlayer, self.num_part_layers)
       

        # slowfast
        # 下采样特征图
        z_block_size = 2       
        self.down_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.Conv3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))
        )
        # 上采样特征图
        self.up_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.ConvTranspose3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))       
        )
          
        self.fusion = fusion
           
        # Initialize weights
        self.apply(self._init_weights)

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

    # def forward_features(self, x):
    def forward(self, x):
        
        B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = self.num_parts
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
        
        # papre up-down conv
        part_up_down = part.clone() 
        part_up_down = part_up_down.permute(1, 0, 2) # [B*T, K, D] --> [K, B*T, D]
        L, NT, C = part_up_down.shape
        N = NT // T
        H = W = int(L ** 0.5)
        part_up_down = part_up_down.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 3, 3]
        
        # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
        part_down = part_up_down.clone()
        part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
        part_down = part_down.flatten(2).transpose(1,2) # [13, 2*9, 768]
        part_down = rearrange(part_down,'b (t k) d -> (b t) k d',  b= B, t=int(T/2), d=int(D))

        # up_conv
        part_up = part_up_down.clone()
        part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
        part_up = part_up.flatten(2).transpose(1,2) # [13, 8*9, 1536]
        part_up = rearrange(part_up,'b (t k) d -> (b t) k d',  b= B, t=int(T*2), d=int(D))

        # take care of cls token
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
        # maping cls_token_dowm/up, up or down cls_token channel dim
        init_cls_token_down = x[:, 0, :].unsqueeze(1)
        cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
        cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
        init_cls_token_up = x[:, 0, :].unsqueeze(1)
        cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
        cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

        # cat cls_token and part_token 
        part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
        part = part.permute(1, 0, 2) # [k+1, b*t, d]

        part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
        part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

        part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
        part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
        
        part = rearrange(part,'k (b t) d -> b (t k) d', b = B, t = T, d = D)
        part_down = rearrange(part_down,'k (b t) d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        part_up = rearrange(part_up,'k (b t) d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
        
        
        # 输出特征
        if self.fusion == 'mean_proj_cat':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            # part_down = self.part_up_linear(part_down)
            part_up = torch.mean(part_up, dim=1)
            # part_up = self.part_down_linear(part_up)
            part_fusion = part_down + part + part_up
            return part_fusion           
        elif self.fusion == 'divided_cat_1':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            part_up = torch.mean(part_up, dim=1)
            return part_down, part, part_up 

                  
# 模块1 + 2 + 3
class Part_Token_Attention_V13_Cross_SlowFast_T_Tcross(nn.Module):
    """ 
    这里的仅仅改变时间分辨率, 特征通道保持不变！
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
                mlp_ratio=4., fusion='divided_cat_1'):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        part_t_cross_att = vit_helper.part_slow_fast_cross_att(embed_dim=self.embed_dim,
                                                                nhead=self.nheads,
                                                                mlp_ratio=self.mlp_ratio,
                                                                dropout=0.1)

        self.PartLayers = _get_clones(partlayer, self.num_part_layers)
        self.part_slowatt_layers = _get_clones(part_t_cross_att, self.num_part_layers)
        self.part_fastatt_layers = _get_clones(part_t_cross_att, self.num_part_layers) # sfa的层数与p2p的层数相同的       
        self.norm_u = nn.LayerNorm(self.embed_dim)
        self.norm_d = nn.LayerNorm(self.embed_dim)
        #-------mlp-mixer----------
        self.mixer_blocks=nn.ModuleList([])
        token_dim = int(self.embed_dim/2)
        channel_dim = int(self.embed_dim * 4)
        depth = 1
        dropout = 0
        
        self.num_patches = int(self.temporal_resolution*self.num_parts) + self.temporal_resolution
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(self.embed_dim, self.num_patches, token_dim, channel_dim, dropout))
        
        self.mixer_blocks_down=nn.ModuleList([])
        token_dim_down = int(self.embed_dim/2)
        channel_dim_down = int(self.embed_dim * 4)
        self.num_patches_down = int(int(self.temporal_resolution /2) *self.num_parts) + int(self.temporal_resolution/2)        
        for _ in range(depth):
            self.mixer_blocks_down.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_down, 
                                                    token_dim=token_dim_down, channel_dim=channel_dim_down, dropout=dropout))
        
        self.mixer_blocks_up=nn.ModuleList([])
        token_dim_up = int(self.embed_dim/2)
        channel_dim_up = int(self.embed_dim * 4)
        self.num_patches_up = int(int(self.temporal_resolution * 2)*self.num_parts) + int(self.temporal_resolution*2)         
        for _ in range(depth):
            self.mixer_blocks_up.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_up, 
                                                    token_dim=token_dim_up, channel_dim=channel_dim_up, dropout=dropout))
        #-------mlp-mixer----------
        
        # slowfast
        # 下采样特征图
        z_block_size = 2       
        self.down_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.Conv3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))
        )
        # 上采样特征图
        self.up_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.ConvTranspose3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))       
        )
          
        self.fusion = fusion

        # SlowFast GATE
        # self.sfgate = SF_GATE(self.embed_dim)
        # SlowFast GATE V1
        self.sfgate = SF_GATE_V1(self.embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

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

    # def forward_features(self, x):
    def forward(self, x):
        
        B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = self.num_parts
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
         
        part_div = part # 单独返回，用作使得每一帧之间的原型都为负样本

        # papre up-down conv
        part_up_down = part.clone() 
        part_up_down = part_up_down.permute(1, 0, 2) # [B*T, K, D] --> [K, B*T, D]
        L, NT, C = part_up_down.shape
        N = NT // T
        H = W = int(L ** 0.5)
        part_up_down = part_up_down.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 3, 3]
        
        # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
        part_down = part_up_down.clone()
        part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
        part_down = part_down.flatten(2).transpose(1,2) # [13, 2*9, 768]
        part_down = rearrange(part_down,'b (t k) d -> (b t) k d',  b= B, t=int(T/2), d=int(D))

        # up_conv
        part_up = part_up_down.clone()
        part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
        part_up = part_up.flatten(2).transpose(1,2) # [13, 8*9, 1536]
        part_up = rearrange(part_up,'b (t k) d -> (b t) k d',  b= B, t=int(T*2), d=int(D))

        # take care of cls token
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
        # maping cls_token_dowm/up, up or down cls_token channel dim
        init_cls_token_down = x[:, 0, :].unsqueeze(1)
        cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
        cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
        init_cls_token_up = x[:, 0, :].unsqueeze(1)
        cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
        cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

        # cat cls_token and part_token 
        part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
        part = part.permute(1, 0, 2) # [k+1, b*t, d]

        part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
        part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

        part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
        part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
                             
        # permute part/partup/part_down for uniformerv2
        part = rearrange(part,'k (b t) d -> (b k) t d', b=B, t=T, d=D)    
        part_down = rearrange(part_down,'k (b t) d -> (b k) t d', b=B, t = int(T/2), d=D)   
        part_up = rearrange(part_up,'k (b t) d -> (b k) t d', b=B, t=int(T*2), d=D)
        # gate的原始流
        part_0 = rearrange(part,'(b k) t d -> b (t k) d', b = B, t = T, d = D)
        part_down_0 = rearrange(part_down,'(b k) t d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        part_up_0 = rearrange(part_up,'(b k) t d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
        # cross-att part_up/down betwwen part
        for layer in self.part_fastatt_layers:
            part_down = layer(self.norm_d(part_down), self.norm_d(part), self.norm_d(part))
        for layer in self.part_slowatt_layers:
            part_up = layer(self.norm_u(part_up), self.norm_u(part), self.norm_u(part))

        part = rearrange(part,'(b k) t d -> b (t k) d', b = B, t = T, d = D)
        part_down = rearrange(part_down,'(b k) t d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        part_up = rearrange(part_up,'(b k) t d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
                
        # mlp-mix,这里用作特征增强
        for mixer_block in self.mixer_blocks:
            part = mixer_block(part)
        for mixer_block in self.mixer_blocks_down:
            part_down = mixer_block(part_down)
        for mixer_block in self.mixer_blocks_up:
            part_up = mixer_block(part_up)
        
        # 输出特征
        if self.fusion == 'mean_proj_cat':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            # part_down = self.part_up_linear(part_down)
            part_up = torch.mean(part_up, dim=1)
            # part_up = self.part_down_linear(part_up)
            part_fusion = part_down + part + part_up
            return part_fusion           
        elif self.fusion == 'divided_cat_1':
            # avgpool
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            part_up = torch.mean(part_up, dim=1)
            # gate 分支
            part_down_0 = torch.mean(part_down_0, dim=1)
            part_up_0 = torch.mean(part_up_0, dim=1)
            part_0 = torch.mean(part_0, dim=1)             
            # sf_gate
            part_up = self.sfgate(part_up_0, part_0, part_up).squeeze(2)
            part_down = self.sfgate(part_down_0, part_0, part_down).squeeze(2)
            return part_down, part, part_up, part_div 

class SF_GATE(nn.Module):
    def __init__(self, channel_size):
        super(SF_GATE, self).__init__()    
        self.fc_d = nn.Conv1d(channel_size*2, channel_size, kernel_size=1)      
        self.fc_g = nn.Conv1d(channel_size, channel_size, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, part_up_down, part, part_up_down_g):
        mix = torch.cat((part_up_down.unsqueeze(2), part.unsqueeze(2)),dim=1)
        mix = self.relu(self.fc_d(mix))
        g = torch.sigmoid(self.fc_g(mix))
        part_up_down_g = part_up_down_g.unsqueeze(2)
        return g * part_up_down_g

class SF_GATE_V1(nn.Module):
    def __init__(self, channel_size):
        super(SF_GATE_V1, self).__init__()      
        self.fc_g1 = nn.Conv1d(channel_size, channel_size, kernel_size=1)
        self.fc_g2 = nn.Conv1d(channel_size, channel_size, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, part_up_down, part, part_up_down_g):
        g1 = self.fc_g1(part_up_down.unsqueeze(2))
        g2 = self.fc_g2(part.unsqueeze(2))
        g = torch.sigmoid(self.relu(g1+g2))
        part_up_down_g = part_up_down_g.unsqueeze(2)
        return g * part_up_down_g

# 模块1 + 2 + 3
class Part_Token_Attention_V13_Cross_SlowFast_T_Tcross_v2(nn.Module):
    """ 
    这里的仅仅改变时间分辨率, 特征通道保持不变！
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
                mlp_ratio=4., fusion='divided_cat_1'):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        part_t_cross_att = vit_helper.part_slow_fast_cross_att(embed_dim=self.embed_dim,
                                                                nhead=self.nheads,
                                                                mlp_ratio=self.mlp_ratio,
                                                                dropout=0.1)

        self.PartLayers = _get_clones(partlayer, self.num_part_layers)
        self.part_slowatt_layers = _get_clones(part_t_cross_att, self.num_part_layers)
        self.part_fastatt_layers = _get_clones(part_t_cross_att, self.num_part_layers)        
        self.norm_u = nn.LayerNorm(self.embed_dim)
        self.norm_d = nn.LayerNorm(self.embed_dim)
        #-------mlp-mixer----------
        self.mixer_blocks=nn.ModuleList([])
        token_dim = int(self.embed_dim/2)
        channel_dim = int(self.embed_dim * 4)
        depth = 1
        dropout = 0
        
        self.num_patches = int(self.temporal_resolution*self.num_parts) + self.temporal_resolution
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(self.embed_dim, self.num_patches, token_dim, channel_dim, dropout))
        
        self.mixer_blocks_down=nn.ModuleList([])
        token_dim_down = int(self.embed_dim/2)
        channel_dim_down = int(self.embed_dim * 4)
        self.num_patches_down = int(int(self.temporal_resolution /2) *self.num_parts) + int(self.temporal_resolution/2)        
        for _ in range(depth):
            self.mixer_blocks_down.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_down, 
                                                    token_dim=token_dim_down, channel_dim=channel_dim_down, dropout=dropout))
        
        self.mixer_blocks_up=nn.ModuleList([])
        token_dim_up = int(self.embed_dim/2)
        channel_dim_up = int(self.embed_dim * 4)
        self.num_patches_up = int(int(self.temporal_resolution * 2)*self.num_parts) + int(self.temporal_resolution*2)         
        for _ in range(depth):
            self.mixer_blocks_up.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_up, 
                                                    token_dim=token_dim_up, channel_dim=channel_dim_up, dropout=dropout))
        #-------mlp-mixer----------
        
        # slowfast
        # 下采样特征图
        z_block_size = 2       
        self.down_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.Conv3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))
        )
        # 上采样特征图
        self.up_conv = nn.Sequential(
            nn.BatchNorm3d(self.embed_dim),
            nn.ConvTranspose3d(in_channels=self.embed_dim,
            out_channels=int(self.embed_dim),
            kernel_size=(z_block_size,1,1),
            stride=(z_block_size,1,1))       
        )
          
        self.fusion = fusion
           
        # Initialize weights
        self.apply(self._init_weights)

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

    # def forward_features(self, x):
    def forward(self, x):
        
        B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = self.num_parts
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
        
        # papre up-down conv
        part_up_down = part.clone() 
        part_up_down = part_up_down.permute(1, 0, 2) # [B*T, K, D] --> [K, B*T, D]
        L, NT, C = part_up_down.shape
        N = NT // T
        H = W = int(L ** 0.5)
        part_up_down = part_up_down.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 3, 3]
        
        # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
        part_down = part_up_down.clone()
        part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
        part_down = part_down.flatten(2).transpose(1,2) # [13, 2*9, 768]
        part_down = rearrange(part_down,'b (t k) d -> (b t) k d',  b= B, t=int(T/2), d=int(D))

        # up_conv
        part_up = part_up_down.clone()
        part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
        part_up = part_up.flatten(2).transpose(1,2) # [13, 8*9, 1536]
        part_up = rearrange(part_up,'b (t k) d -> (b t) k d',  b= B, t=int(T*2), d=int(D))

        # take care of cls token
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
        # maping cls_token_dowm/up, up or down cls_token channel dim
        init_cls_token_down = x[:, 0, :].unsqueeze(1)
        cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
        cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
        init_cls_token_up = x[:, 0, :].unsqueeze(1)
        cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
        cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

        # cat cls_token and part_token 
        part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
        part = rearrange(part,'(b t) k d -> b (t k) d', b=B, t=T, d=D)

        # part = part.permute(1, 0, 2) # [k+1, b*t, d]

        part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
        part_down = rearrange(part,'(b t) k d -> b (t k) d', b=B, t=int(T/2), d=D)
        # part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

        part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
        part_up = rearrange(part,'(b t) k d -> b (t k) d', b=B, t=int(T*2), d=D)
        # part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
                             
        # # permute part/partup/part_down for uniformerv2
        # part = rearrange(part,'k (b t) d -> (b k) t d', b=B, t=T, d=D)    
        # part_down = rearrange(part_down,'k (b t) d -> (b k) t d', b=B, t = int(T/2), d=D)   
        # part_up = rearrange(part_up,'k (b t) d -> (b k) t d', b=B, t=int(T*2), d=D)   

        # cross-att part_up/down betwwen part
        for layer in self.part_fastatt_layers:
            part_down = layer(self.norm_d(part_down), self.norm_d(part), self.norm_d(part))
        for layer in self.part_slowatt_layers:
            part_up = layer(self.norm_u(part_up), self.norm_u(part), self.norm_u(part))

        # part = rearrange(part,'(b k) t d -> b (t k) d', b = B, t = T, d = D)
        # part_down = rearrange(part_down,'(b k) t d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        # part_up = rearrange(part_up,'(b k) t d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
                
        # mlp-mix
        for mixer_block in self.mixer_blocks:
            part = mixer_block(part)
        for mixer_block in self.mixer_blocks_down:
            part_down = mixer_block(part_down)
        for mixer_block in self.mixer_blocks_up:
            part_up = mixer_block(part_up)
        
        # 输出特征
        if self.fusion == 'mean_proj_cat':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            # part_down = self.part_up_linear(part_down)
            part_up = torch.mean(part_up, dim=1)
            # part_up = self.part_down_linear(part_up)
            part_fusion = part_down + part + part_up
            return part_fusion           
        elif self.fusion == 'divided_cat_1':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            part_up = torch.mean(part_up, dim=1)
            return part_down, part, part_up 


# # 模块1 + 2 + 3
# class Part_Token_Attention_V13_Cross_SlowFast_T_Tcross_v1(nn.Module):
#     """ 
#     这里的仅仅改变时间分辨率, 特征通道保持不变！
#     """
#     def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
#                 mlp_ratio=4., fusion='divided_cat_1'):
#         super().__init__()
#         self.temporal_resolution = temporal_resolution  # 4
#         self.num_part_layers = num_part_layers
#         self.embed_dim = embed_dim
#         self.nheads = nheads
#         self.mlp_ratio = mlp_ratio  # 4
        
#         # part embeding
#         self.num_parts = num_parts
#         self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
#         P2P = vit_helper.P2P_Attention(embed_dim=self.embed_dim, 
#                                                 nhead=self.nheads,
#                                                 mlp_ratio=self.mlp_ratio, 
#                                                 temporal_resolution=self.temporal_resolution)
#         SFC = vit_helper.SFC_Attention(embed_dim=self.embed_dim,
#                                                 nhead=self.nheads,
#                                                 mlp_ratio=self.mlp_ratio)

#         self.P2P_Layers = _get_clones(P2P, self.num_part_layers)
#         self.SFC_Layers = _get_clones(SFC, self.num_part_layers)       
#         self.norm_u = nn.LayerNorm(self.embed_dim)
#         self.norm_d = nn.LayerNorm(self.embed_dim)
#         #-------mlp-mixer----------
#         self.mixer_blocks=nn.ModuleList([])
#         token_dim = int(self.embed_dim/2)
#         channel_dim = int(self.embed_dim * 4)
#         depth = 1
#         dropout = 0
        
#         self.num_patches = int(self.temporal_resolution*self.num_parts) + self.temporal_resolution
#         for _ in range(depth):
#             self.mixer_blocks.append(MixerBlock(self.embed_dim, self.num_patches, token_dim, channel_dim, dropout))
        
#         self.mixer_blocks_down=nn.ModuleList([])
#         token_dim_down = int(self.embed_dim/2)
#         channel_dim_down = int(self.embed_dim * 4)
#         self.num_patches_down = int(int(self.temporal_resolution /2) *self.num_parts) + int(self.temporal_resolution/2)        
#         for _ in range(depth):
#             self.mixer_blocks_down.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_down, 
#                                                     token_dim=token_dim_down, channel_dim=channel_dim_down, dropout=dropout))
        
#         self.mixer_blocks_up=nn.ModuleList([])
#         token_dim_up = int(self.embed_dim/2)
#         channel_dim_up = int(self.embed_dim * 4)
#         self.num_patches_up = int(int(self.temporal_resolution * 2)*self.num_parts) + int(self.temporal_resolution*2)         
#         for _ in range(depth):
#             self.mixer_blocks_up.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_up, 
#                                                     token_dim=token_dim_up, channel_dim=channel_dim_up, dropout=dropout))
#         #-------mlp-mixer----------
        
#         # slowfast
#         # 下采样特征图
#         z_block_size = 2       
#         self.down_conv = nn.Sequential(
#             nn.BatchNorm3d(self.embed_dim),
#             nn.Conv3d(in_channels=self.embed_dim,
#             out_channels=int(self.embed_dim),
#             kernel_size=(z_block_size,1,1),
#             stride=(z_block_size,1,1))
#         )
#         # 上采样特征图
#         self.up_conv = nn.Sequential(
#             nn.BatchNorm3d(self.embed_dim),
#             nn.ConvTranspose3d(in_channels=self.embed_dim,
#             out_channels=int(self.embed_dim),
#             kernel_size=(z_block_size,1,1),
#             stride=(z_block_size,1,1))       
#         )
          
#         self.fusion = fusion
           
#         # Initialize weights
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def with_pos_embed(self, tensor, pos):
#         return  tensor + pos

#     # def forward_features(self, x):
#     def forward(self, x):
        
#         B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
#         D = x.size()[2]
#         T = self.temporal_resolution
#         K = self.num_parts
        
#         part_embed = self.part_embed.weight
#         part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
#         part = torch.zeros_like(part_embed)        

#         for layer in self.P2P_Layers:
#             part = layer(part, part_embed, x)
        
#         # papre up-down conv
#         part_up_down = part.clone() 
#         part_up_down = part_up_down.permute(1, 0, 2) # [B*T, K, D] --> [K, B*T, D]
#         L, NT, C = part_up_down.shape
#         N = NT // T
#         H = W = int(L ** 0.5)
#         part_up_down = part_up_down.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous() # [b, 768, 4, 3, 3]
        
#         # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
#         part_down = part_up_down.clone()
#         part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
#         part_down = part_down.flatten(2).transpose(1,2) # [13, 2*9, 768]
#         part_down = rearrange(part_down,'b (t k) d -> (b t) k d',  b= B, t=int(T/2), d=int(D))

#         # up_conv
#         part_up = part_up_down.clone()
#         part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
#         part_up = part_up.flatten(2).transpose(1,2) # [13, 8*9, 1536]
#         part_up = rearrange(part_up,'b (t k) d -> (b t) k d',  b= B, t=int(T*2), d=int(D))

#         # take care of cls token
#         init_cls_token = x[:, 0, :].unsqueeze(1)
#         cls_token = init_cls_token.repeat(1, T, 1)
#         cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
#         # maping cls_token_dowm/up, up or down cls_token channel dim
#         init_cls_token_down = x[:, 0, :].unsqueeze(1)
#         cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
#         cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
#         init_cls_token_up = x[:, 0, :].unsqueeze(1)
#         cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
#         cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

#         # cat cls_token and part_token 
#         part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
#         part = part.permute(1, 0, 2) # [k+1, b*t, d]

#         part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
#         part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

#         part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
#         part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
                             
#         # permute part/partup/part_down for uniformerv2
#         part = rearrange(part,'k (b t) d -> (b k) t d', b=B, t=T, d=D)    
#         part_down = rearrange(part_down,'k (b t) d -> (b k) t d', b=B, t = int(T/2), d=D)   
#         part_up = rearrange(part_up,'k (b t) d -> (b k) t d', b=B, t=int(T*2), d=D)   

#         # cross-att part_up/down betwwen part
#         for layer in self.SFC_Layers:
#             part_down = layer(self.norm_d(part_down), self.norm_d(part), self.norm_d(part))
#         for layer in self.SFC_Layers:
#             part_up = layer(self.norm_u(part_up), self.norm_u(part), self.norm_u(part))

#         part = rearrange(part,'(b k) t d -> b (t k) d', b = B, t = T, d = D)
#         part_down = rearrange(part_down,'(b k) t d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
#         part_up = rearrange(part_up,'(b k) t d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
                
#         # mlp-mix
#         for mixer_block in self.mixer_blocks:
#             part = mixer_block(part)
#         for mixer_block in self.mixer_blocks_down:
#             part_down = mixer_block(part_down)
#         for mixer_block in self.mixer_blocks_up:
#             part_up = mixer_block(part_up)
        
#         # 输出特征
#         if self.fusion == 'mean_proj_cat':
#             part = torch.mean(part, dim=1)
#             part_down = torch.mean(part_down, dim=1)
#             # part_down = self.part_up_linear(part_down)
#             part_up = torch.mean(part_up, dim=1)
#             # part_up = self.part_down_linear(part_up)
#             part_fusion = part_down + part + part_up
#             return part_fusion           
#         elif self.fusion == 'divided_cat_1':
#             part = torch.mean(part, dim=1)
#             part_down = torch.mean(part_down, dim=1)
#             part_up = torch.mean(part_up, dim=1)
#             return part_down, part, part_up 

# 模块1 + 模块2 + 模块3
class Part_Token_Attention_V13_Cross_SlowFast_T_Tcross_2dconv(nn.Module):
    """ 
    这里的仅仅改变时间分辨率, 特征通道保持不变！
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, 
                mlp_ratio=4., fusion='divided_cat_1', sfa_layers=1):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.sfa_layers = sfa_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partl2part = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        part_t_cross_att = vit_helper.part_slow_fast_cross_att(embed_dim=self.embed_dim,
                                                                nhead=self.nheads,
                                                                mlp_ratio=self.mlp_ratio,
                                                                dropout=0.1)

        self.P2P = _get_clones(partl2part, self.num_part_layers)
        self.SFA = _get_clones(part_t_cross_att, self.sfa_layers)       
        self.norm_u = nn.LayerNorm(self.embed_dim)
        self.norm_d = nn.LayerNorm(self.embed_dim)
        #-------mlp-mixer----------
        self.mixer_blocks=nn.ModuleList([])
        token_dim = int(self.embed_dim/2)
        channel_dim = int(self.embed_dim * 4)
        depth = 1
        dropout = 0
        
        self.num_patches = int(self.temporal_resolution*self.num_parts) + self.temporal_resolution
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(self.embed_dim, self.num_patches, token_dim, channel_dim, dropout))
        
        self.mixer_blocks_down=nn.ModuleList([])
        token_dim_down = int(self.embed_dim/2)
        channel_dim_down = int(self.embed_dim * 4)
        self.num_patches_down = int(int(self.temporal_resolution /2) *self.num_parts) + int(self.temporal_resolution/2)        
        for _ in range(depth):
            self.mixer_blocks_down.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_down, 
                                                    token_dim=token_dim_down, channel_dim=channel_dim_down, dropout=dropout))
        
        self.mixer_blocks_up=nn.ModuleList([])
        token_dim_up = int(self.embed_dim/2)
        channel_dim_up = int(self.embed_dim * 4)
        self.num_patches_up = int(int(self.temporal_resolution * 2)*self.num_parts) + int(self.temporal_resolution*2)         
        for _ in range(depth):
            self.mixer_blocks_up.append(MixerBlock(int(self.embed_dim), num_patch=self.num_patches_up, 
                                                    token_dim=token_dim_up, channel_dim=channel_dim_up, dropout=dropout))
        #-------mlp-mixer----------
        
        # slowfast
        # 下采样特征图
        z_block_size = 2       
        self.down_conv = nn.Sequential(
            nn.BatchNorm2d(self.embed_dim),
            nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(z_block_size,1), stride=(z_block_size,1))
        )
        # 上采样特征图
        self.up_conv = nn.Sequential(
            nn.BatchNorm2d(self.embed_dim),
            nn.ConvTranspose2d(in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=(z_block_size,1),
            stride=(z_block_size,1))       
        )
          
        self.fusion = fusion
           
        # Initialize weights
        self.apply(self._init_weights)

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

    # def forward_features(self, x):
    def forward(self, x):
        
        B = N = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution
        K = self.num_parts
        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # [B*T, K, D]，这里的K代表了时空分辨率，即TS共有K个tokens
        part = torch.zeros_like(part_embed)        

        for layer in self.P2P:
            part = layer(part, part_embed, x)
        
        # papre up-down conv
        part_up_down = part.clone() 
        part_up_down = rearrange(part_up_down,'(b t) k d -> b d t k',  b= B, t=T, d=D)
        
        # down_conv: 将4帧时间分辨率降低为2帧，空间和通道数保持不变
        part_down = part_up_down.clone()
        part_down = self.down_conv(part_down) # [13, 768, 2, 3, 3]
        part_down = rearrange(part_down,'b d t k -> (b t) k d',  b= B, t=int(T/2), d=int(D))

        # up_conv
        part_up = part_up_down.clone()
        part_up = self.up_conv(part_up) # [13, 768, 8, 3, 3]
        part_up = rearrange(part_up,'b d t k -> (b t) k d',  b= B, t=int(T*2), d=int(D))

        # take care of cls token
        init_cls_token = x[:, 0, :].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token,'b t d -> (b t) d', b= B, t=T, d=D).unsqueeze(1)
        
        # maping cls_token_dowm/up, up or down cls_token channel dim
        init_cls_token_down = x[:, 0, :].unsqueeze(1)
        cls_token_down = init_cls_token_down.repeat(1, int(T/2), 1)
        cls_token_down = rearrange(cls_token_down,'b t d -> (b t) d', b= B, t=int(T/2), d=D).unsqueeze(1)
        
        init_cls_token_up = x[:, 0, :].unsqueeze(1)
        cls_token_up = init_cls_token_up.repeat(1, int(T*2), 1)
        cls_token_up = rearrange(cls_token_up,'b t d -> (b t) d', b= B, t=int(T*2), d=D).unsqueeze(1)

        # cat cls_token and part_token 
        part = torch.cat((cls_token, part), dim=1) # [b*t, k+1, d]
        part = part.permute(1, 0, 2) # [k+1, b*t, d]

        part_down = torch.cat((cls_token_down, part_down), dim=1) # [b*(t/2), k+1, d]
        part_down = part_down.permute(1, 0, 2) # [k+1, b*(t/2), d]

        part_up = torch.cat((cls_token_up, part_up), dim=1) # [b*(t, k+1, d]
        part_up = part_up.permute(1, 0, 2) # [k+1, b*(t*2), d]                
                             
        # permute part/partup/part_down for uniformerv2
        part = rearrange(part,'k (b t) d -> (b k) t d', b=B, t=T, d=D)    
        part_down = rearrange(part_down,'k (b t) d -> (b k) t d', b=B, t = int(T/2), d=D)   
        part_up = rearrange(part_up,'k (b t) d -> (b k) t d', b=B, t=int(T*2), d=D)   

        # cross-att part_up/down betwwen part
        for layer in self.SFA:
            part_down = layer(self.norm_d(part_down), self.norm_d(part), self.norm_d(part))
        for layer in self.SFA:
            part_up = layer(self.norm_u(part_up), self.norm_u(part), self.norm_u(part))

        part = rearrange(part,'(b k) t d -> b (t k) d', b = B, t = T, d = D)
        part_down = rearrange(part_down,'(b k) t d -> b (t k) d', b = B, t = int(T/2), d = int(D)) # [13, 20, 384]
        part_up = rearrange(part_up,'(b k) t d -> b (t k) d', b = B, t = int(T*2), d = int(D)) # [13, 80, 1536]
                
        # mlp-mix
        for mixer_block in self.mixer_blocks:
            part = mixer_block(part)
        for mixer_block in self.mixer_blocks_down:
            part_down = mixer_block(part_down)
        for mixer_block in self.mixer_blocks_up:
            part_up = mixer_block(part_up)
        
        # 输出特征
        if self.fusion == 'mean_proj_cat':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            # part_down = self.part_up_linear(part_down)
            part_up = torch.mean(part_up, dim=1)
            # part_up = self.part_down_linear(part_up)
            part_fusion = part_down + part + part_up
            return part_fusion           
        elif self.fusion == 'divided_cat_1':
            part = torch.mean(part, dim=1)
            part_down = torch.mean(part_down, dim=1)
            part_up = torch.mean(part_up, dim=1)
            return part_down, part, part_up 
   
                        
# # baseline + 模块1(part-cross)
# class SpaceTimeTransformer(nn.Module):
#     """ Vision Transformer
#     A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
#         https://arxiv.org/abs/2104.00650
#     Based off:
#      - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
#     lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
#     Notable differences:
#      - allows for variable length input frames (<= num_frames)
#      - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
#      - different attention block mechanism
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
#                  num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
#                  act_layer=nn.GELU, is_tanh_gating=False):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_chans (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
#             norm_layer: (nn.Module): normalization layer
#             num_frames: (int) maximum number of frames expected as input
#             time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
#                         as ViT.
#             attention_style: (str) how to attend to space and time.
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_frames = num_frames
#         self.embed_dim = embed_dim
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         print("######USING ATTENTION STYLE: ", attention_style)
#         if hybrid_backbone is not None:
#             raise NotImplementedError('hybrid backbone not implemented')
#         else:
#             self.patch_embed = VideoPatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
#         num_patches = self.patch_embed.num_patches
#         self.patches_per_frame = num_patches // num_frames

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, self.patches_per_frame + 1,
#                         embed_dim))  # remember to take pos_embed[1:] for tiling over time
#         self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

#         if ln_pre:
#             self.ln_pre = nn.LayerNorm(embed_dim)
#         else:
#             self.ln_pre = None

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             SpaceTimeBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
#                 attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#         # self.norm_d = norm_layer(embed_dim)
#         # self.norm_u = norm_layer(embed_dim)

#         # Representation layer
#         if representation_size:
#             # self.num_features = representation_size
#             self.num_features = self.embed_dim
#             # self.pre_logits = nn.Sequential(OrderedDict([
#             #     ('fc', nn.Linear(int(embed_dim * 2), representation_size)),
#             #     ('act', nn.Tanh())
#             # ]))    
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(int(embed_dim * 2), embed_dim)),
#                 ('act', nn.Tanh())
#             ]))         
#         else:
#             self.pre_logits = nn.Identity()

#         # Classifier head
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#         # 这里cat之后, feature变成了1536,所以需要是int(self.num_features * 2
#         # self.head = nn.Linear(int(self.num_features * 2), num_classes) if num_classes > 0 else nn.Identity()
#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)

#         # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
#         if num_frames == 1:
#             self.apply(self._init_weights)

#         # einops transformations
#         self.einops_from_space = 'b (f n) d'
#         self.einops_to_space = '(b f) n d'
#         self.einops_from_time = 'b (f n) d'
#         self.einops_to_time = '(b n) f d'

#         self.part_token_attention = Part_Token_Attention_V13_Cross(temporal_resolution=num_frames,num_part_layers=2,num_parts=9,embed_dim=self.embed_dim,nheads=num_heads,mlp_ratio=mlp_ratio)
#         # self.part_token_attention = Part_Token_Attention_V13_Cross_Uniformerv2_Local_MLP_MIXER_SlowFast_T(temporal_resolution=num_frames,
#         #                                                                                                 num_part_layers=2,
#         #                                                                                                 num_parts=9,
#         #                                                                                                 embed_dim=self.embed_dim,
#         #                                                                                                 nheads=num_heads,
#         #                                                                                                 mlp_ratio=mlp_ratio, 
#         #                                                                                                 mlp_miexer_depth=1, 
#         #                                                                                                 uniformer_layers=2, 
#         #                                                                                                 fusion='divided_cat_1')        
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def freeze_spatial_weights(self):
#         freeze_list = []
#         for n, p in self.named_parameters():
#             if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
#                 pass
#             else:
#                 p.requires_grad = False
#                 freeze_list.append(n)
#         print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

#     def freeze_temporal_weights(self):
#         freeze_list = []
#         for n, p in self.named_parameters():
#             if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
#                 p.requires_grad = False
#                 freeze_list.append(n)
#             else:
#                 pass
#         print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

#     def forward_features(self, x, use_checkpoint=False, cls_at_last=True):
#         # print(x.shape)
#         b, curr_frames, channels, _, _ = x.shape
#         x = self.patch_embed(x)
#         x = x.flatten(2).transpose(2, 1)
#         x = x.reshape(b, -1, self.patch_embed.embed_dim)

#         BF = x.shape[0]
#         cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
#         cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
#         tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
#         # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
#         tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
#         total_pos_embed = tile_pos_embed + tile_temporal_embed
#         total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

#         curr_patches = x.shape[1]
#         x = x + total_pos_embed[:, :curr_patches]
#         if self.ln_pre is not None:
#             x = self.ln_pre(x)
#         x = self.pos_drop(x)
#         n = self.patches_per_frame
#         f = curr_frames

#         for blk in self.blocks:
#             x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
#                     self.einops_to_time,
#                     time_n=n, space_f=f, use_checkpoint=use_checkpoint)
        
#         # add part cross att
#         part = self.part_token_attention(x)
#         # part_down, part, part_up = self.part_token_attention(x)
          
#         if cls_at_last:
#             x_cls = self.norm(x)[:, 0]
#             part =self.norm(part)
#             # part_down = self.norm_d(part_down)
#             # part_up = self.norm_u(part_up)
#             x = torch.cat((x_cls, part), dim=1) 
#             # x = torch.cat((x_cls, part_down), dim=1) + torch.cat((x_cls, part), dim=1) + torch.cat((x_cls, part_up), dim=1)
#             x = self.pre_logits(x) # 并没有linear,而是直接传出去的
#             return x
#         else:
#             return self.norm(x)
#     def forward(self, x, use_checkpoint=False):
#         # Note:  B C T H W => B T C H W
#         # The default input order is different from the one in Frozen-in-Time
#         x = x.permute(0, 2, 1, 3, 4).contiguous()
#         x = self.forward_features(x, use_checkpoint=use_checkpoint)
#         x = self.head(x)
#         return x


# baseline + 模块1 + 2 + 3
class SDP_SpaceTimeTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650
    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
                 act_layer=nn.GELU, is_tanh_gating=False,sfa_layer=1, d2_sfc=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        print("######USING ATTENTION STYLE: ", attention_style)
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        if ln_pre:
            self.ln_pre = nn.LayerNorm(embed_dim)
        else:
            self.ln_pre = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm_d = norm_layer(embed_dim)
        self.norm_u = norm_layer(embed_dim)
        self.d2_sfc = d2_sfc
        # Representation layer
        if representation_size:
            # self.num_features = representation_size
            self.num_features = self.embed_dim
            # self.pre_logits = nn.Sequential(OrderedDict([
            #     ('fc', nn.Linear(int(embed_dim * 2), representation_size)),
            #     ('act', nn.Tanh())
            # ]))    
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(int(embed_dim * 2), embed_dim)),
                ('act', nn.Tanh())
            ]))         
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # 这里cat之后, feature变成了1536,所以需要是int(self.num_features * 2
        # self.head = nn.Linear(int(self.num_features * 2), num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

        # self.part_token_attention = Part_Token_Attention_V13_Cross(temporal_resolution=num_frames,num_part_layers=2,num_parts=9,embed_dim=self.embed_dim,nheads=num_heads,mlp_ratio=mlp_ratio) # baseline_1
        # 只需要替换Part_Token_Attention_V13_Cross_SlowFast_T实现模块1+2 # baseline_1_2
        # baseline_1_2_3
        
        # self.part_token_attention = Part_Token_Attention_V13_Cross_SlowFast_T_Tcross(temporal_resolution=num_frames,
        #                                                                     num_part_layers=2,
        #                                                                     num_parts=9,
        #                                                                     embed_dim=self.embed_dim,
        #                                                                     nheads=num_heads,
        #                                                                     mlp_ratio=mlp_ratio, 
        #                                                                     fusion='divided_cat_1')
        if self.d2_sfc:
            self.part_token_attention = Part_Token_Attention_V13_Cross_SlowFast_T_Tcross_2dconv(temporal_resolution=num_frames,
                                                                                num_part_layers=1,
                                                                                num_parts=12,
                                                                                embed_dim=self.embed_dim,
                                                                                nheads=num_heads,
                                                                                mlp_ratio=mlp_ratio, 
                                                                                fusion='divided_cat_1',
                                                                                sfa_layers=sfa_layer)
        else:
            self.part_token_attention = Part_Token_Attention_V13_Cross_SlowFast_T_Tcross(temporal_resolution=num_frames,
                                                                                num_part_layers=2,
                                                                                num_parts=9,
                                                                                embed_dim=self.embed_dim,
                                                                                nheads=num_heads,
                                                                                mlp_ratio=mlp_ratio, 
                                                                                fusion='divided_cat_1')              


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_spatial_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                pass
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def freeze_temporal_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                p.requires_grad = False
                freeze_list.append(n)
            else:
                pass
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def forward_features(self, x, use_checkpoint=False, cls_at_last=True):
        # print(x.shape)
        b, curr_frames, channels, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for blk in self.blocks:
            x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f, use_checkpoint=use_checkpoint)
        
        # add part cross att
        # part = self.part_token_attention(x)
        part_down, part, part_up, part_div = self.part_token_attention(x)
          
        if cls_at_last:
            x_cls = self.norm(x)[:, 0]
            part =self.norm(part)
            part_down = self.norm_d(part_down)
            part_up = self.norm_u(part_up)
            # x = torch.cat((x_cls, part), dim=1) 
            x = torch.cat((x_cls, part_down), dim=1) + torch.cat((x_cls, part), dim=1) + torch.cat((x_cls, part_up), dim=1)
            x = self.pre_logits(x) # 并没有linear,而是直接传出去的
            return x, part_div
        else:
            return self.norm(x)
    def forward(self, x, use_checkpoint=False):
        # Note:  B C T H W => B T C H W
        # The default input order is different from the one in Frozen-in-Time
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.forward_features(x, use_checkpoint=use_checkpoint)
        x = self.head(x)
        return x



class Space_Prompt_Communcation(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
        self, 
        input_size: Tuple[int, int] = (224, 224),
        qkv_dim: int = 768,
        num_frames: int = 8,
        mlp_dropout: float = 0.0, 
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        num_part_layers: int = 2, 
        num_parts: int = 9, 
        in_feature_dim: int =768,
        patch_size: Tuple[int, int] = (16, 16), 
        num_heads: int = 12, 

        use_local_prompt=False
    ):
        super().__init__()
        self.attn = Attention_P(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.num_frames = num_frames  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        # mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        # self.mlp_p_ = vit_helper.Mlp_P_(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        self.use_local_prompt = use_local_prompt
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        
        self.PartLayers = _get_clones(partlayer, self.num_part_layers)
        
        if self.use_local_prompts:
            self.cls_proj = nn.Linear(in_feature_dim, in_feature_dim)
            self.local_prompts = nn.Parameter(torch.zeros(1, self.num_frames, in_feature_dim))
            self._initialize_cls_prompts(patch_size, in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)
    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def _initialize_cls_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.local_prompts.data, -val, val)

    # def forward_features(self, x):
    def forward(self, x):
        BT, N, C = x.size() # print("x.pos_shape",x.shape) # [8, 785, 768]
        T = self.num_frames
        B = BT//T

        if self.use_local_prompt:
            cls_token = x[:, 0, :].view(B, T, C) # 表示，每一帧的cls token
            cls_token_proj = self.cls_proj(cls_token)
            local_prompts = self.local_prompts.expand(B, -1, -1) # (B, T, D)
           
            # use additive conditioning
            local_prompts = local_prompts + cls_token_proj # 加上相应的cls token, 可被理解成加上身份信息 (B, T, D)

            # repeat across frames
            local_prompts = local_prompts.repeat_interleave(repeats=T, dim=0) # （BT, T, D）表示每一帧的都被T个prompt尽心编码，prompt有D维度
            x = torch.cat((x[:, :1, :], local_prompts, x[:, 1:, :]), dim=1) # 1 + T + T
       
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)

        
        part_embed = self.part_embed.weight
        part_embed = part_embed.unsqueeze(0).repeat(B*T, 1, 1) # 这个地方实际上，帧级别的prompt
        part = torch.zeros_like(part_embed)        

        for layer in self.PartLayers:
            part = layer(part, part_embed, x)
        
        part = rearrange(part,'(b t) k d -> b (t k) d', b = B, t =T, d=D)
   
        part = torch.mean(part, dim=1)

        return part


class Attention_P(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int,
        out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out


# 原始的lavila
# baseline
class SpaceTimeTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650
    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
                 act_layer=nn.GELU, is_tanh_gating=False, caption=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.caption = caption

        print("######USING ATTENTION STYLE: ", attention_style)
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        if ln_pre:
            self.ln_pre = nn.LayerNorm(embed_dim)
        else:
            self.ln_pre = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            # 实际上，输出为以下：
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.caption:
            self.prompts_visual_proj = nn.Linear(self.num_features, 512)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_spatial_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                pass
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def freeze_temporal_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                p.requires_grad = False
                freeze_list.append(n)
            else:
                pass
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def forward_features(self, x, use_checkpoint=False, cls_at_last=True):

        b, curr_frames, channels, _, _ = x.shape # b=12, num_frames=16

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0] # [B, TS, D]
        # CLS_Tokesn [B, 1, D]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1) # [B, TS+1, D]

        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1) # [b, 1, d]
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1) # [b, ts, d]

        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1) # [1, ts,d]
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1] # 
        x = x + total_pos_embed[:, :curr_patches]
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames
        # print("x送入到attention之前的尺寸", x.size()) # [b, 1 + t*s, d]
        for blk in self.blocks:
            x = blk(x, 
                    self.einops_from_space, 
                    self.einops_to_space, 
                    self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f, 
                    use_checkpoint=use_checkpoint)

        if cls_at_last:
            x = self.norm(x)[:, 0]
            x = self.pre_logits(x) # 实际上，并没有
            if self.caption:
                x_proj = self.prompts_visual_proj(x)
                return x, x_proj
            else:
                return x
        else:
            return self.norm(x)

    def forward(self, x, use_checkpoint=False):
        # Note:  B C T H W => B T C H W
        # The default input order is different from the one in Frozen-in-Time
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.caption:
            x, x_proj = self.forward_features(x, use_checkpoint=use_checkpoint)
            x = self.head(x)
            return x, x_proj            
        else:
            x = self.forward_features(x, use_checkpoint=use_checkpoint)
            x = self.head(x)
            return x

# 模块1
class Space_Prompt_Attention(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, temporal_resolution=8, num_part_layers=2, num_parts=9, embed_dim=768, nheads=12, mlp_ratio=4.):
        super().__init__()
        self.temporal_resolution = temporal_resolution  # 4
        self.num_part_layers = num_part_layers
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio  # 4
        # mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        # self.mlp_p_ = vit_helper.Mlp_P_(in_features=self.embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        # part embeding
        self.num_parts = num_parts
        self.part_embed = nn.Embedding(self.num_parts, self.embed_dim)
        
        partlayer = vit_helper.PartDecoderLayer4(embed_dim=self.embed_dim, 
                                                nhead=self.nheads,
                                                mlp_ratio=self.mlp_ratio, 
                                                dropout=0.1,
                                                temporal_resolution=self.temporal_resolution)
        
        self.PartLayers = _get_clones(partlayer, self.num_part_layers)

    def with_pos_embed(self, tensor, pos):
        return  tensor + pos

    def forward(self, space_prompt, space_prompt_position, x):
        B = x.size()[0] # print("x.pos_shape",x.shape) # [8, 785, 768]
        D = x.size()[2]
        T = self.temporal_resolution

        for layer in self.PartLayers:
            part = layer(space_prompt, space_prompt_position, x)
        # part = torch.mean(part, dim=1) # [b*t, 1, d]       
        # part = rearrange(part,'(b t) k d -> b (t k) d', b = B, t =T, d=D)
        return part


# # baseline + space_prmmpt + time_prompt
# class SpaceTimeTransformer(nn.Module):
#     """ Vision Transformer
#     A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
#         https://arxiv.org/abs/2104.00650
#     Based off:
#      - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
#     lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
#     Notable differences:
#      - allows for variable length input frames (<= num_frames)
#      - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
#      - different attention block mechanism
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
#                  num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
#                  act_layer=nn.GELU, is_tanh_gating=False, caption=False,
#                  use_local_prompts = False, 
#                 #  use_global_prompts = False, 
#                  num_space_prompts = False):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_chans (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
#             norm_layer: (nn.Module): normalization layer
#             num_frames: (int) maximum number of frames expected as input
#             time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
#                         as ViT.
#             attention_style: (str) how to attend to space and time.
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_frames = num_frames
#         self.embed_dim = embed_dim
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         self.caption = caption
        
#         #### Vita_CLIP prompt #######
#         self.use_local_prompts = use_local_prompts
#         # self.use_global_prompts = use_global_prompts
#         self.num_space_prmmpts = num_space_prompts

#         print("######USING ATTENTION STYLE: ", attention_style)
#         if hybrid_backbone is not None:
#             raise NotImplementedError('hybrid backbone not implemented')
#         else:
#             self.patch_embed = VideoPatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
#         num_patches = self.patch_embed.num_patches
#         self.patches_per_frame = num_patches // num_frames

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, self.patches_per_frame + 1,
#                         embed_dim))  # remember to take pos_embed[1:] for tiling over time
#         self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

#         if ln_pre:
#             self.ln_pre = nn.LayerNorm(embed_dim)
#         else:
#             self.ln_pre = None

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             SpaceTimeBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
#                 attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating, use_space_prompt=self.use_local_prompts, num_frames=self.num_frames)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         # Representation layer
#         if representation_size:
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(embed_dim, representation_size)),
#                 ('act', nn.Tanh())
#             ]))
#         else:
#             # 实际上，输出为以下：
#             self.pre_logits = nn.Identity()

#         # Classifier head
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#         if self.caption:
#             self.prompts_visual_proj = nn.Linear(self.num_features, 512)

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)

#         # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
#         if num_frames == 1:
#             self.apply(self._init_weights)

#         # einops transformations
#         self.einops_from_space = 'b (f n) d'
#         self.einops_to_space = '(b f) n d'
#         self.einops_from_time = 'b (f n) d'
#         self.einops_to_time = '(b n) f d'

#         if self.use_local_prompts:
#             self.space_prompt = nn.Embedding(self.space_prmmpts, self.embed_dim)


#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def freeze_spatial_weights(self):
#         freeze_list = []
#         for n, p in self.named_parameters():
#             if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
#                 pass
#             else:
#                 p.requires_grad = False
#                 freeze_list.append(n)
#         print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

#     def freeze_temporal_weights(self):
#         freeze_list = []
#         for n, p in self.named_parameters():
#             if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
#                 p.requires_grad = False
#                 freeze_list.append(n)
#             else:
#                 pass
#         print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

#     def forward_features(self, x, use_checkpoint=False, cls_at_last=True):

#         b, curr_frames, channels, _, _ = x.shape # b=12, num_frames=16

#         x = self.patch_embed(x)
#         x = x.flatten(2).transpose(2, 1)
#         x = x.reshape(b, -1, self.patch_embed.embed_dim)

#         BF = x.shape[0] # [B, TS, D]
#         # CLS_Tokesn [B, 1, D]
#         cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1) # [B, TS+1, D]

#         # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
#         cls_embed = self.pos_embed[:, 0, :].unsqueeze(1) # [b, 1, d]
#         tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1) # [b, ts, d]

#         # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
#         tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1) # [1, ts,d]
#         total_pos_embed = tile_pos_embed + tile_temporal_embed
#         total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

#         curr_patches = x.shape[1] # 
#         x = x + total_pos_embed[:, :curr_patches]
#         if self.ln_pre is not None:
#             x = self.ln_pre(x)
#         x = self.pos_drop(x)
#         n = self.patches_per_frame
#         f = curr_frames
#         # print("x送入到attention之前的尺寸", x.size()) # [b, 1 + t*s, d]

#         ########### Design Space Prompt###############
#         if self.use_local_prompts:
#             space_prompt = self.space_prompt.weight
#             space_prompt_position = space_prompt.unsqueeze(0).repeat(b*f, 1, 1) # 维度是 [B*T, K, D]，这里的K代表了每一帧设置K个可学习的tokens
#             space_prompt = torch.zeros_like(space_prompt_position)   
        
#         for blk in self.blocks:
#             if self.use_local_prompts:
#                 x, space_prompt = blk(x,
#                         space_prompt,
#                         space_prompt_position,
#                         self.einops_from_space, 
#                         self.einops_to_space, 
#                         self.einops_from_time,
#                         self.einops_to_time,
#                         time_n=n, space_f=f, 
#                         use_checkpoint=use_checkpoint)
#             else:
#                 x = blk(x,
#                         self.einops_from_space, 
#                         self.einops_to_space, 
#                         self.einops_from_time,
#                         self.einops_to_time,
#                         time_n=n, space_f=f, 
#                         use_checkpoint=use_checkpoint)

#         if cls_at_last:
#             x = self.norm(x)[:, 0]
#             x = self.pre_logits(x) # 实际上，并没有
#             # space_prompt = rearrange(space_prompt,"")
#             if self.use_local_prompts:
#                 space_prompt = rearrange(space_prompt,'(b t) k d -> b (t k) d', b = b, t =f, d=x.shape(3))
#                 space_prompt = torch.mean(space_prompt,dim=1)
#                 x = torch.cat((x, space_prompt), dim=1)
#             if self.caption:
#                 x_proj = self.prompts_visual_proj(x)
#                 return x, x_proj
#             else:
#                 return x
#         else:
#             return self.norm(x)

#     def forward(self, x, use_checkpoint=False):
#         # Note:  B C T H W => B T C H W
#         # The default input order is different from the one in Frozen-in-Time
#         x = x.permute(0, 2, 1, 3, 4).contiguous()
#         if self.caption:
#             x, x_proj = self.forward_features(x, use_checkpoint=use_checkpoint)
#             x = self.head(x)
#             return x, x_proj            
#         else:
#             x = self.forward_features(x, use_checkpoint=use_checkpoint)
#             x = self.head(x)
#             return x
