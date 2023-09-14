# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, GPT2LMHeadModel

import lavila.models.loss as loss
from lavila.models.gpt2_gated import GPT2LMHeadModel as GatedGPT2LMHeadModel
from lavila.models.gpt2_gated import augment_gpt2_config
from lavila.models.narrator import VCLM_HF
from lavila.models.openai_clip import load as load_openai_clip
from lavila.models.openai_model import QuickGELU, Transformer
from lavila.models.timesformer import SpaceTimeTransformer
# from lavila.models.vit import TimeSformer
from lavila.models.utils import remap_keys, rsetattr
from lavila.models.tokenizer import SimpleTokenizer as _Tokenizer
from lavila.models.openai_clip import tokenize
import csv

class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(int(vision_model.num_features * 2), num_classes, bias=True)
        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, image, use_checkpoint=False):
        image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit

class VideoClassifierMultiHead(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 use_vanila_clip: bool,
                 use_text4vis: bool,
                 **kwargs,
                 ):
        super().__init__()
        self.use_vanila_clip = use_vanila_clip
        if self.use_vanila_clip:
            self.visual = ImageCLIP(name='ViT-B/16', device='cpu')
        else:
            self.visual = vision_model
            self.dropout = nn.Dropout(dropout)
        self.use_text4vis = use_text4vis
        
        # 返回的特征是768*2
        if self.use_vanila_clip:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(512, num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]   
        elif self.use_text4vis:
            self.proj = nn.ModuleList(
                [nn.Linear(int(vision_model.num_features), 512, bias=True),
                 nn.Linear(int(vision_model.num_features), 512, bias=True),
                 nn.Linear(int(vision_model.num_features), 512, bias=True)
                 ]
            ) # EPIC: [512 512...]              
            for m in self.proj:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

        else:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(int(vision_model.num_features), num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]  

            for m in self.fc_cls:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, image, use_checkpoint=False):
        image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        if self.use_text4vis:
            logit_list = [m(image_embed) for m in self.proj]            
        else:
            logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
        return logit_list

class VideoClassifierMultiHead_TEXT4VIS(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 use_vanila_clip: bool,
                 use_text4vis: bool,
                 **kwargs,
                 ):
        super().__init__()
        self.use_vanila_clip = use_vanila_clip
        if self.use_vanila_clip:
            self.visual = ImageCLIP(name='ViT-B/16', device='cpu')
        else:
            self.visual = vision_model
            self.dropout = nn.Dropout(dropout)
        self.use_text4vis = use_text4vis
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 返回的特征是768*2
        if self.use_vanila_clip:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(512, num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]   
        elif self.use_text4vis:
            self.proj = nn.ModuleList(
                [nn.Linear(int(vision_model.num_features), 512, bias=True),
                 nn.Linear(int(vision_model.num_features), 512, bias=True),
                 nn.Linear(int(vision_model.num_features), 512, bias=True)
                 ]
            ) # EPIC: [512 512...]              
            for m in self.proj:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

        else:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(int(vision_model.num_features), num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]  

            for m in self.fc_cls:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, image, text_emb, use_checkpoint=False):
        
        ###################### Encoder #############################
        image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        v_text_embed, n_text_embed, a_text_embed = text_emb[0], text_emb[1], text_emb[2]
        print("image_embed",image_embed.size())
        print("v_text_embed",v_text_embed.size())
        print("n_text_embed",n_text_embed.size())
        print("a_text_embed",a_text_embed.size())

        
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        if self.use_text4vis:
            image_embed_list = [m(image_embed) for m in self.proj]

        #################### Normlization ###########################
        v_text_embed = v_text_embed / v_text_embed.norm(dim=-1, keepdim=True)
        n_text_embed = n_text_embed / n_text_embed.norm(dim=-1, keepdim=True)
        a_text_embed = a_text_embed / a_text_embed.norm(dim=-1, keepdim=True)
                                                          
        ###################### Logit ###############################
        logit_scale = self.logit_scale.exp()
        v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_list[0], v_text_embed)
        n_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_list[1], n_text_embed)        
        a_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_list[2], a_text_embed)   

        return [v_logits_per_image, n_logits_per_image, a_logits_per_image]




# class VideoClassifierMultiHead_FShot(nn.Module):
#     def __init__(self,
#                  vision_model: nn.Module,
#                  text_model: nn.Module,
#                  dropout: float,
#                  num_classes_list: list,
#                  **kwargs,
#                  ):
#         super().__init__()
#         self.visual = vision_model
#         self.text = text_model
#         self.dropout = nn.Dropout(dropout)
#         # 返回的特征是768*2
#         self.fc_cls = nn.ModuleList(
#             [nn.Linear(512, num_classes, bias=True) for num_classes in num_classes_list]
#         ) # EPIC: [97 300 3806]

#         for m in self.fc_cls:
#             m.weight.data.normal_(mean=0.0, std=0.01)
#             m.bias.data.zero_()

#     def forward(self, image, use_checkpoint=False):
#         image_embed = self.visual(image, use_checkpoint=use_checkpoint)
#         text_embed = self.text(image, use_checkpoint=use_checkpoint)
#         if isinstance(image_embed, list):
#             assert len(image_embed) == 1
#             image_embed = image_embed[0]
#         logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
#         return logit_list

class VideoClassifierMultiHead_V1(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        # 返回的特征是768*2
        self.fc_cls = nn.ModuleList(
            # [nn.Linear(int(vision_model.num_features * 2), num_classes, bias=True) for num_classes in num_classes_list]
            [nn.Linear(int(vision_model.num_features), num_classes, bias=True) for num_classes in num_classes_list]
            # [nn.Linear(256, num_classes, bias=True) for num_classes in num_classes_list]
        ) # EPIC: [97 300 3806]

        for m in self.fc_cls:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

    def forward(self, image_embed):
        # image_embed, part = self.visual(image, use_checkpoint=use_checkpoint)
        # image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
        return logit_list


class TextCLIP(nn.Module):
    def __init__(self, name, device):
        super(TextCLIP, self).__init__()
        model, _ = load_openai_clip(name, device)
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class ImageCLIP(nn.Module):
    def __init__(self, name, device):
        super(ImageCLIP, self).__init__()
        model, _ = load_openai_clip(name, device)
        self.visual = model.visual

    def forward(self, image, use_checkpoint=False):
        image = image.permute(0,2,1,3,4)
        bb, tt, _, _, _ = image.shape
        x = self.visual(image.reshape(-1, *image.shape[2:]), use_checkpoint=use_checkpoint)  # ND
        x = x.view(bb, tt, -1)
        image_features = x.mean(1)
        # image_features = x.max(1).values
        return image_features        



class TextCLIP_Prompt(nn.Module):
    def __init__(self, name, device):
        super(TextCLIP_Prompt, self).__init__()
        model, _ = load_openai_clip(name, device)
        # self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

    def forward(self, prompts, tokenized_prompts):
        # [batch_size, n_ctx, d_model]
        # embedding_no_use = self.token_embedding
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class VideoClassifierMultiHead_V2(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.ModuleList(
            [nn.Linear(int(vision_model.num_features), num_classes, bias=True) for num_classes in num_classes_list]
        ) # EPIC: [97 300 3806]

    
        for m in self.fc_cls:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

        ############ Text Encoder ############
        self.text_encoder = TextCLIP(name='ViT-B/16',device='cpu')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, image, text, use_checkpoint=False):
        #################### Visual Encoder ########################
        image_embed, image_embed_ = self.visual(image, use_checkpoint=use_checkpoint)

        if not self.training:
            if isinstance(image_embed, list):
                assert len(image_embed) == 1
                image_embed = image_embed[0]
            logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
            return logit_list
                  
        #################### Text Encoder #########################

        verb_caption, noun_caption = text[0], text[1]
        v_text_embed = self.text_encoder(verb_caption)
        n_text_embed = self.text_encoder(noun_caption)
        v_text_embed = v_text_embed / v_text_embed.norm(dim=-1, keepdim=True)
        n_text_embed = n_text_embed / n_text_embed.norm(dim=-1, keepdim=True)
        
        ################# Cacluate Similarity #####################                                                        
        logit_scale = self.logit_scale.exp()
        v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_, v_text_embed)
        n_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_, n_text_embed)
        # v_logits_per_text  = v_logits_per_image.T
        # n_logits_per_text  = n_logits_per_image.T

        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]

        logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]

        return logit_list, v_logits_per_image, n_logits_per_image

class VideoClassifierMultiHead_V3(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 csc_text_prompt: bool,
                 metadata_verb: str,
                 metadata_noun: str,
                 use_vanila_clip: bool,
                 **kwargs,
                 ):
        super().__init__()        
        self.use_vanila_clip = use_vanila_clip
        if self.use_vanila_clip:
            self.visual = ImageCLIP(name='ViT-B/16', device='cpu')
        else:
            self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        if self.use_vanila_clip:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(512, num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]
        else:
            self.fc_cls = nn.ModuleList(
                [nn.Linear(int(vision_model.num_features), num_classes, bias=True) for num_classes in num_classes_list]
            ) # EPIC: [97 300 3806]
        self.csc_text_prompt = csc_text_prompt

        for m in self.fc_cls:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

        ############ Text Encoder ############
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 确定是否要使用CSC的文本编码，
        if self.csc_text_prompt:
            self.text_encoder = TextCLIP_Prompt(name='ViT-B/16',device='cpu')
            self.verb_class_name = []
            self.noun_class_name = []
            
            with open(metadata_verb) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)
                for row in csv_reader:
                    verb_class = str(row[1])
                    if verb_class not in self.verb_class_name:
                        self.verb_class_name.append(verb_class)              
            
            with open(metadata_noun) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)
                for row in csv_reader:
                    noun_class = str(row[1])
                    if noun_class not in self.noun_class_name:
                        self.noun_class_name.append(noun_class) 

            self.prompt_learner_verb = TextPromptLearner(
                            classnames=self.verb_class_name,
                            text_model=self.text_encoder,
                            num_prompts=8,
                            prompts_init='', # 默认是空
                            CSC=self.csc_text_prompt, # true
                            ctx_pos='end' # end
                            )
            self.tokenized_prompts_verb = self.prompt_learner_verb.tokenized_prompts

            self.prompt_learner_noun = TextPromptLearner(
                            classnames=self.noun_class_name,
                            text_model=self.text_encoder,
                            num_prompts=8,
                            prompts_init='', # 默认是空
                            CSC=self.csc_text_prompt, # true
                            ctx_pos='end' # end
                            )
            self.tokenized_prompts_noun = self.prompt_learner_noun.tokenized_prompts
        else:
            self.text_encoder = TextCLIP(name='ViT-B/16', device='cpu')
    def forward(self, image, text, use_checkpoint=False):
        
        #################### Visual Encoder ########################
        if self.use_vanila_clip:
            image_embed_ = self.visual(image)           
            if not self.training:
                if isinstance(image_embed_, list):
                    assert len(image_embed_) == 1
                    image_embed_ = image_embed_[0]
                logit_list = [m(self.dropout(image_embed_)) for m in self.fc_cls]
                return logit_list
        else:
            image_embed, image_embed_ = self.visual(image, use_checkpoint=use_checkpoint)
            if not self.training:
                if isinstance(image_embed, list):
                    assert len(image_embed) == 1
                    image_embed = image_embed[0]
                logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
                return logit_list
        ############### Text Encoder + CSC Prompt [optional] ###################
        verb_caption, noun_caption = text[0], text[1]
        if self.csc_text_prompt:
            prompts_v = self.prompt_learner_verb()
            tokenized_prompts_v = self.tokenized_prompts_verb
            v_text_embed = self.text_encoder(prompts_v, tokenized_prompts_v)

            prompts_n = self.prompt_learner_noun()
            tokenized_prompts_n = self.tokenized_prompts_noun
            n_text_embed = self.text_encoder(prompts_n, tokenized_prompts_n)
        else:
            v_text_embed = self.text_encoder(verb_caption)
            n_text_embed = self.text_encoder(noun_caption)
        
        #################### Normlization #############################
        v_text_embed = v_text_embed / v_text_embed.norm(dim=-1, keepdim=True)
        n_text_embed = n_text_embed / n_text_embed.norm(dim=-1, keepdim=True)
                                                          
        logit_scale = self.logit_scale.exp()
        v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_, v_text_embed)
        n_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_, n_text_embed)

        if self.use_vanila_clip:
            if isinstance(image_embed_, list):
                assert len(image_embed_) == 1
                image_embed_ = image_embed_[0]
        else:
            if isinstance(image_embed, list):
                assert len(image_embed) == 1
                image_embed = image_embed[0]

        if self.use_vanila_clip:
            logit_list = [m(self.dropout(image_embed_)) for m in self.fc_cls]
        else:
            logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]

        return logit_list, v_logits_per_image, n_logits_per_image

class TextPromptLearner(nn.Module):
    def __init__(self, classnames, text_model, num_prompts, prompts_init='', CSC=False, ctx_pos='end'):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        n_ctx = num_prompts
        ctx_init = prompts_init
        ctx_dim = text_model.ln_final.weight.shape[0]
        self.token_embedding = nn.Embedding(49408, 512)
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = text_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        # print(tokenized_prompts.shape)
        with torch.no_grad():
            embedding = text_model.token_embedding(tokenized_prompts)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        x = x @ self.image_projection
        return x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}

# class TextCLIP(nn.Module):
#     def __init__(self, model) :
#         super(TextCLIP, self).__init__()
#         self.model = model

#     def forward(self,text):
#         return self.model.encode_text(text)

# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  vision_width: int,
#                  vision_model: nn.Module,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int,
#                  tempearture_init=0.07,
#                  **kwargs,
#                  ):
#         super().__init__()

#         self.context_length = context_length
#         self.vision_width = vision_width

#         self.visual = vision_model

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask(),
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
#         self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm```
#         self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         print("=> initialize initial temperature with {}".format(tempearture_init))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

#         self.initialize_parameters()


#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)

#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

#         nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
#         nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         # mask = torch.empty(self.context_length*2, self.context_length*2)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask

#     def encode_image(self, image, use_checkpoint=False, apply_project=True):
#         x = self.visual(image, use_checkpoint=use_checkpoint)
#         # x_cls = x
#         if isinstance(x, list):
#             assert len(x) == 1
#             x = x[0]
#         if not apply_project:
#             return x
#         x = x @ self.image_projection
#         return x
#         # return x, x_


#     def encode_text(self, text, use_checkpoint=False):
#         # text:[b, 512*2]
#         x = self.token_embedding(text)
#         v, n = x.chunk(2, dim=1)

#         # v, n = text.chunk(2, dim=-1)
#         # v = self.token_embedding(v)  # [batch_size, n_ctx, d_model]
#         # n = self.token_embedding(n)  # [batch_size, n_ctx, d_model]

#         v = v + self.positional_embedding
#         n = n + self.positional_embedding

#         x = torch.cat((v, n), dim=1)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, use_checkpoint=use_checkpoint)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x)

#         # v = v.permute(1, 0, 2)  # NLD -> LND
#         # v = self.transformer(v, use_checkpoint=use_checkpoint)
#         # v = v.permute(1, 0, 2)  # LND -> NLD
#         # v_embed = self.ln_final(v)

#         # n = n.permute(1, 0, 2)  # NLD -> LND
#         # n = self.transformer(n, use_checkpoint=use_checkpoint)
#         # n = n.permute(1, 0, 2)  # LND -> NLD
#         # n_embed = self.ln_final(n)

#         v_embed, n_embed = x[:, :self.context_length, :], x[:, self.context_length:self.context_length*2, :]

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         text1, text2 = text.chunk(2,dim=-1)
#         v_embed = v_embed[torch.arange(v_embed.shape[0]), text1.argmax(dim=-1)] @ self.text_projection
#         n_embed = n_embed[torch.arange(n_embed.shape[0]), text2.argmax(dim=-1)] @ self.text_projection

#         return v_embed, n_embed


#     def forward(self, image, text, use_checkpoint=False, norm_embed=False):
#         if not self.training:
#             image_embed, _ = self.encode_image(image, use_checkpoint=use_checkpoint)
#             return image_embed
#         ################# Visual Encoder ##################
#         image_embed, image_embed_1 = self.encode_image(image, use_checkpoint=use_checkpoint)
#         image_embed_1 = image_embed_1 / image_embed_1.norm(dim=-1, keepdim=True)
        
#         ################# Text Encoder ####################
#         text = torch.cat((text[0], text[1]), dim=1)
#         v_text_embed, n_text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)
#         v_text_embed = v_text_embed / v_text_embed.norm(dim=-1, keepdim=True)
#         n_text_embed = n_text_embed / n_text_embed.norm(dim=-1, keepdim=True)

#         ################# Cacluate Similarity ####################                                                        
#         logit_scale = self.logit_scale.exp()
#         v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_1, v_text_embed)
#         n_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_embed_1, n_text_embed)

#         if norm_embed:
#             image_embed = F.normalize(image_embed, dim=-1)

#         return {'image_embed': image_embed,
#                 'v_logits_per_image': v_logits_per_image,
#                 'n_logits_per_image': n_logits_per_image,
#                 }

class CLIP_HF(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 text_width: int,
                 text_model: nn.Module,
                 text_use_cls_token: bool,
                 text_is_regressive: bool,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model
        self.text_width = text_width
        self.textual = text_model
        self.text_use_cls_token = text_use_cls_token
        self.text_is_regressive = text_is_regressive

        if 'projection' not in kwargs:
            self.projection = 'default'
        else:
            self.projection = kwargs['projection']
        if self.projection == 'default':
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        elif self.projection == 'frozen_in_time':
            self.image_projection = nn.Sequential(
                nn.Linear(vision_width, embed_dim)
            )
            self.text_projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(text_width, embed_dim)
            )
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self.projection == 'default':
            nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection, std=self.text_width ** -0.5)
        else:
            nn.init.normal_(self.image_projection[0].weight, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection[1].weight, std=self.text_width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        if self.projection == 'default':
            x = x @ self.image_projection
        else:
            x = self.image_projection(x)

        return x

    def encode_text(self, text, attention_mask=None, use_checkpoint=False):
        if use_checkpoint:
            if isinstance(self.textual, DistilBertModel):
                pass
                # print("DistilBertModel does not support gradient checkpointing. Skipping even if use_checkpoint=True")
            else:
                self.textual.gradient_checkpointing_enable()
        else:
            self.textual.gradient_checkpointing_disable()
        # text, attention_mask = text.squeeze(1), attention_mask.squeeze(1)
        # ^ uncomment this only when doing local debugging (distributed=False)
        x = self.textual(text, attention_mask=attention_mask)

        if self.text_is_regressive:
            # gpt-style
            x = x.last_hidden_state
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        else:
            # bert-style
            if self.text_use_cls_token:
                x = x.last_hidden_state
                x = x[torch.arange(x.shape[0]), 0, :]
            else:
                x = x.pooler_output
        if self.projection == 'default':
            x = x @ self.text_projection
        else:
            x = self.text_projection(x)

        return x

    def forward(self, image, text, mask=None, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, attention_mask=mask, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}

def get_loss(model, args, tokenizer=None):
    if model.startswith('CLIP'):
        return loss.CLIPLoss(
            use_vissl=args.contrastive_use_vissl,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )
    elif model.startswith('VCLM'):
        return loss.CaptionLoss(tokenizer=tokenizer)
    else:
        raise NotImplementedError

def get_metric_names(model):
    if model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    elif model.startswith('VCLM'):
        return ['loss', 'caption_loss', 'caption_acc', 'ppl']
    else:
        raise NotImplementedError


def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, caption=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
        caption=caption)
    
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    # 没有freeze
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    # 下面加载的是文本分支的权重

    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())


    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_LARGE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_336PX(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_BASE_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_HF_EGOVLP_DISTILBERT_BASE(num_frames=4, project_embed_dim=256, **kwargs):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
    )
    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
    vision_model.load_state_dict(vit_model.state_dict(), strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    kwargs.update({'projection': 'frozen_in_time'})
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        **kwargs,
    )

    return model


def CLIP_HF_TIMESFORMER_DISTILBERT_BASE(num_frames=4, drop_path_rate=0, temperature_init=0.07, project_embed_dim=256, **kwargs):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        drop_path_rate=drop_path_rate,
    )
    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
    vision_model.load_state_dict(vit_model.state_dict(), strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITB16_GPT2_LARGE(gated_xattn=False, freeze_lm_vclm=False,
                                  freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-large",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1280,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=20,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITB16_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                               freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITL14_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                               freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITL14_336PX_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                                     freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
    gated_xattn=False,
    random_init_gpt2=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=1, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    if not random_init_gpt2:
        print('Loading LM from pretrained weights..')
        for n, p in gpt2.named_parameters():
            rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=768,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=12,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_BASE_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
 ):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
 ):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14x) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_GPT2(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14x) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=1, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=768,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=12,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=3, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def CLIP_OPENAI_VITB32(**kwargs):
    model, _ = load_openai_clip('ViT-B/32', 'cpu')
    return model


def CLIP_OPENAI_VITB16(**kwargs):
    model, _ = load_openai_clip('ViT-B/16', 'cpu')
    return model


def CLIP_OPENAI_VITL14(**kwargs):
    model, _ = load_openai_clip('ViT-L/14', 'cpu')
    return model


def CLIP_OPENAI_VITL14_336PX(**kwargs):
    model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    return model

