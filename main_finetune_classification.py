# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize
from lavila.utils.preprocess import generate_tokenizer
from lavila.models.models import VideoClassifierMultiHead, VideoClassifierMultiHead_V1, VideoClassifierMultiHead_V2, TextCLIP, VideoClassifierMultiHead_TEXT4VIS
from lavila.models.openai_clip import load as load_openai_clip
# from lavila.data import VideoCaptionDatasetBase
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import traceback
def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila finetune and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='ek100_cls', type=str,
                        choices=['ek100_cls', 'egtea'])
    parser.add_argument('--root',
                        default='datasets/EK100/video_ht256px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-train',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
                        type=str, help='path to metadata file (train set)')
    parser.add_argument('--metadata-val',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--metadata-verb',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv',
                        type=str, help='path to metadata file (verb set)')
    parser.add_argument('--metadata-noun',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv',
                        type=str, help='path to metadata file (noun set)')
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for val')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')  
    parser.add_argument('--metadata-aux', default=None, nargs='+',
                        type=str, help='path to metadata file (auxiliary data with pseudo narrations)')
    # Model
    parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
    parser.add_argument('--norm-embed', action='store_true', help='norm text and visual embed if set True')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
    parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio for the last linear layer')
    parser.add_argument('--num-classes', default=3806, nargs='+', type=int, help='number of classes for the last linear layer')
    parser.add_argument('--use-vn-classifier', action='store_true')
    parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
    parser.add_argument('--contrastive-use-vissl', action='store_true', help='use contrastive implementation in vissl')
    parser.add_argument('--gated-xattn', action='store_true', help='use gated x-attn in VCLM_GPT2')
    parser.add_argument('--random-init-gpt2', action='store_true', help='random initialize params of text decoder in VCLM_GPT2')
    # Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--use-sgd', action='store_true')
    parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
    parser.add_argument('--lr', default=3e-3, type=float) # finetune用的学习率
    # parser.add_argument('--lr', default=3e-5, type=float) # 预训练用的学习率
    parser.add_argument('--lr-ratio-text', default=0.01, type=int)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--lr-multiplier-on-backbone', default=0.1, type=float, help='lr multiplier for the backbone')
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=None, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing')
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--save-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true',
                        help='use ZeroRedundancyOptimizer to save memory')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing during training for significantly less GPU usage')
    parser.add_argument('--temperature-init', default=0.07, type=float,
                        help='init. logit temperature for samples')    
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--caption', action='store_true', help='Weither use caption.')
    parser.add_argument('--hard-prompt-as-actionclip', default=False, type=bool, help='Weither use actionclip text prompt.')
    parser.add_argument('--hard-prompt-as-xclip', default=False, type=bool, help='Weither use xclip text prompt.')
    parser.add_argument('--csc_text_prompt', default=False, type=bool, help='Weither use csc_text_prompt.')
    # ref --> VitaCLIP_cvpr2023
    # use summary token attn
    parser.add_argument('--use_summary_token', action='store_true', dest='use_summary_token',
                        help='use summary token')
    # use local prompts
    parser.add_argument('--use_local_prompts', action='store_true', dest='use_local_prompts',
                        help='use local (frame-level conditioned) prompts')
    # use global prompts
    parser.add_argument('--use_global_prompts', action='store_true', dest='use_global_prompts',
                        help='use global (video-level unconditioned) prompts')
    parser.add_argument('--num_global_prompts', type=int, default=8,
                        help='number of global prompts')
    # set defaults
    parser.set_defaults(use_summary_token=False, use_local_prompts=False, use_global_prompts=False)
    parser.add_argument('--use-few-shot', action='store_true')
    parser.add_argument('--shot', default=1, type=int, help='Number of shot.')
    # parser.add_argument('--sfa_layer', default=1, type=int, help='Number OF SFA_lAYER.')
    # parser.add_argument('--d2_sfc', action='store_true', help='weither use 2d sfc.') # 默认是false,激活时为true
    # use local prompts
    parser.add_argument('--use_vanila_clip', action='store_true', dest='use_vanila_clip',
                        help='use_vanila_clip')

    parser.add_argument('--use_text4vis', action='store_true', dest='use_text4vis',
                        help='use_text4vis')
    return parser

try:
    def main(args):
        dist_utils.init_distributed_mode(args)

        global best_acc1
        random_seed(args.seed, dist_utils.get_rank())
        if args.use_vanila_clip:
            pass
        else:
            if args.pretrain_model:
                ckpt_path = args.pretrain_model
            else:
                raise Exception('no checkpoint found')
            ckpt = torch.load(ckpt_path, map_location='cpu')

        if args.use_vn_classifier:
            assert args.dataset == 'ek100_cls' and len(args.num_classes) == 3

        #################----使用纯净的CLIP-400M，即vanilla clip，则不需要加载下面的模型；否侧，则是加载ego4d的预训练权重---#####
        if args.use_vanila_clip:
            pass
        else:
            state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                state_dict[k.replace('module.', '')] = v

            old_args = ckpt['args']
            print("=> creating model: {}".format(old_args.model)) # CLIP_OPENAI_TIMESFORMER_BASE
            # print("=> this is old_args: {}".format(old_args))


        # 这里加载了完整的 “CLIP_OPENAI_TIMESFORMER_BASE” CLIP_OPENAI_VITB16
        if args.use_vanila_clip:
            pass
        else:
            model = getattr(models, old_args.model)(
                pretrained=old_args.load_visual_pretrained, # None
                pretrained2d=old_args.load_visual_pretrained is not None,
                text_use_cls_token=old_args.use_cls_token, # False
                project_embed_dim=old_args.project_embed_dim, # 256
                timesformer_gated_xattn=False,
                timesformer_freeze_space=False,
                num_frames=args.clip_length, # 输入的帧数
                drop_path_rate=args.drop_path_rate,
                caption=args.caption)

            if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
                # inflate weight
                print('=> inflating PE in models due to different frame numbers')
                state_dict = inflate_positional_embeds(
                    model.state_dict(), state_dict,
                    num_frames=args.clip_length,
                    load_temporal_fix='bilinear',
                )
            model.load_state_dict(state_dict, strict=False)
        
        # if ckpt['epoch'] is not None:
        #     print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
        # else:
        #     print("=> loaded resume checkpoint '{baseline}'")

        
        # 实际上，最终用于EPIC的模型，并没有利用完整的模型而是仅仅利用clip中的视觉模型
        if args.use_vn_classifier:
            if args.caption:
                if args.use_few_shot:
                    # if args.use_vanila_clip:
                    model = models.VideoClassifierMultiHead_V3(
                        None if args.use_vanila_clip else model.visual,
                        dropout=args.dropout_ratio,
                        num_classes_list=args.num_classes,
                        csc_text_prompt = args.csc_text_prompt,
                        metadata_verb = args.metadata_verb,
                        metadata_noun = args.metadata_noun,
                        use_vanila_clip = args.use_vanila_clip    
                    )
                else:                    
                    model = models.VideoClassifierMultiHead_V3(
                        model.visual,
                        dropout=args.dropout_ratio,
                        num_classes_list=args.num_classes,
                        csc_text_prompt = args.csc_text_prompt,
                        metadata_verb = args.metadata_verb,
                        metadata_noun = args.metadata_noun,
                        use_vanila_clip = args.use_vanila_clip 
                )       
            elif args.use_text4vis:
                model = models.VideoClassifierMultiHead_TEXT4VIS(
                    None if args.use_vanila_clip else model.visual,
                    dropout=args.dropout_ratio,
                    num_classes_list=args.num_classes,
                    use_vanila_clip=args.use_vanila_clip,
                    use_text4vis=args.use_text4vis
            )                
            else:
                model = models.VideoClassifierMultiHead(
                    None if args.use_vanila_clip else model.visual,
                    dropout=args.dropout_ratio,
                    num_classes_list=args.num_classes,
                    use_vanila_clip=args.use_vanila_clip,
                    use_text4vis=args.use_text4vis
            )
        else:
            assert len(args.num_classes) == 1
            model = models.VideoClassifier(
                model.visual,
                dropout=args.dropout_ratio,
                num_classes=args.num_classes[0]
            )
        
        # 打印出所使用的模型及其结构
        # print(" model: ", model)
        
        if args.distributed:
            if args.gpu is None:
                cur_device = torch.cuda.current_device()
            else:
                cur_device = args.gpu 
            model = model.cuda(device=cur_device)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=args.find_unused_parameters
            )
        # 打印模型所有的参数
        # print("model.named_parameters", model.named_parameters)
    

        p_wd, p_non_wd = [], []
        p_head_wd, p_head_non_wd = [], []
        for n, p in model.named_parameters():
            if 'fc_cls' in n:
                if 'bias' in n:
                    p_head_non_wd.append(p)
                else:
                    p_head_wd.append(p)
            # elif 'text_encoder' in n or 'prompt_learner_verb.token_embedding' in n or 'prompt_learner_noun.token_embedding' in n or not p.requires_grad:
            elif 'text_encoder' in n or not p.requires_grad:
                continue  # frozen weights
            elif p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                 p_non_wd.append(p)
            else:
                p_wd.append(p)

        optim_params = [
            {"params": p_wd, "weight_decay": args.wd,  "lr": args.lr * args.lr_multiplier_on_backbone},
            {"params": p_non_wd, "weight_decay": 0, "lr": args.lr * args.lr_multiplier_on_backbone},
            {"params": p_head_wd, "weight_decay": args.wd},
            {"params": p_head_non_wd, "weight_decay": 0}
        ]

        if args.use_zero:
            optimizer = ZeroRedundancyOptimizer(
                optim_params, optimizer_class=torch.optim.SGD if args.use_sgd else torch.optim.AdamW,
                lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
            )
        else:
            if args.use_sgd:
                optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=args.betas[0], weight_decay=args.wd)
            else:
                optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                            eps=args.eps, weight_decay=args.wd)
        scaler = amp.GradScaler(enabled=not args.disable_amp)
        # optionally resume from a checkpoint (takes precedence over autoresume)
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            args.resume = ''
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading resume checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
                args.start_epoch = epoch
                if not args.distributed:
                    state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        state_dict[k.replace('module.', '')] = v
                    result = model.load_state_dict(state_dict, strict=False)
                else:
                    result = model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(result)
                optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
                scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
                best_acc1 = checkpoint['best_acc1']
                print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})"
                    .format(args.resume, epoch, best_acc1))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        else:
            # auto-resume from latest checkpoint in output directory
            latest = os.path.join(args.output_dir, 'checkpoint.pt')
            if os.path.isfile(latest):
                print("=> loading latest checkpoint '{}'".format(latest))
                latest_checkpoint = torch.load(latest, map_location='cpu')
                args.start_epoch = latest_checkpoint['epoch']
                model.load_state_dict(latest_checkpoint['state_dict'])
                optimizer.load_state_dict(latest_checkpoint['optimizer'])
                scaler.load_state_dict(latest_checkpoint['scaler'])
                best_acc1 = latest_checkpoint['best_acc1']
                print("=> loaded latest checkpoint '{}' (epoch {})"
                    .format(latest, latest_checkpoint['epoch']))

        cudnn.benchmark = True

        # Data loading code
        print("=> creating dataset")
        if args.use_vanila_clip:
            tokenizer = SimpleTokenizer()
        elif old_args.model.endswith('DISTILBERT_BASE'):
            tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
        elif old_args.model.endswith('BERT_BASE'):
            tokenizer = MyBertTokenizer('bert-base-uncased')
        elif old_args.model.endswith('BERT_LARGE'):
            tokenizer = MyBertTokenizer('bert-large-uncased')
        elif old_args.model.endswith('GPT2'):
            tokenizer = MyGPT2Tokenizer('gpt2')
        elif old_args.model.endswith('GPT2_MEDIUM'):
            tokenizer = MyGPT2Tokenizer('gpt2-medium')
        elif old_args.model.endswith('GPT2_LARGE'):
            tokenizer = MyGPT2Tokenizer('gpt2-large')
        elif old_args.model.endswith('GPT2_XL'):
            tokenizer = MyGPT2Tokenizer('gpt2-xl')
        else:
            print("Using SimpleTokenizer because of model '{}'. "
                "Please check if this is what you want".format(old_args.model))
            tokenizer = SimpleTokenizer()

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

        if args.use_vanila_clip:
            crop_size = 224 
        else:
            crop_size = 224 if '336PX' not in old_args.model else 336
    
        transforms_list = [
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if args.use_vanila_clip:
           transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
        else:
            if 'OPENAI' in old_args.model:
                transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
            else:
                transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        train_transform = transforms.Compose(transforms_list)

        val_transform = transforms.Compose([
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in "CLIP_OPENAI_TIMESFORMER_BASE" else
                transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
                TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
                SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
            ])

        # build dataset

        _, mapping_vn2act = generate_label_map(args.dataset)
        
        if args.dataset == 'ek100_cls':
            args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
            args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
            args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
        
        num_clips_at_val = args.num_clips
        args.num_clips = 1
        train_dataset = datasets.get_downstream_dataset(
            train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
        )


        args.num_clips = num_clips_at_val
        val_dataset = datasets.get_downstream_dataset(
            val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
        )

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # disable distributed
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
        )
        print('len(train_loader) = {}'.format(len(train_loader)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=int(args.batch_size*4), shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
        )
        print('len(val_loader) = {}'.format(len(val_loader)))


        if args.use_text4vis:
            verb_caption = train_dataset.verb_caption
            noun_caption = train_dataset.noun_caption
            action_caption = train_dataset.action_caption
            verb_caption_val = val_dataset.verb_caption
            noun_caption_val = val_dataset.noun_caption
            action_caption_val = val_dataset.action_caption
            Text_Encoder = TextCLIP(name='ViT-B/16', device='cpu')            
            with torch.no_grad():
                v_classes_features = Text_Encoder(verb_caption)
                n_classes_features = Text_Encoder(noun_caption)
                a_classes_features = Text_Encoder(action_caption)

                v_classes_features_val = Text_Encoder(verb_caption_val)
                n_classes_features_val = Text_Encoder(noun_caption_val)
                a_classes_features_val = Text_Encoder(action_caption_val)
        if args.evaluate:
            if args.use_vn_classifier:
                if args.caption:
                    val_stats = validate_multihead(val_loader, model, args)
                else:
                    val_stats = validate_multihead(val_loader, model, args)
            else:
                val_stats = validate(val_loader, model, args)
            return

        if args.fix_lr:
            lr_schedule = None
        # lr_schedule: finetune默认是有的
        else:
            lr_schedule = cosine_scheduler(
                args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
                warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
            )

        if dist_utils.is_main_process() and args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(project='LaViLa', id=wandb_id, config=args, resume='allow')

        print(args)

        best_metric = 0.
        print("=> beginning training")
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train_stats = train(train_loader, model, visual_cls, criterion, criterion_caption_loss, optimizer, scaler, epoch, lr_schedule, args, old_args)
            if args.use_text4vis:
                train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 
                                    v_classes_features, n_classes_features, a_classes_features,
                                    args)
            else:
                train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

            is_epoch = ((epoch + 1) % args.save_freq) == 0

            print('=> saving checkpoint')
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': 0,
                'args': args,
            }, False, args.output_dir, is_epoch=is_epoch)

            if ((epoch + 1) % args.eval_freq) == 0:
                if args.use_vn_classifier: #v_classes_features, n_classes_features, a_classes_features,
                    val_stats = validate_multihead(val_loader, model, args)
                elif args.use_text4vis:
                    val_stats = validate_multihead(val_loader, model, v_classes_features_val, n_classes_features_val, a_classes_features_val, args)
                else:
                    val_stats = validate(val_loader, model, args)
                if val_stats['acc1'] > best_metric:
                    is_best = True
                    best_metric = val_stats['acc1']
                else:
                    is_best = False

                print('=> saving checkpoint')
                dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_metric,
                    'args': args,
                }, is_best, args.output_dir, is_epoch=is_epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch}

                if dist_utils.is_main_process():
                    if args.wandb:
                        wandb.log(log_stats)
                    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                        f.write(json.dumps(log_stats) + '\n')
except Exception as e:
    traceback.print_exc()


#  单loss
def train_0(train_loader, model, visual_cls, criterion, criterion_caption_loss, optimizer, scaler, epoch, lr_schedule, args, old_args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')

    metric_names = models.get_metric_names(old_args.model)
    if args.metadata_aux is not None:
        metric_names.extend(['num_gt', 'num_pseudo', 'clip_acc_gt', 'clip_acc_pseudo'])
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])    
    if args.caption:
        # progress = ProgressMeter(
        #     iters_per_epoch,
        #     [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb, *metrics.values()],
        #     prefix="Epoch: [{}]".format(epoch))
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))        
    else:
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    if args.caption:
        for data_iter, (images, target, noun_caption) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if args.caption:
                noun_caption = noun_caption.cuda(args.gpu, non_blocking=True)

            # compute output
            with amp.autocast(enabled=not args.disable_amp):

                if args.caption:
                    outputs = model(images, noun_caption, use_checkpoint=args.use_checkpoint, norm_embed=args.norm_embed)
                    # output = visual_cls(outputs['image_cls_embed'].cuda(args.gpu, non_blocking=True))
                else:
                    output = model(images, use_checkpoint=args.use_checkpoint)            

                if args.metadata_aux is None:
                    loss_dict = criterion_caption_loss(outputs)
                loss_ = loss_dict['loss']
                loss_ /= args.update_freq

                # if isinstance(output, list):
                #     assert len(output) == 3
                #     target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                #     loss = criterion(output[0], target_to_verb)
                #     target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                #     loss += criterion(output[1], target_to_noun)
                #     loss += criterion(output[2], target)
                # else:
                #     loss = criterion(output, target)
                # loss /= args.update_freq

            # if not math.isfinite(loss.item()):
            #     print("Loss is {}, stopping training".format(loss.item()))
            #     sys.exit(1)

            if not math.isfinite(loss_.item()):
                print("Loss is {}, stopping training".format(loss_.item()))
                sys.exit(1)


            scaler.scale(loss_).backward()
            # scaler.scale(loss_).backward(retain_graph=True)
            # scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            # if isinstance(output, list):
            #     target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            #     acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
            #     top1_verb.update(acc1_verb.item(), images.size(0))
            #     target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            #     acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
            #     top1_noun.update(acc1_noun.item(), images.size(0))
            #     acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
            #     losses.update(loss.item(), images.size(0))
            #     top1.update(acc1.item(), images.size(0))
            #     top5.update(acc5.item(), images.size(0))
            # else:
            #     output = torch.softmax(output, dim=1)
            #     acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #     losses.update(loss.item(), images.size(0))
            #     top1.update(acc1.item(), images.size(0))
            #     top5.update(acc5.item(), images.size(0))
            #     if args.dataset == 'ek100_cls':
            #         vi = get_marginal_indexes(args.actions, 'verb')
            #         ni = get_marginal_indexes(args.actions, 'noun')
            #         verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
            #         noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
            #         target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            #         target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            #         acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
            #         acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
            #         top1_verb.update(acc1_verb.item(), images.size(0))
            #         top1_noun.update(acc1_noun.item(), images.size(0))
            #     else:
            #         top1_verb.update(0., images.size(0))
            #         top1_noun.update(0., images.size(0))

            if hasattr(dist_utils.get_model(model), 'logit_scale'):
                # clamp logit scale to [0, 100]
                dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
            else:
                logit_scale = torch.nan
            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), args.batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        # 'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        # 'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                        **{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(), 'logit': logit_scale
                    })
                    
                progress.display(optim_iter)

    else:
        for data_iter, (images, target) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with amp.autocast(enabled=not args.disable_amp):

                output = model(images, use_checkpoint=args.use_checkpoint)            

                if isinstance(output, list):
                    assert len(output) == 3
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                    # loss += 0.01*div_loss

                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list):
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                    })
                progress.display(optim_iter)
            
        
        progress.synchronize()
    if args.caption:
        return {
            **{k: v.avg for k, v in metrics.items()},
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }
    else:
        return {
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }        

# 双loss: 视觉loss + 名词的对比loss
def train_1(train_loader, model, visual_cls, criterion, criterion_caption_loss, optimizer, scaler, epoch, lr_schedule, args, old_args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')

    metric_names = models.get_metric_names(old_args.model)
    if args.metadata_aux is not None:
        metric_names.extend(['num_gt', 'num_pseudo', 'clip_acc_gt', 'clip_acc_pseudo'])
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])    
    if args.caption:
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))        
    else:
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    if args.caption:
        for data_iter, (images, target, noun_caption) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if args.caption:
                noun_caption = noun_caption.cuda(args.gpu, non_blocking=True)

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                # output, part_div = model(images, use_checkpoint=args.use_checkpoint)
                # N, _, _ = part_div.size()
                
                # output = model(images, use_checkpoint=args.use_checkpoint)
                if args.caption:
                    outputs = model(images, noun_caption, use_checkpoint=args.use_checkpoint, norm_embed=args.norm_embed)
                    output = visual_cls(outputs['image_cls_embed'].cuda(args.gpu, non_blocking=True))
                else:
                    output = model(images, use_checkpoint=args.use_checkpoint)            

                if args.metadata_aux is None:
                    loss_dict = criterion_caption_loss(outputs)
                loss_ = loss_dict['loss']
                loss_ /= args.update_freq

                if isinstance(output, list):
                    assert len(output) == 3
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss_.item()))
                sys.exit(1)


            scaler.scale(loss_).backward(retain_graph=True)
            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list):
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))

            if hasattr(dist_utils.get_model(model), 'logit_scale'):
                # clamp logit scale to [0, 100]
                dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
            else:
                logit_scale = torch.nan
            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), args.batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                        **{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(), 'logit': logit_scale
                    })
                    
                progress.display(optim_iter)

    else:
        for data_iter, (images, target) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with amp.autocast(enabled=not args.disable_amp):

                output = model(images, use_checkpoint=args.use_checkpoint)            

                if isinstance(output, list):
                    assert len(output) == 3
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                    # loss += 0.01*div_loss

                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list):
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                    })
                progress.display(optim_iter)
            
        
        progress.synchronize()
    if args.caption:
        return {
            **{k: v.avg for k, v in metrics.items()},
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }
    else:
        return {
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }        

# 双loss，但是视觉loss和名词的对比loss加在一起
# def train(train_loader, model, visual_cls, criterion, criterion_caption_loss, optimizer, scaler, epoch, lr_schedule, args, old_args):
# def train(train_loader, model, visual_cls, criterion, optimizer, scaler, epoch, lr_schedule, args, old_args):
def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, 
          v_classes_features, n_classes_features, a_classes_features, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')

    # metric_names = models.get_metric_names(old_args.model)
    # if args.metadata_aux is not None:
    #     metric_names.extend(['num_gt', 'num_pseudo', 'clip_acc_gt', 'clip_acc_pseudo'])
        
    # iters_per_epoch = len(train_loader) // args.update_freq
    # metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])    
    if args.caption:
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
            prefix="Epoch: [{}]".format(epoch))   
    else:
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
            prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    if args.caption:
        # for data_iter, (images, target, v_caption, n_caption, a_caption) in enumerate(train_loader):
        for data_iter, (images, target, v_caption, n_caption) in enumerate(train_loader):

            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if args.caption:
                # list 是没法加载上cuda上的，因此加载的tokenizaer的Tensor
                # v_caption = v_caption.cuda(args.gpu, non_blocking=True)
                # n_caption = n_caption.cuda(args.gpu, non_blocking=True)
                # verb_gt_class_name = verb_gt_class_name.cuda(args.gpu, non_blocking=True)
                # noun_gt_class_name = noun_gt_class_name.cuda(args.gpu, non_blocking=True)
                caption = [v_caption, n_caption]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                if args.csc_text_prompt:
                    output, v_logits_per_image, n_logits_per_image = model(images, caption, use_checkpoint=args.use_checkpoint)
                    v_labels = torch.arange(v_logits_per_image.shape[0], device="cuda").long()
                    n_labels = torch.arange(n_logits_per_image.shape[0], device="cuda").long()
                    loss_ = (criterion(v_logits_per_image, v_labels) + 
                            criterion(n_logits_per_image, n_labels)) /2
                    loss_ /= args.update_freq                    
                else:
                    output, v_logits_per_image, n_logits_per_image = model(images, caption, use_checkpoint=args.use_checkpoint)
                    v_labels = torch.arange(v_logits_per_image.shape[0], device="cuda").long()
                    n_labels = torch.arange(n_logits_per_image.shape[0], device="cuda").long()
                    loss_ = (criterion(v_logits_per_image, v_labels) + 
                            criterion(n_logits_per_image, n_labels)) /2
                    loss_ /= args.update_freq
          
                # 视觉损失函数
                if isinstance(output, list):
                    assert len(output) == 3
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                    # 总的loss加在一起
                    loss += 0.3*loss_
                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            # scaler.scale(loss_).backward(retain_graph=True)
            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list):
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                    })
                    
                progress.display(optim_iter)
    elif args.use_text4vis:
        v_classes_features = v_classes_features.cuda(args.gpu, non_blocking=True)
        n_classes_features = n_classes_features.cuda(args.gpu, non_blocking=True)
        a_classes_features = a_classes_features.cuda(args.gpu, non_blocking=True)
        text_emb = [v_classes_features, v_classes_features, a_classes_features]
        for data_iter, (images, target) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)


            # compute output
            with amp.autocast(enabled=not args.disable_amp):

                output = model(images, text_emb, use_checkpoint=args.use_checkpoint)            

                if isinstance(output, list):
                    assert len(output) == 3
                    print("output[0].size",output[0].size()) # [12, 3568]
                    print("output[1].size",output[1].size())
                    print("output[2].size",output[2].size())
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list): # 默认是执行这个
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                    })
                progress.display(optim_iter)
            
        
        progress.synchronize()    
    else:
        for data_iter, (images, target) in enumerate(train_loader):
            optim_iter = data_iter // args.update_freq

            # measure data loading time
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            it = iters_per_epoch * epoch + optim_iter  # global training iteration
            for k, param_group in enumerate(optimizer.param_groups):
                if lr_schedule is not None:
                    param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
                else:
                    param_group['lr'] = lr_schedule[it]

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with amp.autocast(enabled=not args.disable_amp):

                output = model(images, use_checkpoint=args.use_checkpoint)            

                if isinstance(output, list):
                    assert len(output) == 3
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss = criterion(output[0], target_to_verb)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    loss += criterion(output[1], target_to_noun)
                    loss += criterion(output[2], target)
                    # loss += 0.01*div_loss

                else:
                    loss = criterion(output, target)
                loss /= args.update_freq

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            scaler.scale(loss).backward()
            
            # check parameters with no grad
            for n, p in model.named_parameters():
                if p.grad is None and p.requires_grad is True:
                    # p.requires_grad = False
                    print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model 
            
            if (data_iter + 1) % args.update_freq != 0:
                continue

            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                if args.clip_grad_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_value, norm_type=2.
                    )
                elif args.clip_grad_type == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip mode ({args.clip_grad_type})."
            # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            if isinstance(output, list):
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
                top1_noun.update(acc1_noun.item(), images.size(0))
                acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            else:
                output = torch.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                if args.dataset == 'ek100_cls':
                    vi = get_marginal_indexes(args.actions, 'verb')
                    ni = get_marginal_indexes(args.actions, 'noun')
                    verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                    noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                    target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                    acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                    acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                    top1_verb.update(acc1_verb.item(), images.size(0))
                    top1_noun.update(acc1_noun.item(), images.size(0))
                else:
                    top1_verb.update(0., images.size(0))
                    top1_noun.update(0., images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if optim_iter % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({
                        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                    })
                progress.display(optim_iter)
            
        
        progress.synchronize()
    
    if args.caption:
        return {
            # **{k: v.avg for k, v in metrics.items()},
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }
    else:
        return {
            'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }        

def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: '
    )

    # switch to eval mode
    model.eval()
    if args.use_half:
        model.half()

    all_outputs = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if isinstance(images, list):
                logit_allcrops = []
                for crop in images:
                    crop = crop.cuda(args.gpu, non_blocking=True)
                    if args.use_half:
                        crop = crop.half()
                    logit = model(crop, use_checkpoint=args.use_checkpoint)
                    logit_allcrops.append(logit)
                logit_allcrops = torch.stack(logit_allcrops, 0)
                logit = logit_allcrops.mean(0)
                logit = torch.softmax(logit, dim=1)
                target = target.cuda(args.gpu, non_blocking=True)
                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))
            else:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                if args.use_half:
                    images = images.half()

                logit = model(images, use_checkpoint=args.use_checkpoint)
                logit = torch.softmax(logit, dim=1)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

            all_outputs.append(logit)
            all_targets.append(target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    progress.synchronize()
    if args.dataset == 'ek100_cls':
        print('EK100 * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    else:
        print('EGTEA * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    cm = confusion_matrix(all_targets, all_outputs.argmax(axis=1))
    mean_acc, acc = get_mean_accuracy(cm)
    print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_acc, acc))

    if args.dataset == 'ek100_cls':
        vi = get_marginal_indexes(args.actions, 'verb')
        ni = get_marginal_indexes(args.actions, 'noun')
        verb_scores = marginalize(all_outputs, vi)
        noun_scores = marginalize(all_outputs, ni)
        target_to_verb = np.array([args.mapping_act2v[a] for a in all_targets.tolist()])
        target_to_noun = np.array([args.mapping_act2n[a] for a in all_targets.tolist()])
        cm = confusion_matrix(target_to_verb, verb_scores.argmax(axis=1))
        _, acc = get_mean_accuracy(cm)
        print('Verb Acc@1: {:.3f}'.format(acc))
        cm = confusion_matrix(target_to_noun, noun_scores.argmax(axis=1))
        _, acc = get_mean_accuracy(cm)
        print('Noun Acc@1: {:.3f}'.format(acc))
    return {'acc1': top1.avg, 'acc5': top5.avg, 'mean_acc': mean_acc}

def validate_multihead(val_loader, model,
                       v_classes_features_val, n_classes_features_val, a_classes_features_val,
                       args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, top1_verb, top1_noun],
        prefix='Test: '
    )

    # switch to eval mode
    model.eval()
    if args.use_half:
        model.half()

    all_verb_outputs = []
    all_noun_outputs = []
    all_action_outputs = []
    all_verb_targets = []
    all_noun_targets = []
    all_action_targets = []
    v_classes_features_val = v_classes_features_val.cuda(args.gpu, non_blocking=True)
    n_classes_features_val = n_classes_features_val.cuda(args.gpu, non_blocking=True)
    a_classes_features_val = a_classes_features_val.cuda(args.gpu, non_blocking=True)
    text_emb = [v_classes_features_val, v_classes_features_val, a_classes_features_val]
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if isinstance(images, torch.Tensor):
                images = [images, ]
            logit_verb_allcrops = []
            logit_noun_allcrops = []
            logit_action_allcrops = []
            for crop in images:
                crop = crop.cuda(args.gpu, non_blocking=True)
                if args.use_half:
                    crop = crop.half()

                if args.caption: # 实际上，第二个crop是无效输入，因为没有文本分支！
                    logit = model(crop, crop, use_checkpoint=args.use_checkpoint)
                elif args.use_text4vis:
                    logit = model(crop, text_emb, use_checkpoint=args.use_checkpoint)
                else:
                    logit = model(crop, use_checkpoint=args.use_checkpoint)

                logit_verb_allcrops.append(logit[0])
                logit_noun_allcrops.append(logit[1])
                logit_action_allcrops.append(logit[2]) 

            logit_verb_allcrops = torch.stack(logit_verb_allcrops, 0)
            logit_noun_allcrops = torch.stack(logit_noun_allcrops, 0)
            logit_action_allcrops = torch.stack(logit_action_allcrops, 0)

            logit_verb = logit_verb_allcrops.mean(0)
            logit_noun = logit_noun_allcrops.mean(0)
            logit_action = logit_action_allcrops.mean(0)

            logit_noun = torch.softmax(logit_noun, dim=1)
            logit_verb = torch.softmax(logit_verb, dim=1)
            logit_action = torch.softmax(logit_action, dim=1)

            target = target.cuda(args.gpu, non_blocking=True)
            target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)

            acc1, acc5 = accuracy(logit_action, target, topk=(1, 5))
            acc1_verb, _ = accuracy(logit_verb, target_to_verb, topk=(1, 5))
            acc1_noun, _ = accuracy(logit_noun, target_to_noun, topk=(1, 5))
            top1.update(acc1.item(), target.size(0))
            top5.update(acc5.item(), target.size(0))
            top1_verb.update(acc1_verb.item(), target_to_verb.size(0))
            top1_noun.update(acc1_noun.item(), target_to_noun.size(0))

            all_verb_outputs.append(logit_verb)
            all_noun_outputs.append(logit_noun)
            all_action_outputs.append(logit_action)
            all_verb_targets.append(target_to_verb)
            all_noun_targets.append(target_to_noun)
            all_action_targets.append(target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    progress.synchronize()
    print('EK100 * Verb Acc@1 {top1.avg:.3f}'.format(top1=top1_verb))
    print('EK100 * Noun Acc@1 {top1.avg:.3f}'.format(top1=top1_noun))
    print('EK100 * Action Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return {'acc1': top1.avg, 'acc5': top5.avg, 'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
