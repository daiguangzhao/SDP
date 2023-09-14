# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# from visualizer_packge.visualizer import get_local
# get_local.activate()

# # import torchvision.transforms as T
# from timm.models.vision_transformer import vit_small_patch16_224
# import json
# from PIL import Image, ImageDraw
# import numpy as np
# import matplotlib.pyplot as plt



import argparse
from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  #（代表仅使用第0，1号GPU）


import cv2

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
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for val')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    # Model
    parser.add_argument('--pretrain-model', default='pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth', type=str, help='path to pretrain model')
    parser.add_argument('--resume', default='checkpoint/experiments/lavila_ft/EK100_Part_Token_Attention_V13_Cross_SlowFast_T_Tcross/checkpoint_best.pt', type=str, help='path to resume from')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
    parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio for the last linear layer')
    parser.add_argument('--num-classes', default=3806, nargs='+', type=int, help='number of classes for the last linear layer')
    parser.add_argument('--use-vn-classifier', action='store_true')
    parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
    # Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--use-sgd', action='store_true')
    parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
    parser.add_argument('--lr', default=3e-3, type=float)
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
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
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
    parser.add_argument('--sfa_layer', default=1, type=int, help='Number OF SFA_lAYER.')
    parser.add_argument('--d2_sfc', action='store_const', default=True, const=False, help='weither use 2d sfc.')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)

    global best_acc1
    random_seed(args.seed, dist_utils.get_rank())

    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception('no checkpoint found')
    # 加载预训练的权重
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if args.use_vn_classifier:
        assert args.dataset == 'ek100_cls' and len(args.num_classes) == 3

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model)) # OPENAI_CLIP_TIMESFORMER
    model = getattr(models, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        sfa_layer=args.sfa_layer,
        d2_sfc=args.d2_sfc,
    )
    print("model", model)
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))

    if args.use_vn_classifier:
        model = models.VideoClassifierMultiHead(
            model.visual,
            dropout=args.dropout_ratio,
            num_classes_list=args.num_classes
        )
    else:
        assert len(args.num_classes) == 1
        model = models.VideoClassifier(
            model.visual,
            dropout=args.dropout_ratio,
            num_classes=args.num_classes[0]
        )
    print(" model.visual", model) # 这里是将model.visual部分打印出来，并不是完整的模型
    
    
    
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.gpu], bucket_cap_mb=200,
    #         find_unused_parameters=args.find_unused_parameters
    #     )

    # model.cuda(args.gpu)

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


    #     print("ddp后的model", model) # 这里是将model.visual部分打印出来，并不是完整的模型
    # exit(0)


    p_wd, p_non_wd = [], []
    p_head_wd, p_head_non_wd = [], []
    for n, p in model.named_parameters():
        if 'fc_cls' in n:
            if 'bias' in n:
                p_head_non_wd.append(p)
            else:
                p_head_wd.append(p)
        elif not p.requires_grad:
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
    
    # 这里需要手动设定是否续传
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
            # optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            # scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            # best_acc1 = checkpoint['best_acc1']
            # print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})"
            #       .format(args.resume, epoch, best_acc1))
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
    if old_args.model.endswith('DISTILBERT_BASE'):
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
        # 这个是真正的选择
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(old_args.model))
        tokenizer = SimpleTokenizer()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    crop_size = 224 if '336PX' not in old_args.model else 336
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if 'OPENAI' in old_args.model:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
    else:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    train_transform = transforms.Compose(transforms_list)

    val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
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
        val_dataset, batch_size=int(args.batch_size), shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    if args.evaluate:
        # epic100分类
        if args.use_vn_classifier:
            print("=> beginning vislization")
            val_stats = validate_multihead(val_loader, model, args)
        else:
            val_stats = validate(val_loader, model, args)
        return

    # if args.fix_lr:
    #     lr_schedule = None
    # else:
    #     lr_schedule = cosine_scheduler(
    #         args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
    #         warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
    #     )

    # if dist_utils.is_main_process() and args.wandb:
    #     wandb_id = os.path.split(args.output_dir)[-1]
    #     wandb.init(project='LaViLa', id=wandb_id, config=args, resume='allow')

    # print(args)

    # best_metric = 0.
    # print("=> beginning training")
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)

    #     if args.dataset == 'ek100_cls':
    #         train_stats = train_epic(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
    #     else:
    #         train_stats = train_egtea(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
    #     is_epoch = ((epoch + 1) % args.save_freq) == 0

    #     print('=> saving checkpoint')
    #     dist_utils.save_on_master({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'scaler': scaler.state_dict(),
    #         'best_acc1': 0,
    #         'args': args,
    #     }, False, args.output_dir, is_epoch=is_epoch)

    #     if ((epoch + 1) % args.eval_freq) == 0:
    #         if args.use_vn_classifier:
    #             val_stats = validate_multihead(val_loader, model, args)
    #         else:
    #             val_stats = validate(val_loader, model, args)
    #         if val_stats['acc1'] > best_metric:
    #             is_best = True
    #             best_metric = val_stats['acc1']
    #         else:
    #             is_best = False

    #         print('=> saving checkpoint')
    #         dist_utils.save_on_master({
    #             'epoch': epoch + 1,
    #             'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'scaler': scaler.state_dict(),
    #             'best_acc1': best_metric,
    #             'args': args,
    #         }, is_best, args.output_dir, is_epoch=is_epoch)

    #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                      **{f'test_{k}': v for k, v in val_stats.items()},
    #                      'epoch': epoch}

    #         if dist_utils.is_main_process():
    #             if args.wandb:
    #                 wandb.log(log_stats)
    #             with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
    #                 f.write(json.dumps(log_stats) + '\n')


def train_epic(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top5_noun = AverageMeter('Noun Acc@5', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    top5_verb = AverageMeter('Verb Acc@5', ':6.2f')
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, losses, top1, top5, top1_noun, top5_noun, top1_verb, top5_verb],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
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
            acc1_verb, acc5_verb, = accuracy(output[0], target_to_verb, topk=(1, 5))
            top1_verb.update(acc1_verb.item(), images.size(0))
            top5_verb.update(acc5_verb.item(), images.size(0))
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            acc1_noun, acc5_noun = accuracy(output[1], target_to_noun, topk=(1, 5))
            top1_noun.update(acc1_noun.item(), images.size(0))
            top5_noun.update(acc5_noun.item(), images.size(0))
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
                    'acc1_verb': top1_verb.avg, 'acc5_verb': top5_verb.avg, 'acc1_noun': top1_noun.avg, 'acc5_noun': top5_noun.avg,
                })
            progress.display(optim_iter)
    progress.synchronize()
    return {
        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
        'acc1_verb': top1_verb.avg, 'acc5_verb': top5_verb.avg, 'acc1_noun': top1_noun.avg, 'acc5_noun': top5_noun.avg,
        'lr': optimizer.param_groups[0]['lr'],
    }


def train_egtea(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
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
            else:
                loss = criterion(output, target)
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

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


@torch.no_grad()
def validate_multihead(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top5_verb = AverageMeter('Verb Acc@5', ':6.2f')
    top5_noun = AverageMeter('Noun Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, top1_verb, top5_verb, top1_noun, top5_noun],
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
    
   
    # with torch.no_grad():
    end = time.time() 
    for i, (images, target) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if isinstance(images, torch.Tensor):
            images = [images, ]
        logit_verb_allcrops = []
        logit_noun_allcrops = []
        logit_action_allcrops = []

        attn = []
        hooks = [
        # model.module.visual.part_token_attention.PartLayers[-1].multihead_attn.register_forward_hook(
        #     lambda self, input, output: attn.append(output[1]))
        #     ] 

        model.module.visual.part_token_attention.part_slowatt_layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: attn.append(output[1]))
            ] 

        for crop in images:
            crop = crop.cuda(args.gpu, non_blocking=True)                   
            if args.use_half:
                crop = crop.half()
            # 获得logits
            logit = model(crop, use_checkpoint=args.use_checkpoint)
            logit_verb_allcrops.append(logit[0])
            logit_noun_allcrops.append(logit[1])
            logit_action_allcrops.append(logit[2])
        
        for hook in hooks:
            hook.remove()
            
        ####-------取出attention map------########
        print(len(attn)) # 1
        attn_map = attn[0] # torch.Size([16, 9, 196])
        print('attn_map:\t', attn_map.size()) # [10, 16, 32]
        # exit(0)
        attn_map = attn_map[:,:,:].permute(0,2,1)
        
        ('attn_map1:\t', attn_map.size())
        attn_map = attn_map.clone().cpu()
        # attn_map = attn_map.mean(dim=1).reshape(16,14,14).detach().numpy()
        # attn_map = attn_map.mean(dim=1).reshape(16,3,3).detach().numpy()
        attn_map = attn_map.mean(dim=1).reshape(10,4,4)
        attn_map = nn.functional.interpolate(attn_map, size=(14, 14),
                                          mode='bilinear', align_corners=True)
        # print('attn_map2:\t', attn_map.size())
        # print('attn_map:\t', attn_map.size())
        # attn_map = attn_map.squeeze(1)
        # print('attn_map1:\t', attn_map.size())
        # attn_map = attn_map[:,1:].reshape(12,4,196)
        # print('attn_map2:\t', attn_map.size())
        # attn_map = attn_map[:,:,:196]
        # print('attn_map3:\t', attn_map.size())
        
        # ####
        # attn_map = attn_map.clone().cpu()
        # attn_map = attn_map.mean(dim=0).reshape(4,14,14).numpy()


        ### save image
        # print('inputs:\t', images[0].size())
        frames_vis = images[0].clone().cpu()
        frames_vis = frames_vis.squeeze(0)
        frames_vis = frames_vis.permute(1, 2, 3, 0)
        frames_vis = revert_tensor_normalize(
                            frames_vis,
                            mean=[108.3272985/256, 116.7460125/256, 104.09373615000001/256], 
                            std=[68.5005327/256, 66.6321579/256, 70.32316305/256]
                            # mean=[0.485, 0.456, 0.406], 
                            # std=[0.229, 0.224, 0.225]
                            # mean=[0.5, 0.5, 0.5], 
                            # std=[0.5, 0.5, 0.5]                            
                            )
        frames_vis = frames_vis.permute(0, 3, 1, 2)
        frames_vis = frames_vis[:,(0,1,2),:,:]
        # print(frames_vis.size())
        # 这里定义两个文件夹用来存放图片和attn img
        # imgs_path = '/opt/data/private/project/LaViLa-main/demo/{}/org_img'.format(str(i))
        # if os.path.exists(imgs_path):
        #     pass
        # else:
        #     os.makedirs(imgs_path)
        # attn_path = '/opt/data/private/project/LaViLa-main/demo/{}/attn_img'.format(str(i))
        # if os.path.exists(attn_path):
        #     pass
        # else:
        #     os.makedirs(attn_path)  
        # 这里定义两个文件夹用来存放图片
        
        # step1: 这里定义两个文件夹用来存放图像
        imgs_path = '/opt/data/private/project/LaViLa-main/demo1/{}/org_img'.format(str(i))
        if os.path.exists(imgs_path):
            pass
        else:
            os.makedirs(imgs_path)        
        # step1: 这里定义两个文件夹用来存放attn_map hot map
        attn_path = '/opt/data/private/project/LaViLa-main/demo1/{}/attn_img'.format(str(i))
        if os.path.exists(attn_path):
            pass
        else:
            os.makedirs(attn_path)         
        
        list_vis_imgs = frame_to_list_img(frames_vis)
        for k, img in enumerate(list_vis_imgs):
            # 保存结果
            img.save('/opt/data/private/project/LaViLa-main/demo1/{}/org_img/img_{}.jpg'.format(str(i), str(k)))
        # exit(0)
        for k, mask in enumerate(attn_map): 
            # step2: 保存结果           
            save_path = '/opt/data/private/project/LaViLa-main/demo1/{}/attn_img/attn_img_{}.jpg'.format(str(i),str(k))
            run_grid_attention_example(list_vis_imgs[k], save_path=save_path, attention_mask = mask) 
        #################################################            

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
        # print("the {} target is {}:".format(str(i),target))
        target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
        target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
        print("the {} target is {}:".format(str(i),target_to_verb))
        print("the {} target is {}:".format(str(i),target_to_noun))

        acc1, acc5 = accuracy(logit_action, target, topk=(1, 5))
        acc1_verb, acc5_verb = accuracy(logit_verb, target_to_verb, topk=(1, 5))
        acc1_noun, acc5_noun = accuracy(logit_noun, target_to_noun, topk=(1, 5))
        top1.update(acc1.item(), target.size(0))
        top5.update(acc5.item(), target.size(0))
        top1_verb.update(acc1_verb.item(), target_to_verb.size(0))
        top1_noun.update(acc1_noun.item(), target_to_noun.size(0))
        top5_verb.update(acc5_verb.item(), target_to_verb.size(0))
        top5_noun.update(acc5_noun.item(), target_to_noun.size(0))
        
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
    print('EK100 * Verb Acc@5 {top5.avg:.3f}'.format(top5=top5_verb))
    print('EK100 * Noun Acc@1 {top1.avg:.3f}'.format(top1=top1_noun))
    print('EK100 * Noun Acc@5 {top5.avg:.3f}'.format(top5=top5_noun))
    print('EK100 * Action Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return {'acc1': top1.avg, 'acc5': top5.avg, 'acc1_verb': top1_verb.avg, 'acc5_verb': top5_verb.avg, 'acc1_noun': top1_noun.avg, 'acc5_noun': top5_noun.avg}

def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor

def frame_to_list_img(frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    
def run_grid_attention_example(img=[], save_path="", attention_mask=None, quality=100):
    
    visualize_grid_attention_v2(img=img,
                                save_path=save_path,
                                attention_mask=attention_mask,
                                save_image=True,
                                save_original_image=True,
                                quality=quality)
    
def visualize_grid_attention_v2(img, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    # img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # # build save path
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        # img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        # print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=quality)

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    # print(rows)
    it = iter(to_shows)
    # print(it)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*16, cols))
    # for i in range(rows):
    #     for j in range(cols):
    #         try:
    #             image, title = next(it)
    #         except StopIteration:
    #             image = np.zeros_like(to_shows[0][0])
    #             title = 'pad'
    #         axs[i, j].imshow(image)
    #         axs[i, j].set_title(title)
    #         axs[i, j].set_yticks([])
    #         axs[i, j].set_xticks([])
    # for i in range(rows):
    for j in range(cols):
        try:
            image, title = next(it)
        except StopIteration:
            # image = np.zeros_like(to_shows[0][0])
            # title = 'pad'
            pass
        axs[j].imshow(image)
        axs[j].set_title(title)
        axs[j].set_yticks([])
        axs[j].set_xticks([])
    plt.show()
    save_path = "/opt/data/private/code/new_proj/TIP23/12.jpg"
    plt.savefig(save_path, dpi=200)

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
            to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    # print(len(to_shows))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
