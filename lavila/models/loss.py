# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

from .distributed_utils import gather_from_all


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    # Adapted from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class CLIPLoss(nn.Module):

    def __init__(
            self,
            use_vissl=False,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.use_vissl = use_vissl
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, outputs):
        image_features = outputs['image_embed']
        text_features = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        device = image_features.device
        if self.world_size > 1:
            if self.use_vissl:
                all_image_features = gather_from_all(image_features)
                all_text_features = gather_from_all(text_features)
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            else:
                all_image_features, all_text_features = gather_features(
                    image_features, text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size)

                if self.local_loss:
                    logits_per_image = logit_scale * image_features @ all_text_features.T
                    logits_per_text = logit_scale * text_features @ all_image_features.T
                else:
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / logits_per_image.size(0)

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}


# # 自己改的            
# class CLIPLoss(nn.Module):

#     def __init__(
#             self,
#             use_vissl=False,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#     ):
#         super().__init__()
#         self.use_vissl = use_vissl
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size

#         # cache state
#         self.v_prev_num_logits = 0
#         self.n_prev_num_logits = 0
#         # self.a_prev_num_logits = 0

#         self.v_labels = {}
#         self.n_labels = {}
#         # self.a_labels = {}

#     def forward(self, outputs):
#         image_features = outputs['image_embed']
#         # v_image_features = outputs['v_embed']
#         # n_image_features = outputs['n_embed']

#         v_text_features = outputs['v_text_embed']
#         n_text_features = outputs['n_text_embed']
#         # a_text_features = outputs['a_text_embed']
#         logit_scale = outputs['logit_scale']
#         # logit_scale_ = outputs['logit_scale_']
#         # device = image_features.device
        
#         # device = v_image_features.device

#         # if self.world_size > 1:
#         #     if self.use_vissl:
#         #         print("33333333") # 这是真的！
#         #         v_all_image_features = gather_from_all(v_image_features)
#         #         n_all_image_features = gather_from_all(n_image_features)
#         #         v_all_text_features = gather_from_all(v_text_features)
#         #         n_all_text_features = gather_from_all(n_text_features)
#         #         # a_all_text_features = gather_from_all(a_text_features)

#         #         v_logits_per_image = logit_scale * v_all_image_features @ v_all_text_features.T
#         #         n_logits_per_image = logit_scale * n_all_image_features @ n_all_text_features.T
#         #         # a_logits_per_image = logit_scale * all_image_features @ a_all_text_features.T

#         #         v_logits_per_text = v_logits_per_image.T
#         #         n_logits_per_text = n_logits_per_image.T
#         #         # a_logits_per_text = a_logits_per_image.T
#         #     else:
#         #         all_image_features, all_text_features = gather_features(
#         #             image_features, text_features,
#         #             self.local_loss, self.gather_with_grad, self.rank, self.world_size)

#         #         if self.local_loss:
#         #             logits_per_image = logit_scale * image_features @ all_text_features.T
#         #             logits_per_text = logit_scale * text_features @ all_image_features.T
#         #         else:
#         #             logits_per_image = logit_scale * all_image_features @ all_text_features.T
#         #             logits_per_text = logits_per_image.T
#         # else:
#         #     print("111111")
#         #     # logits_per_image = logit_scale * image_features @ text_features.T
#         #     # logits_per_text = logit_scale * text_features @ image_features.T

#         #     v_logits_per_image = logit_scale * image_features @ v_text_features.T
#         #     n_logits_per_image = logit_scale * image_features @ n_text_features.T

#         #     v_logits_per_text  = logit_scale * v_text_features @ image_features.T
#         #     n_logits_per_text  = logit_scale * n_text_features @ image_features.T
#         #     # a_logits_per_image = logit_scale * all_image_features @ a_all_text_features.T

#         #     # v_logits_per_text = v_logits_per_image.T
#         #     # n_logits_per_text = n_logits_per_image.T

#         # v_image_features = v_image_features / v_image_features.norm(dim=-1, keepdim=True)
#         # n_image_features = n_image_features / n_image_features.norm(dim=-1, keepdim=True)

#         # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         v_text_features = v_text_features / v_text_features.norm(dim=-1, keepdim=True)
#         n_text_features = n_text_features / n_text_features.norm(dim=-1, keepdim=True)

#         v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_features, v_text_features)
#         n_logits_per_image = torch.einsum("bd, td->bt", logit_scale * image_features, n_text_features)

#         # v_logits_per_image = torch.einsum("bd, td->bt", logit_scale * v_image_features, v_text_features)
#         # n_logits_per_image = torch.einsum("bd, td->bt", logit_scale_ * n_image_features, n_text_features)
#         # v_logits_per_image = logit_scale * v_image_features @ v_text_features.T
#         # n_logits_per_image = logit_scale * n_image_features @ n_text_features.T

#         # v_logits_per_text = torch.einsum("bd, td->bt", logit_scale * v_text_features, v_image_features)
#         # n_logits_per_text = torch.einsum("bd, td->bt", logit_scale_ * n_text_features, n_image_features)
#         v_logits_per_text  = v_logits_per_image.t()
#         n_logits_per_text  = n_logits_per_image.t()
        
#         # calculated ground-truth and cache if enabled
#         v_num_logits = v_logits_per_image.shape[0]
#         n_num_logits = n_logits_per_image.shape[0]

#         # print("v_logits_per_image", v_logits_per_image)
#         # print("n_logits_per_image", n_logits_per_image)
#         # print("v_logits_per_text",  v_logits_per_text)
#         # print("n_logits_per_text",  n_logits_per_text)

#         # a_num_logits = a_logits_per_image.shape[0]
#         # if self.v_prev_num_logits != v_num_logits or device not in self.labels:
#         #     print("222222") # 这是真的!!!

#         v_labels = torch.arange(v_num_logits, device="cuda").long()
#         n_labels = torch.arange(n_num_logits, device="cuda").long()
#             # v_labels = torch.arange(v_num_logits, device="cuda", dtype=torch.long)
#             # n_labels = torch.arange(n_num_logits, device="cuda", dtype=torch.long)
#             # a_labels = torch.arange(a_num_logits, device=device, dtype=torch.long)

#         #     if self.world_size > 1 and self.local_loss:
#         #         labels = labels + num_logits * self.rank
#         #         print("444444")
#         #     if self.cache_labels: # 这是真的！！！！！
#         #         self.v_labels[device] = v_labels
#         #         self.v_prev_num_logits = v_num_logits
#         #         self.n_labels[device] = n_labels
#         #         self.n_prev_num_logits = n_num_logits
#         #         # self.a_labels[device] = a_labels
#         #         # self.a_prev_num_logits = a_num_logits
#         #         # print("555555555") 
#         #         # v_labels = self.v_labels[device]
#         #         # v_num_logits = self.v_prev_num_logits
#         #         # n_labels = self.n_labels[device]
#         #         # n_num_logits = self.n_prev_num_logits                               
#         # else:
#         #     # labels = self.labels[device]
#         #     # self.v_labels[device] = v_labels
#         #     # self.v_prev_num_logits = v_num_logits
#         #     # self.n_labels[device] = n_labels
#         #     # self.n_prev_num_logits = n_num_logits
#         #     v_labels = self.v_labels[device]
#         #     v_num_logits = self.v_prev_num_logits
#         #     n_labels = self.n_labels[device]
#         #     n_num_logits = self.n_prev_num_logits
#         #     print("6666666666")

#         loss = (
#             F.cross_entropy(v_logits_per_image, v_labels) + 
#             F.cross_entropy(v_logits_per_text, v_labels)  +
#             F.cross_entropy(n_logits_per_image, n_labels) +
#             F.cross_entropy(n_logits_per_text, n_labels)  
#             # F.cross_entropy(a_logits_per_image, a_labels) +
#             # F.cross_entropy(a_logits_per_text, a_labels)
#             ) / 4

#         # compute accuracy
#         with torch.no_grad():
#             v_pred = torch.argmax(v_logits_per_image, dim=-1)
#             v_correct = v_pred.eq(v_labels).sum()
#             v_acc = 100 * v_correct / v_logits_per_image.size(0)

#             n_pred = torch.argmax(n_logits_per_image, dim=-1)
#             n_correct = n_pred.eq(n_labels).sum()
#             n_acc = 100 * n_correct / n_logits_per_image.size(0)

#             # a_pred = torch.argmax(a_logits_per_image, dim=-1)
#             # a_correct = a_pred.eq(a_labels).sum()
#             # a_acc = 100 * a_correct / a_logits_per_image.size(0)
#         # return {'loss': loss, 'clip_loss': loss, 'clip_acc': (v_acc + n_acc + a_acc)/3}
#         # 这是真正的错误loss

#         return {'loss': loss, 'clip_loss': loss, 'clip_acc': (v_acc + n_acc)/2}



class SSLCLIPLoss(nn.Module):

    def __init__(
            self,
            use_vissl=False,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            scale_init=0.08,
            freeze_scale=False,
    ):
        super().__init__()
        self.use_vissl = use_vissl
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.logit_scale_pseudo = nn.Parameter(torch.ones([]) * np.log(1 / scale_init))
        if freeze_scale:
            self.logit_scale_pseudo.requires_grad = False

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, outputs, gt_indicators):
        image_features = outputs['image_embed']
        text_features = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        logit_scale_pseudo = self.logit_scale_pseudo.exp()
        device = image_features.device
        if self.world_size > 1:
            if self.use_vissl:
                all_image_features = gather_from_all(image_features)
                all_text_features = gather_from_all(text_features)
                all_gt_indicators = gather_from_all(gt_indicators)
                num = all_gt_indicators.shape[0]
                mask = all_gt_indicators.repeat(num, 1) + all_gt_indicators.repeat(num, 1).T
                logit_scale_mat = torch.ones((num, num), device=device)
                logit_scale_mat[mask == 0] = logit_scale_pseudo
                logit_scale_mat[mask == 1] = torch.sqrt(logit_scale_pseudo * logit_scale)
                logit_scale_mat[mask == 2] = logit_scale
                logits_per_image = logit_scale_mat * (all_image_features @ all_text_features.T)
                logits_per_text = logits_per_image.T
            else:
                raise NotImplementedError
        else:
            all_gt_indicators = gt_indicators
            num = gt_indicators.shape[0]
            mask = gt_indicators.repeat(num, 1) + gt_indicators.repeat(num, 1).T
            logit_scale_mat = torch.ones((num, num), device=device)
            logit_scale_mat[mask == 0] = logit_scale_pseudo
            logit_scale_mat[mask == 1] = torch.sqrt(logit_scale_pseudo * logit_scale)
            logit_scale_mat[mask == 2] = logit_scale
            logits_per_image = logit_scale_mat * (image_features @ text_features.T)
            logits_per_text = logit_scale_mat * (text_features @ image_features.T)

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / logits_per_image.size(0)
            pred_gt = pred[all_gt_indicators == 1]
            labels_gt = labels[all_gt_indicators == 1]
            pred_pseudo = pred[all_gt_indicators == 0]
            labels_pseudo = labels[all_gt_indicators == 0]
            num_gt = pred_gt.shape[0]
            num_pseudo = pred_pseudo.shape[0]
            correct_gt = pred_gt.eq(labels_gt).sum()
            correct_pseudo = pred_pseudo.eq(labels_pseudo).sum()
            acc_gt = 100 * correct_gt / num_gt
            acc_pseudo = 100 * correct_pseudo / num_pseudo

        return {
            'loss': loss, 'clip_loss': loss, 'num_gt': torch.tensor([num_gt]), 'num_pseudo': torch.tensor([num_pseudo]),
            'clip_acc': acc, 'clip_acc_gt': acc_gt, 'clip_acc_pseudo': acc_pseudo
        }


class CaptionLoss(nn.Module):
    def __init__(self, pad_id=0, tokenizer=None):
        super().__init__()
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def forward(self, outputs):
        logits = outputs['text_tokens_logits']
        labels = outputs['labels']
        # loss = F.cross_entropy(logits, labels, ignore_index=self.pad_id)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_id, reduction='none')

        # compute accuracy
        with torch.no_grad():
            correct = 0.
            total = 0.
            ppls = []
            for i in range(logits.size(0)):
                pred = torch.argmax(logits[i], dim=0)
                nopad = labels[i].ne(self.pad_id)
                correct += (pred.eq(labels[i]) & nopad).sum()
                total += nopad.sum()
                ppl = torch.exp(loss[i].sum() / nopad.sum())
                ppls.append(ppl)
                # TODO: for debug only
                # sep_pos = labels[i].tolist().index(self.tokenizer.tokenizer.sep_token_id)
                # if self.tokenizer is not None:
                #     print('{} {} {}'.format(
                #         i, self.tokenizer.tokenizer.convert_ids_to_tokens(pred[:sep_pos]),
                #         self.tokenizer.tokenizer.convert_ids_to_tokens(labels[i, :sep_pos]),
                #     ))
            acc = 100 * correct / (total + 1e-8)
        return {'loss': loss.mean(), 'caption_loss': loss.mean(), 'caption_acc': acc, 'ppl': torch.tensor(ppls).mean()}


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, outputs, weight=None):
        image_features = outputs['image_embed']
        text_features = outputs['text_embed']

        all_image_features = gather_from_all(image_features)
        all_text_features = gather_from_all(text_features)
        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return {
            'loss': max_margin.mean(),
            'max_margin_loss': max_margin.mean()
        }


class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, outputs, weight=None):
        image_features = outputs['image_embed']
        text_features = outputs['text_embed']

        all_image_features = gather_from_all(image_features)
        all_text_features = gather_from_all(text_features)
        all_weights = gather_from_all(weight)
        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = all_weights.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(w1_ * self.margin - (x1_ - x2_))

        return {
            'loss': max_margin.mean(),
            'max_margin_loss': max_margin.mean()
        }
