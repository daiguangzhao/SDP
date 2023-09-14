#!/bin/bash
killall python
source /root/anaconda3/bin/activate /root/anaconda3/envs/SDP
# 不要用sh命令 直接在终端复制下面一行
torchrun --nproc_per_node=1 main_finetune_classification.py --dataset egtea --metadata-train datasets/EGTEA/train_split1.txt --metadata-val datasets/EGTEA/test_split1.txt --root datasets/EGTEA/cropped_clips/ --output-dir output --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth --num-classes 106 --use-sgd --wd 4e-5 --resume checkpoint/experiments/lavila_ft/egtea_lavila_slowfastformer_final_6gpu_a6000_batch28/checkpoint_best.pt --evaluate --num-crops 3 --num-clips 10 --batch-size 64 --use-half
