#!/bin/bash
killall python
source /root/anaconda3/bin/activate /root/anaconda3/envs/SDP
python run_with_submitit_finetune_classification.py \
    --dataset egtea \
    --metadata-train datasets/EGTEA/train_split1.txt \
    --metadata-val datasets/EGTEA/test_split1.txt \
    --root datasets/EGTEA/cropped_clips/ \
    --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
    --num-classes 106 \
    --use-sgd \
    --wd 4e-5 \
    --ngpus 3 \
    --batch-size 12 \
    --use-checkpoint \
    --node 1 \
    # --find-unused-parameters

torchrun --nproc_per_node=1 main_finetune_classification.py --dataset egtea --metadata-train datasets/EGTEA/train_split1.txt --metadata-val datasets/EGTEA/test_split1.txt --root datasets/EGTEA/cropped_clips/ --output-dir output --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth --num-classes 106 --use-sgd --wd 4e-5 --resume checkpoint/experiments/lavila_ft/1138/checkpoint_best.pt --num-crops 3 --num-clips 10 --batch-size 64 --use-half
