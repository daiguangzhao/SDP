#!/bin/bash
killall python
source /root/anaconda3/bin/activate /root/anaconda3/envs/SDP
# python run_with_submitit_finetune_classification.py \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 6 \
#     --batch-size 12 \
#     --node 1 \
#     --resume checkpoint/experiments/lavila_ft/1_85/checkpoint_0035.pt \
#     --evaluate \
#     # --find-unused-parameters

# A10将batch设置为11

# remove llm pretraining
# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 6 \
#     --batch-size 12 \
#     --node 1 \
#     --resume checkpoint/experiments/lavila_ft/87/checkpoint_0040.pt \
#     # --evaluate
#     # --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     # --resume checkpoint/experiments/lavila_ft/1_85/checkpoint.pt \



# freeze timesformer space weights
# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 4 \
#     --batch-size 12 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --caption True \
#     --hard-prompt-as-actionclip True \
#     --use-few-shot \

# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd \
#     --wd 4e-5 \
#     --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 5 \
#     --batch-size 10 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --hard-prompt-as-actionclip True \
#     --use-few-shot \
#     --use_vanila_clip \
#     # --caption \

# mv checkpoint/experiments/lavila_ft/86 checkpoint/experiments/lavila_ft/few_shot_vanila_clip_only_visual

# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd \
#     --wd 4e-5 \
#     --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 5 \
#     --batch-size 8 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --hard-prompt-as-actionclip True \
#     --use-few-shot \
#     --use_vanila_clip \
#     --caption \

# mv checkpoint/experiments/lavila_ft/86 checkpoint/experiments/lavila_ft/few_shot_vanila_clip_visual_and_text_encoder

killall python
python run_with_submitit_finetune_classification.py \
    --use-vn-classifier \
    --num-classes 97 300 3806 \
    --use-sgd \
    --wd 4e-5 \
    --lr-multiplier-on-backbone 0.1 \
    --use-checkpoint \
    --ngpus 8 \
    --batch-size 12 \
    --node 1 \
    --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
    --hard-prompt-as-actionclip True \
    --use-few-shot \
    --use_text4vis
    # --caption \
# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 4 \
#     --batch-size 12 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --caption True \
#     --hard-prompt-as-actionclip True \
#     --use-few-shot \
#     --shot 5

# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 4 \
#     --batch-size 12 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
#     --hard-prompt-as-actionclip True \
#     --use-few-shot \
#     --shot 5

# mv checkpoint/experiments/lavila_ft/86 checkpoint/experiments/lavila_ft/86_1

# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 6 \
#     --batch-size 12 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \


    # --find-unused-parameters
    # --contrastive-use-vissl \
    # --evaluate \
    # --resume checkpoint/experiments/lavila_ft/675/checkpoint_0005.pt
    # --find-unused-parameters
    # --pretrain-model pretrained_weight/clip_openai_timesformer_base.baseline.ep_0003.pth \
    # --freeze-temperature \
    # --evaluate \
    # --resume checkpoint/experiments/lavila_ft/86/checkpoint_0005.pt \


# killall python
# python run_with_submitit_finetune_classification.py \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 4 \
#     --batch-size 12 \
#     --node 1 \
#     --pretrain-model pretrained_weight/clip_openai_timesformer_base.baseline.ep_0003.pth \
#     --resume checkpoint/experiments/lavila_ft/87/checkpoint_0020.pt \



# killall python
# python run_with_submitit_finetune_classification.py \
#     --pretrain-model pretrained_weight/TimeSformer_divST_8x32_224_K400.pyth \
#     --use-vn-classifier \
#     --num-classes 97 300 3806 \
#     --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
#     --use-checkpoint \
#     --ngpus 2 \
#     --batch-size 11 \
#     --node 1 \
#     # --resume checkpoint/experiments/lavila_ft/1_85/checkpoint.pt \

# zero-shot 的代码
# python eval_zeroshot.py --dataset ek100_cls --metadata-val datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv  --resume $PATH 
