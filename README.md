# SDP
## Requirements  
#### Environment
conda create --name SDP python=3.8 -y  
conda activate SDP  
pip install  
timm==0.5.4  
torch==1.10.1  
torchvision==0.11.2  
decord==0.6.0  
einops==0.4.1  
pandas==1.4.2  
pytorchvideo==0.1.5  
transformers==4.21  
ftfy==4.4.3  
spacy==3.4.1  
scikit-learn==1.1.1  
git+https://github.com/Maluuba/nlg-eval.git@master
#### Datasets  
Please follow the instructions in <a href="https://github.com/facebookresearch/LaViLa/blob/main/datasets/README.md" target="_blank">lavila/datasets/README.md</a>  
## Train  
#### EK100 (train & test)
python run_with_submitit_finetune_classification.py \
    --pretrain-model <a href="https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth" target="_blank">clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth</a> \
    --use-vn-classifier \
    --num-classes 97 300 3806 \
    --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
    --use-checkpoint \
    --ngpus 6 \
    --batch-size 28 \
    --node 1 \
    --resume checkpoint/experiments/lavila_ft/1_85/checkpoint_0035.pt \
    --evaluate \
#### EGTEA  
python run_with_submitit_finetune_classification.py \
    --dataset egtea \
    --metadata-train datasets/EGTEA/train_split1.txt \
    --metadata-val datasets/EGTEA/test_split1.txt \
    --root datasets/EGTEA/cropped_clips/ \
    --pretrain-model <a href="https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth" target="_blank">clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth</a> \
    --num-classes 106 \
    --use-sgd \
    --wd 4e-5 \
    --ngpus 6 \
    --batch-size 28 \
    --use-checkpoint \
    --node 1 \
## Test  
torchrun --nproc_per_node=1 main_finetune_classification.py  \
--dataset egtea  \
--metadata-train datasets/EGTEA/train_split1.txt  \
--metadata-val datasets/EGTEA/test_split1.txt  \
--root datasets/EGTEA/cropped_clips/  \
--output-dir output  \
--pretrain-model pretrained_weight/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth  \
--num-classes 106  \
--use-sgd  \
--wd 4e-5  \
--resume checkpoint/experiments/lavila_ft/1138/checkpoint_best.pt  \
--num-crops 3  \
--num-clips 10  \
--batch-size 64  \
--use-half

