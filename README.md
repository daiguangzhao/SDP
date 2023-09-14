# SDP
## Requirements  
### Environment
conda create --name SDP python=3.8 -y  
conda activate SDP  
pip install -r requirements.txt  
### Datasets  
Please follow the instructions in <a href="https://github.com/facebookresearch/LaViLa/blob/main/datasets/README.md" target="_blank">lavila/datasets/README.md</a>  
## Train  
### EK100 (train & test)
sh train.sh 
### EGTEA  
sh train_egtea.sh
## Test  
sh test_egtea.sh

