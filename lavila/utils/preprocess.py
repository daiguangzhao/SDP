# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv

from lavila.models.tokenizer import MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer
import random

def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        # v_list, n_list = [], []
        mapping_vn2narration = {}
        # mapping_v2narration, mapping_n2narration = {}, {}

        for f in [
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)

                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)

        vn_list = sorted(vn_list) # 按照动词的起始顺序对v:n的结果进行排序，输出的结果形式为：['0:0', '0:1', '0:10', '0:100', '0:101',...
        print('# of action= {}'.format(len(vn_list))) # 3806个v&n的组合，即action=3806个动名词
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)} # 将v&n的组合再次进行映射成action的索引，这样的索引赋予了action一个从0-3807个label，方便计算，它的结果形式为：mapping_vn2act: {'0:0'（原有动名词标签）: 0（映射后的标签）, '0:1': 1, '0:10': 2, '0:100': 3,... 
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))] # 结果的形式是字符串列表，注意这是字符串列表！！！

        print(labels[:5]) # 是每大类action label下子集列表：[['remove the tap on top plate on the pan', 'grab tap', 'get some tap water'], ...
        
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act # label的整数型标签



def generate_tokenizer(model):
    if model.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif model.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif model.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    elif model.endswith('GPT2'):
        tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)
    elif model.endswith('GPT2_MEDIUM'):
        tokenizer = MyGPT2Tokenizer('gpt2-medium', add_bos=True)
    elif model.endswith('GPT2_LARGE'):
        tokenizer = MyGPT2Tokenizer('gpt2-large', add_bos=True)
    elif model.endswith('GPT2_XL'):
        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(model))
        tokenizer = SimpleTokenizer()
    return tokenizer
