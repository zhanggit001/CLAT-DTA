import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
import re
import csv
import esm
import torch
from rdkit import Chem

# 生成蛋白质预训练表示:davis.npz,kiba.npz
def generate_protein_pretraining_representation(dataset, prots):
    data_dict = {}  # 数据字典
    prots_tuple = [(str(i), prots[i][:1022]) for i in range(len(prots))]  # 创建蛋白质元组
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()  # 加载transformer模型和字母表
    batch_converter = alphabet.get_batch_converter()  # 获取批处理转换器
    i = 0
    batch = 1

    while (batch*i) < len(prots):  # 循环处理蛋白质
        print('converting protein batch: '+ str(i))  # 打印转换蛋白质批次信息
        if (i + batch) < len(prots):  # 判断是否有下一个批次
            pt = prots_tuple[batch*i:batch*(i+1)]  # 获取当前批次的蛋白质元组
        else:
            pt = prots_tuple[batch*i:]  # 获取剩余的蛋白质元组

        batch_labels, batch_strs, batch_tokens = batch_converter(pt)  # 批量转换蛋白质
        #model = model.cuda()
        #batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)  # 获取结果
        token_representations = results["representations"][33].numpy()  # 提取表示
        data_dict[i] = token_representations  # 存储表示
        i += 1
    np.savez(dataset + '.npz', dict=data_dict)  # 保存数据字典为npz文件

# 数据集处理
datasets = ['davis','kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'  # 数据集路径
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # 加载训练集折叠
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # 加载验证集折叠
    folds = train_fold + [valid_fold]  # 合并折叠
    valid_ids = [5,4,3,2,1]  # 验证集ID
    valid_folds = [folds[vid] for vid in valid_ids]  # 验证集折叠
    train_folds = []  # 训练集折叠
    for i in range(5):
        temp = []
        for j in range(6):
            if j != valid_ids[i]:
                temp += folds[j]
        train_folds.append(temp)

    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # 加载药物分子
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # 加载蛋白质
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')  # 加载亲和力
    drugs = []  # 药物列表
    prots = []  # 蛋白质列表
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)  # 将药物分子转换为SMILES格式
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]  # 转换亲和力为-pH值

    # 生成蛋白质预训练表示
    # generate_protein_pretraining_representation(dataset, prots)

    #  ----------------------------------------------------------------------------------------------------

    affinity = np.asarray(affinity)
    opts = ['train','test']
    for i in range(5):
        train_fold = train_folds[i]
        valid_fold = valid_folds[i]
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)
            if opt=='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt=='test':
                rows,cols = rows[valid_fold], cols[valid_fold]

            if i == 0:
                # 生成标准数据
                print('generating standard data')
                with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]]  ]
                        ls += [ prots[cols[pair_ind]]  ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                        ls += [ cols[pair_ind] ]
                        f.write(','.join(map(str,ls)) + '\n')

                print('generating cold data')
                if opt=='train':
                    with open('data/' + dataset + '_cold' + '.csv', 'w') as f:
                        f.write('compound_iso_smiles,target_sequence,affinity,protein_id,drug_id\n')
                        for pair_ind in range(len(rows)):
                            ls = []
                            ls += [ drugs[rows[pair_ind]]  ]
                            ls += [ prots[cols[pair_ind]]  ]
                            ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                            ls += [ cols[pair_ind] ]
                            ls += [ rows[pair_ind] ]
                            f.write(','.join(map(str,ls)) + '\n')
                else:
                    with open('data/' + dataset + '_cold' + '.csv', 'a') as f:
                        for pair_ind in range(len(rows)):
                            ls = []
                            ls += [ drugs[rows[pair_ind]]  ]
                            ls += [ prots[cols[pair_ind]]  ]
                            ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                            ls += [ cols[pair_ind] ]
                            ls += [ rows[pair_ind] ]
                            f.write(','.join(map(str,ls)) + '\n')

            # 生成冷数据
            print('generating 5-fold validation data')
            with open('data/' + dataset + '/' + dataset + '_' + opt + '_fold' + str(i) + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity,protein_id\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    ls += [ cols[pair_ind] ]
                    f.write(','.join(map(str,ls)) + '\n')       

