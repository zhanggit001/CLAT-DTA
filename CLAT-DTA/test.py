import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv    # 获取药物分子SMILES，目标序列和亲和度
from src.utils import DrugTargetDataset, collate, AminoAcid, ci
from src.models.DAT import DAT3

parser = argparse.ArgumentParser()  # 创建参数解析器
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')  # 添加CUDA参数，默认为True，禁用CUDA训练
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')  # 添加epochs参数，类型为整数，默认为1000，训练的轮数
parser.add_argument('--batchsize', type=int, default=256, help='Number of batch_size')  # 添加batchsize参数，类型为整数，默认为256，批量大小
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')  # 添加lr参数，类型为浮点数，默认为0.001，初始学习率
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')  # 添加weight-decay参数，类型为浮点数，默认为0.00001，权重衰减（参数的L2损失）
parser.add_argument('--embedding-dim', type=int, default=1280, help='dimension of embedding (default: 512)')  # 添加embedding-dim参数，类型为整数，默认为1280，嵌入维度（默认为512）
parser.add_argument('--rnn-dim', type=int, default=128, help='hidden unit/s of RNNs (default: 256)')  # 添加rnn-dim参数，类型为整数，默认为128，RNN的隐藏单元数（默认为256）
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden units of FC layers (default: 256)')  # 添加hidden-dim参数，类型为整数，默认为256，全连接层的隐藏单元数（默认为256）
parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')  # 添加graph-dim参数，类型为整数，默认为256，隐藏单元的数量
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')  # 添加n_heads参数，类型为整数，默认为8，注意力头的数量
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')  # 添加dropout参数，类型为浮点数，默认为0.3，丢弃率（1 - 保留概率）
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')  # 添加alpha参数，类型为浮点数，默认为0.2，leaky_relu的alpha值
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')  # 添加pretrain参数，动作为store_false，指示蛋白质是否预训练
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')  # 添加dataset参数，默认为'davis'，数据集为davis或kiba
parser.add_argument('--training-dataset-path', default='data/davis_train.csv', help='training dataset path: davis or kiba/ 5-fold or not')  # 添加training-dataset-path参数， 默认为'data/davis_train.csv'，训练数据集路径：davis或kiba/ 5-fold或其他
parser.add_argument('--testing-dataset-path', default='data/davis_test.csv', help='training dataset path: davis or kiba/ 5-fold or not')  # 添加testing-dataset-path参数， 默认为'data/davis_test.csv'，测试数据集路径：davis或kiba/ 5-fold或其他

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay

embedding_dim = args.embedding_dim
rnn_dim = args.rnn_dim
hidden_dim = args.hidden_dim
graph_dim = args.graph_dim

n_heads = args.n_heads
dropout = args.dropout
alpha = args.alpha

is_pretrain = args.pretrain

Alphabet = AminoAcid()

training_dataset_address = args.training_dataset_path
testing_dataset_address = args.testing_dataset_path

# ---加载训练数据
# 获取药物分子SMILES，目标序列和亲和度
if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)
else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
# 将训练亲和度转换为浮点型
train_affinity = torch.from_numpy(np.array(train_affinity)).float()

# 创建DrugTargetDataset数据集对象，传入训练药物、蛋白质、亲和度、pid等参数
dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid, is_target_pretrain=is_pretrain, self_link=False, dataset=dataset)
# 创建训练数据集的DataLoader对象，设置批量大小、是否打乱数据和数据整理函数
dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

# ---加载训练数据
if is_pretrain:
    test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
else:
    test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
    test_protein = [x.encode('utf-8').upper() for x in test_protein]
    test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)
dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )
print(dataset_train.X2)