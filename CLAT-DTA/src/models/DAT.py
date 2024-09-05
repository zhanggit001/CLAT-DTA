# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:50:04 2020

@author: a
"""

from src.models.layers import LinkAttention
import torch
import torch.nn as nn
import numpy as np
from src.utils import pack_sequences, pack_pre_sequences, unpack_sequences, split_text, load_protvec, graph_pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from src.models.transformer import Transformer

device_ids = [0, 1, 2, 3]


class DAT3(nn.Module):
    def __init__(self, embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout_rate,
                 alpha, n_heads, graph_input_dim=78, rnn_layers=2, n_attentions=1,
                 attn_type='dotproduct', vocab=26, smile_vocab=63, is_pretrain=True,
                 is_drug_pretrain=False, n_extend=1):
        super(DAT3, self).__init__()

        # attention部分
        self.dropout = nn.Dropout(dropout_rate)  # 丢弃率
        self.leakyrelu = nn.LeakyReLU(alpha)  # LeakyReLU激活函数
        self.relu = nn.ReLU()  # ReLU激活函数
        self.elu = nn.ELU()  # ELU激活函数
        self.n_attentions = n_attentions  # 注意力数
        self.n_heads = n_heads  # 头数
        self.graph_head_fc1 = nn.Linear(graph_dim * n_heads, graph_dim)  # 图头部全连接层1
        self.graph_head_fc2 = nn.Linear(graph_dim * n_heads, graph_dim)  # 图头部全连接层2
        self.graph_out_fc = nn.Linear(graph_dim, hidden_dim)  # 图输出全连接层
        self.out_attentions1 = LinkAttention(hidden_dim, n_heads)  # 输出注意力1

        # SMILES部分
        self.smiles_vocab = smile_vocab  # SMILES词汇表大小
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=smile_vocab)  # SMILES嵌入层
        self.rnn_layers = 2  # RNN层数
        self.is_bidirectional = True  # 是否双向
        self.encoder = Transformer(256, 256)  # ////////////// MACCS 指纹的长度=2
        self.smiles_input_fc = nn.Linear(256, rnn_dim)  # SMILES输入全连接层，rnn_dim=128
        # 输入维度128，输出维度128，LSTM层数2，输入数据的维度顺序，是否使用双向LSTM，丢弃率
        self.smiles_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)  # SMILES循环神经网络
        # self.smiles_rnn = nn.Conv2d(rnn_dim,rnn_dim,3)  # //////我的尝试
        self.smiles_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)  # SMILES输出全连接层
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)  # 输出注意力3
        self.concat_input_fc = nn.Linear(128, 256)  # 句子输入全连接层

        # 蛋白质部分
        self.is_pretrain = is_pretrain  # 是否预训练
        if not is_pretrain:
            self.vocab = vocab  # 词汇表大小
            self.embed = nn.Embedding(vocab + 1, embedding_dim, padding_idx=vocab)  # 嵌入层
        self.rnn_layers = 2  # RNN层数
        self.is_bidirectional = True  # 是否双向
        self.sentence_input_fc = nn.Linear(embedding_dim, rnn_dim)  # 句子输入全连接层
        self.encode_rnn = nn.LSTM(rnn_dim, rnn_dim, self.rnn_layers, batch_first=True
                                  , bidirectional=self.is_bidirectional, dropout=dropout_rate)  # 编码RNN
        self.rnn_out_fc = nn.Linear(rnn_dim * 2, rnn_dim)  # RNN输出全连接层
        self.sentence_head_fc = nn.Linear(rnn_dim * n_heads, rnn_dim)  # 句子头部全连接层
        self.sentence_out_fc = nn.Linear(2 * rnn_dim, hidden_dim)  # 句子输出全连接层
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)  # 输出注意力2

        # 连接部分
        self.out_attentions = LinkAttention(hidden_dim, n_heads)  # 输出注意力
        # self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)  # 全连接层1
        self.out_fc1 = nn.Linear(hidden_dim * 2, 256 * 8)  # 全连接层1
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)  # 全连接层2
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)  # 全连接层3
        self.layer_norm = nn.LayerNorm(rnn_dim * 2)  # 层归一化

        self.Protein_max_pool = nn.MaxPool1d(640)
        self.Drug_max_pool = nn.MaxPool1d(128)
        self.attention_layer = nn.MultiheadAttention(256, 1)


    def forward(self, protein, smiles):
        # 药物分子
        batchsize = len(protein)
        #256
        smiles_lengths = np.array([len(x) for x in smiles])
        temp = (torch.zeros(batchsize, max(smiles_lengths)) * 63).long()
        for i in range(batchsize):
            temp[i, :len(smiles[i])] = smiles[i]
        smiles = temp.cuda()
        #         smiles = self.smiles_embed(smiles)  # smiles序列进行嵌入，输出维度为256
        smiles_out = self.encoder(smiles)  # smiles序列进入encoder，输出维度为256
#         smiles = self.smiles_input_fc(smiles)  # smiles序列进入nn.Linear(256, 128)，输出维度为128
#         smiles_out, _ = self.smiles_rnn(smiles)  # 使用双向LSTM，输出维度256
        #smiles_out(256,166,256)
        Drug_QKV = smiles_out.permute(1, 0, 2)
        #  proteins
        if self.is_pretrain:
            protein_lengths = np.array([x.shape[0] for x in protein])
            protein = graph_pad(protein, max(protein_lengths))

        h = self.sentence_input_fc(protein)  # smiles序列进入nn.Linear(256, 128)，输出维度为128
        sentence_out, _ = self.encode_rnn(h)  # 使用双向LSTM，输出维度128
        #sentence_out(256,1024,256)
        protein_QKV = sentence_out.permute(1, 0, 2)
        #pro_QKV(1024,256,256)
        x_att, _ = self.attention_layer(Drug_QKV, protein_QKV, protein_QKV)
        #x_att(166,256,256)

        xt_att, _ = self.attention_layer(protein_QKV, Drug_QKV, Drug_QKV)
        #xt_att(1024,256,256)
        x_att = x_att.permute(1, 0, 2)
        #(256,166,256)
        xt_att = xt_att.permute(1, 0, 2)
        #(256,1024,256)
        x_cat = smiles_out * 0.5 + x_att * 0.5
        #(256,166,256)
        xt_cat = sentence_out * 0.5 + xt_att * 0.5
        #(256,1024,256)
        x_cat = x_cat.permute(0,2,1)
        # 256,256,166)
        xt_cat = xt_cat.permute(0,2,1)
        drug_pool = self.Drug_max_pool(x_cat).squeeze(2)
       # (256,256)
        pro_pool = self.Protein_max_pool(xt_cat).squeeze(2)
        #256，256
        #drug_pool = x_cat.squeeze(2)
        #256,166,256
        #pro_pool = xt_cat.squeeze(2)
        #256,1024,256
        #  concat

        out = torch.cat((drug_pool, pro_pool), dim=1)
        #(256,512)
        # 拼接后
        d_block = self.dropout(self.relu(self.out_fc1(out)))  # 256*3->256*8
        #256,256*8
        out = self.dropout(self.relu(self.out_fc2(d_block)))  # 256*8->256*2
       #256,512
        out = self.out_fc3(out).squeeze()  # 256*2->1
        #256,1
        return d_block, out








