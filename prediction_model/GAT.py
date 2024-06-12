import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data, accuracy
from torch.autograd import Variable
import os
import glob
import time
import random
import torch.optim as optim
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha,bs, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout #dropout参数
        self.in_features = in_features #结点向量的特征维度
        self.out_features = out_features #经过GAT之后的特征维度
        self.alpha = alpha#LeakyReLU参数
        self.concat = concat# 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.empty(size=(bs,in_features, out_features)))#[16,16]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)# xavier初始化
        self.a = nn.Parameter(torch.empty(size=(bs,2*out_features, 1)))#[32,1]
        nn.init.xavier_uniform_(self.a.data, gain=1.414)# xavier初始化

        # 定义leakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        adj图邻接矩阵，维度[N,N]非零即一
        h.shape: (N, in_features), self.W.shape:(in_features,out_features)
        Wh.shape: (N, out_features)
        '''
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)# 对应eij的计算公式[9,16]
        e = self._prepare_attentional_mechanism_input(Wh)#对应LeakyReLU(eij)计算公式[9,9]

        zero_vec = -9e15*torch.ones_like(e)#将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)#[N,N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask设置为非常小的值，因为softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=1)# softmax形状保持不变[N,N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)# dropout,防止过拟合
        h_prime = torch.matmul(attention, Wh)#[N,N].[N,out_features]=>[N,out_features]
        # 得到由周围节点通过注意力权重进行更新后的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # 先分别与a相乘再进行拼接
        Wh1 = torch.matmul(Wh, self.a[:,:self.out_features, :])#[9,16]*[16,1]=[9,1]
        Wh2 = torch.matmul(Wh, self.a[:,self.out_features:, :])#[9,1]
        # broadcast add
        e = Wh1 + Wh2.transpose(1,2)#[9,9]
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,bs,d_model):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # 加入Multi-head机制
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,bs=bs, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,bs=bs, concat=False)
        self.liner = nn.Linear(in_features=nclass*nclass*nhid,out_features=bs*nhid*d_model,bias=True)

    def forward(self, x, adj,d_model):
        sql_len = x.shape[2]
        bs = x.shape[0]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        print(x.shape)
        x = self.liner(x.reshape(x.shape[0]*x.shape[1]*x.shape[2])).reshape(bs,sql_len,d_model)
        return x
        # return F.log_softmax(x, dim=1)
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# # no-cuda=False
# fastmode=False
# sparse=False
# seed=72
# epochs=10000
# lr=0.005
# weight_decay=5e-4
# hidden=8
# nb_heads=8
# dropout=0.6
# alpha=0.2
# patience=100
# hidden=8
# nb_heads=8
# dropout=0.6
# alpha=0.2

# model = GAT(nfeat=features.shape[1], 
#                 nhid=hidden, 
#                 nclass=int(labels.max()) + 1, 
#                 dropout=dropout, 
#                 nheads=nb_heads, 
#                 alpha=alpha)
# optimizer = optim.Adam(model.parameters(), 
#                        lr=lr, 
#                        weight_decay=weight_decay)
# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


# def train(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output = model(features, adj)
#     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#     loss_train.backward()
#     optimizer.step()

#     if not fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output = model(features, adj)

#     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.data.item()),
#           'acc_train: {:.4f}'.format(acc_train.data.item()),
#           'loss_val: {:.4f}'.format(loss_val.data.item()),
#           'acc_val: {:.4f}'.format(acc_val.data.item()),
#           'time: {:.4f}s'.format(time.time() - t))

#     return loss_val.data.item()
# def compute_test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.data.item()),
#           "accuracy= {:.4f}".format(acc_test.data.item()))

# # Train model
# t_total = time.time()
# loss_values = []
# bad_counter = 0
# best = epochs + 1
# best_epoch = 0
# for epoch in range(epochs):
#     loss_values.append(train(epoch))

#     torch.save(model.state_dict(), '{}.pkl'.format(epoch))
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1

#     if bad_counter == patience:
#         break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# # Testing
# compute_test()
