import torch 
from torch import nn
import torch.nn.functional as F 
from torch.optim import Adam

import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import date
import argparse
from progressbar import *
device="cpu"
class TemporalPatternAttention(nn.Module):
    
    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size # 1
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1 # hidden_size
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()
    
    def forward(self, H, ht): # H:(batch_size, 1, obs_len-1, hidden_size) ht:(batch_size, hidden_size)       
        _, channels, _, attn_size = H.size()

        conv_vecs = self.conv(H) # (batch_size, filter_num, 1, hidden_size)      
        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num) # (batch_size, hidden_size, filter_num)
        conv_vecs = self.relu(conv_vecs) # (batch_size, hidden_size, filter_num)
        
        # score function
        htt = self.linear1(ht) # (batch_size, filter_num) 
        htt = htt.view(-1, self.filter_num, 1) # (batch_size, filter_num, 1)
        s = torch.bmm(conv_vecs, htt) # (batch_size, hidden_size, 1)
        alpha = torch.sigmoid(s) # (batch_size, hidden_size, 1)
        v = torch.bmm(conv_vecs.view(-1,self.filter_num,attn_size), alpha).view(-1, self.filter_num) # (batch_size, filter_num)
        
        concat = torch.cat([ht, v], dim=1) # (batch_size, hidden_size+filter_num)
        new_ht = self.linear2(concat) # (batch_size, hidden_size)
        return new_ht

class TPALSTM(nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, \
                    bias=True, batch_first=True) # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 16
        self.filter_size = 1
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, \
            self.filter_num, obs_len-1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, obs_len, features_size = x.shape #(batch_size, obs_len, features_size)
        xconcat = self.hidden(x) #(batch_size, obs_len, hidden_size)

        H = torch.zeros(batch_size, obs_len-1, self.hidden_size).to(device) #(batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device) # (num_layers, batch_size, hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1) #(batch_size, 1, hidden_size)
            out, (ht, ct) = self.lstm(xt, (ht, ct)) # ht size (num_layers, batch_size, hidden_size)
            htt = ht[-1, :, :] # (batch_size, hidden_size)
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H) #(batch_size, obs_len-1, hidden_size)
        
        # reshape hidden states H
        H = H.view(batch_size, 1, obs_len-1, self.hidden_size) #(batch_size, 1, obs_len-1, hidden_size)
        new_ht = self.attention(H, htt) # (batch_size, hidden_size)
        ypred = self.linear(new_ht) # (batch_size, output_horizon)
        return ypred
