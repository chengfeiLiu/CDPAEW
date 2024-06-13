import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from copy import deepcopy
plt.rcParams['font.family'] = 'Microsoft YaHei'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.cuda.is_available())
# device="cpu"
scaler = MinMaxScaler() 
class handler_data(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        data = data.reset_index()
        print(data)
        all_fenchen_list = data['all_fenchen_list']
        all_fengsu_list = data['all_fengsu_list']
        all_wendu_list = data['all_wendu_list']
        all_target_list = data['all_target_list']
        self.all_fenchen_list =  all_fenchen_list
        self.all_fengsu_list = all_fengsu_list
        self.all_wendu_list = all_wendu_list
        self.all_target_list =  all_target_list
        self.target = data['target']
        mask_list = []
        for data_item in self.all_fenchen_list:
            curr_mask = []
            if len(data_item)<seq_len:
                curr_mask=[1]*len(data_item)+[0]*(seq_len-len(data_item))
            else:
                curr_mask = [1]*seq_len
            mask_list.append(curr_mask)
        self.src_mask = mask_list
    def __getitem__(self, index):
        return {"fenchen":torch.tensor([self.all_fenchen_list[index]],dtype=torch.float),"fengsu":torch.tensor([self.all_fengsu_list[index]],dtype=torch.float),"wendu":torch.tensor([self.all_wendu_list[index]],dtype=torch.float),
                "all_target_list":torch.tensor([self.all_target_list[index]],dtype=torch.float),"target":torch.tensor([self.target[index]],dtype=torch.float),
                "src_mask":~torch.tensor(self.src_mask[index],dtype=torch.bool)}
    def __len__(self):
        return len(self.all_fenchen_list)
batch_size = 16
d_model=64
seq_len =16 
nhead=4
d_k=d_v=d_q=32
n_layers=4*d_model
d_ff = 1024  # FeedForward dimension
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)  

    def forward(self, x):
            """
            x: [seq_len, batch_size, d_model]
            """
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)
class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * nhead)
        self.W_K = nn.Linear(d_model, d_k * nhead)
        self.W_V = nn.Linear(d_model, d_v * nhead)
        self.linear = nn.Linear(nhead * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self,Q1, K1, V1,Q2, K2, V2,Q3, K3, V3,attn_mask):
        residual1, batch_size1 = Q1, Q1.size(0)
        residual2, batch_size2 = Q2, Q2.size(0)
        residual3, batch_size3 = Q3, Q3.size(0)
        # print("batch_size{}".format(batch_size1))
        q_s1 = self.W_Q(Q1).view(batch_size1, -1, nhead, d_k).transpose(1,2)
        k_s1 = self.W_K(K1).view(batch_size1, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s1 = self.W_K(V1).view(batch_size1, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]

        q_s2 = self.W_Q(Q2).view(batch_size2, -1, nhead, d_k).transpose(1,2)
        k_s2 = self.W_K(K2).view(batch_size2, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s2 = self.W_K(V2).view(batch_size2, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        
        q_s3 = self.W_Q(Q3).view(batch_size3, -1, nhead, d_k).transpose(1,2)
        k_s3 = self.W_K(K3).view(batch_size3, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s3 = self.W_K(V3).view(batch_size3, -1, nhead, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(3).repeat(1,nhead,1,seq_len)

        context1 = ScaledDotProductAttention()(q_s1, k_s1, v_s1, attn_mask)
        context2 = ScaledDotProductAttention()(q_s2, context1, v_s2, attn_mask)
        context3 = ScaledDotProductAttention()(q_s3, context1, v_s3, attn_mask)
        context1 = context1.transpose(1, 2).contiguous().view(batch_size, -1, nhead * d_v) # context: [batch_size x len_q x n_heads * d_v]
        context2 = context2.transpose(1, 2).contiguous().view(batch_size, -1, nhead * d_v) # context: [batch_size x len_q x n_heads * d_v]
        context3 = context3.transpose(1, 2).contiguous().view(batch_size, -1, nhead * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output1 = self.linear(context1)
        output2 = self.linear(context2)
        output3 = self.linear(context3)
        output = output1+output2+output3
        return self.layer_norm(output+residual1)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs_1,enc_inputs_2, enc_inputs_3,enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs_1, enc_inputs_1, enc_inputs_1, enc_inputs_2,enc_inputs_2,enc_inputs_2,enc_inputs_3,enc_inputs_3,enc_inputs_3,enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs_1,enc_inputs_2,enc_inputs_3,mask):
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs = layer(enc_inputs_1=enc_inputs_1,enc_inputs_2=enc_inputs_2,enc_inputs_3=enc_inputs_3,enc_self_attn_mask = mask)
        return enc_outputs
class transformers(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(transformers,self).__init__()
        self.linear1 = nn.Linear(1,d_model)
        self.pos_embedding  = PositionalEncoding(d_model=d_model,max_len=seq_len).to(device)
        self.transformerdecoder_layer = nn.TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=4*d_model,batch_first=True)
        self.encoder = Encoder()
        self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer,num_layers=n_layers).to(device)
        self.linear2 = nn.Linear(d_model,1)
        self.linear3 = nn.Linear(seq_len,5)
    def forward(self,fenchen_matrix,fengsu_matrix,wendu_matrix,src_mask_):
        fenchen_matrix = self.linear1(fenchen_matrix)
        fengsu_matrix = self.linear1(fengsu_matrix)
        wendu_matrix = self.linear1(wendu_matrix)
        fenchen_matrix = self.pos_embedding(fenchen_matrix.transpose(0,1)).transpose(0,1)
        fengsu_matrix = self.pos_embedding(fengsu_matrix.transpose(0,1)).transpose(0,1)
        wendu_matrix = self.pos_embedding(wendu_matrix.transpose(0,1)).transpose(0,1)
        # print(tras_martix.shape)
        out = self.encoder(fenchen_matrix,fengsu_matrix,wendu_matrix,mask=src_mask_)
        print(out.shape)
        dec_out = self.decoder(fenchen_matrix,out)
        tgt = self.linear2(dec_out)
        tgt = tgt.reshape(tgt.shape[0],tgt.shape[1]*tgt.shape[2])
        tgt = self.linear3(tgt)
        tgt=tgt.reshape([tgt.shape[0],tgt.shape[1],1])
        return torch.exp(tgt)
        # return tgt
criterion = nn.L1Loss(reduction='sum').to(device)
_model = torch.load('./model/model_899.pt')#18
optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4)
def train(trans_data_loader):
    loss_data=[]
    _model.train()
    Echop = 100
    for Echop in range(Echop):
        total_loss = 0
        items = 0
        for date_item in trans_data_loader:
            all_fenchen_list = date_item['fenchen'].transpose(1,2).to(device)
            all_fengsu_list = date_item['fengsu'].transpose(1,2).to(device)
            all_wendu_list = date_item['wendu'].transpose(1,2).to(device)
            all_target_list = date_item['all_target_list'].transpose(1,2).to(device)
            target = date_item['target']
            src_mask = date_item['src_mask'].to(device)
            print(src_mask.shape)
            print(all_target_list[0])
            outputs = _model(fenchen_matrix=all_fenchen_list,fengsu_matrix=all_fengsu_list,wendu_matrix=all_wendu_list,src_mask_= src_mask).to(device)
            print(outputs[0])
            output = criterion(all_target_list, outputs).to(device)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            acc=0
            items+=1
            total_loss+=output.item()
            print('Epoch {}, Step {} Train Loss {} '.format(Echop, items, output.item()) )
        torch.save(_model,'.\model\model_'+str(80+Echop)+".pt")
        loss_data.append(total_loss)
        print('Epoch {},  Totle Loss {}'.format(Echop, total_loss) )
        
    torch.save(_model,'.\model\model_'+str(800+Echop)+".pt")
    plt.plot(np.arange(0,Echop+1),loss_data)
    print(loss_data)
    plt.show()
def vaildation():
    return

def main():
    data = pd.read_excel("./data/all/31060/22-10_all_tot.xlsx",usecols=[0,1,2,3,4,5,6,7]).dropna(how='any')
    data_org = pd.read_excel("./data/all/31060/22-10_all_org.xlsx",usecols=[0,1,2,3,4,5,6,7]).dropna(how='any')
    fenchen_data = data['fenchen'].to_list()
    fengsu_data = data['fengsu'].to_list()
    state_data = data['状态'].to_list()
    wendu_data = data['wendu'].to_list()
    fenchen_data_org = data_org['fenchen'].to_list()
    fengsu_data_org = data_org['fengsu'].to_list()
    state_data_org = data_org['状态'].to_list()
    wendu_data_org = data_org['wendu'].to_list()
    org_data = deepcopy(fenchen_data)
    org_fengsu_data = deepcopy(fengsu_data)
    org_state_data = deepcopy(state_data)
    org_wendu_data = deepcopy(wendu_data)
    np.random.seed(12345)
    for i in range(10):
        white_noise = 0.2*np.random.randn(len(org_data))+0.4
        fenchen_data.extend(org_data+white_noise)
        fengsu_data.extend(org_fengsu_data)
        state_data.extend(org_state_data)
        wendu_data.extend(org_wendu_data)
        fenchen_data.extend(fenchen_data_org[i*len(org_data):(i+1)*len(org_data)])
        fengsu_data.extend(fengsu_data_org[i*len(org_fengsu_data):(i+1)*len(org_fengsu_data)])
        state_data.extend(state_data_org[i*len(org_state_data):(i+1)*len(org_state_data)])
        wendu_data.extend(wendu_data_org[i*len(org_wendu_data):(i+1)*len(org_wendu_data)])
        print(org_data+white_noise)
    print(len(fenchen_data))
    print(len(fenchen_data))
    print(len(state_data))
    data = pd.DataFrame({'fengsu':fengsu_data,'fenchen':fenchen_data,'wendu':wendu_data,'状态':state_data})[11:]
    data.reset_index(drop=True, inplace=True)
    alarm_list = []
    noalarm_list = []
    all_fenchen_list = []
    all_fengsu_list = []
    all_wendu_list = []
    all_target_list = []
    print(data.info())
    target=[]
    count_1 = 0
    i = 0
    for i in range(len(data['fenchen'])):
        if (i+seq_len)>=(len(data['fenchen'])-4):
            break
        flag = False
        pre = 0
        cur_fenchen = [data_ for data_ in data['fenchen'][i:i+seq_len]]
        cur_fengsu = [data_ for data_ in data['fengsu'][i:i+seq_len]]
        cur_wendu = [data_ for data_ in data['wendu'][i:i+seq_len]]
        for state_data in data['状态'][i:i+seq_len]:
            if state_data==1:
                if pre==1:
                    target.append(1)
                    count_1+=1
                    flag=True
                    alarm_list.append(cur_fenchen)
                    break
                else:
                    pre==1
        if not flag:
            target.append(0)
            noalarm_list.append(cur_fenchen)
        all_fenchen_list.append(cur_fenchen)
        all_fengsu_list.append(cur_fengsu)
        all_wendu_list.append(cur_wendu)
        all_target_list.append([data_ for data_ in data['fenchen'][i+seq_len:i+seq_len+5]])
    data_after = pd.DataFrame({'all_fenchen_list':all_fenchen_list,'all_fengsu_list':all_fengsu_list,'all_target_list':all_target_list,'all_wendu_list':all_wendu_list,'target':target})
    print(data_after)
    print(len(target))
    print(noalarm_list)
    print(len(alarm_list))
    plot_data = pd.Series({"target":target})
    classes = data.状态.unique()
    print('classed',classes)
    class_names=[0,1]
    # ax = sns.countplot(x="target", data=plot_data)
    plt.show()
    tras_data,test_vaildation = train_test_split(data_after,train_size=0.9,test_size=0.1,shuffle=False,random_state=1)
    test_vaildation.reset_index()
    test_data,vaildation_data = train_test_split(test_vaildation,train_size=0.1,shuffle=False,random_state=1)
    vaildation_data.reset_index()
    test_data.reset_index()
    handler_tras_data = handler_data(data=tras_data)
    handler_vaildation_data = handler_data(data=vaildation_data)
    handler_test_data = handler_data(data=test_data)
    trans_data_loader = DataLoader(handler_tras_data,batch_size=batch_size,shuffle=False)
    vaildation_data_loader = DataLoader(handler_vaildation_data,batch_size=batch_size,shuffle=False)
    test_data_loader = DataLoader(handler_test_data,batch_size=batch_size,shuffle=False)
    
    return trans_data_loader,vaildation_data_loader,test_data_loader
def predict(model, dataset):
    predictions, losses = [], []
    total_loss = 0
    all_data_true = []
    all_data_pred = []
    i = 0
    criterion = nn.L1Loss(reduction='none').to(device)
    with torch.no_grad():
        model.eval()
        for date_item in dataset:
            print(date_item)
            if len(date_item['fenchen'])!=batch_size:
                continue
            all_fenchen_list = date_item['fenchen'].transpose(1,2).to(device)
            all_fengsu_list = date_item['fengsu'].transpose(1,2).to(device)
            all_wendu_list = date_item['wendu'].transpose(1,2).to(device)
            all_target_list = date_item['all_target_list'].transpose(1,2).to(device)
            target = date_item['target']
            src_mask = date_item['src_mask'].to(device)
            all_data_true.extend(all_target_list.cpu()[:,0,:].T[0].numpy().tolist())
            print(src_mask.shape)
            print(all_target_list[0])
            outputs = _model(fenchen_matrix=all_fenchen_list,fengsu_matrix=all_fengsu_list,wendu_matrix=all_wendu_list,src_mask_= src_mask).to(device)
            print(outputs[0])
            all_data_pred.extend(outputs.cpu()[:,0, :].T[0].numpy().tolist())
            # print(outputs)
            output = criterion(all_target_list, outputs).to(device)
            output = output.reshape(output.shape[0],output.shape[1]).sum(dim=1)
            print('Train Loss {} '.format(torch.sum(output)) )
            losses.extend(torch.tensor(output))
                
            plt.title('实际值与预测值')
            plt.xlabel("时间步数")
            plt.ylabel("值")
            plt.plot(np.arange(0,len(all_data_true)),all_data_true,'rs:',label='实际值')
            plt.plot(np.arange(0,len(all_data_pred)),all_data_pred,'m<:',label='预测值')
            plt.legend()
            plt.show()
        result = pd.DataFrame({"true":all_data_true,"pred":all_data_pred})
        data_result_write = pd.ExcelWriter("./result_31_notshuffle.xlsx")
        result.to_excel(data_result_write)
        data_result_write.close()

    return predictions, losses
if __name__=='__main__':
    trans_data_loader,vaildation_data_loader,test_data_loader=main()
    _, losses = predict(_model, trans_data_loader)
