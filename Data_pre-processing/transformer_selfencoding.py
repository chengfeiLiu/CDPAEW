import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# device="cpu"
class handler_data(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        self.data = data
        mask_list = []
        for data_item in self.data:
            curr_mask = []
            if len(data_item)<seq_len:
                curr_mask=[1]*len(data_item)+[0]*(seq_len-len(data_item))
            else:
                curr_mask = [1]*seq_len
            mask_list.append(curr_mask)
        self.src_mask = mask_list
    def __getitem__(self, index):
        return {"tras_martix":torch.tensor([self.data[index]]),"src_mask":~torch.tensor(self.src_mask[index],dtype=torch.bool)}
    def __len__(self):
        return len(self.data)
batch_size = 16
d_model=512
seq_len =40 
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
            pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
            ## 上面代码获取之后得到的pe:[max_len*d_model]

            ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
            """
            x: [seq_len, batch_size, d_model]
            """
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)
class transformers(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(transformers,self).__init__()
        self.linear1 = nn.Linear(1,d_model)
        self.pos_embedding  = PositionalEncoding(d_model=d_model,max_len=seq_len).to(device)
        self.transformerencoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=8,dim_feedforward=4*d_model,batch_first=True)
        self.transformerdecoder_layer = nn.TransformerDecoderLayer(d_model=d_model,nhead=8,dim_feedforward=4*d_model,batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformerencoder_layer,num_layers=8).to(device)
        self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer,num_layers=8).to(device)
        self.linear = nn.Linear(d_model,1)
    def forward(self,tras_martix,src_mask_):
        tras_martix = self.linear1(tras_martix)
        tras_martix = self.pos_embedding(tras_martix.transpose(0,1)).transpose(0,1)
        # print(tras_martix.shape)
        out = self.encoder(src = tras_martix,src_key_padding_mask=src_mask_)
        dec_out = self.decoder(tras_martix,out)
        tgt = self.linear(dec_out)
        # print(tgt)
        return tgt
criterion = nn.L1Loss(reduction='sum').to(device)
# _model = transformers(3,1).to(device)
_model = torch.load('./lrm_model/model_94.pt')
# 9675.930725097656,8732.237854003906,9075.17398071289,9186.740936279297,8704.185089111328,2660.466354370117,1915.9348068237305,1915.9348068237305,1528.6329193115234, 3314.258773803711
# ,1034.1473274230957,1539.7082443237305,986.1357460021973,2441.1692504882812,1608.1407470703125,1137.1153945922852, 1212.6370735168457,854.7054290771484,1418.8653259277344,1385.4924621582031
# ,1883.807632446289,925.9901237487793,728.9442596435547,864.4033737182617
optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4)
def train(trans_data_loader):
    loss_data=[]
    _model.train()
    Echop = 100
    for Echop in range(Echop):
        total_loss = 0
        items = 0
        for date_item in trans_data_loader:
            tras_martix = date_item['tras_martix'].transpose(1,2).to(device)
            src_mask = date_item['src_mask'].to(device)
            # tras_martix = tras_martix.reshape(4,5)
            # src_mask = src_mask.reshape(4,1,d_model)
            # print(tras_martix)
            # print(src_mask.shape)
            outputs = _model(tras_martix=tras_martix,src_mask_= src_mask).to(device)
            output = criterion(tras_martix, outputs).to(device)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            acc=0
            items+=1
            total_loss+=output.item()
            print('Epoch {}, Step {} Train Loss {} '.format(Echop, items, output.item()) )
        torch.save(_model,'.\lrm_model\model_'+str(Echop)+".pt")
        loss_data.append(total_loss)
        print('Epoch {},  Totle Loss {}'.format(Echop, total_loss) )
    plt.plot(np.arange(0,Echop+1),loss_data)
    df = pd.DataFrame(loss_data, columns=['loss_data'])
    df['epoch']=np.arange(0,Echop+1)
    df.to_excel("loss_list.xlsx", index=False)
    print(loss_data)
    plt.show()
def vaildation():
    return

def plot_time_series_class(data, class_name, ax, n_steps=5):
    """
    param data:数据
    param class_name: 不同心跳类名
    param ax:画布
    """
    time_series_df = pd.DataFrame(data)
    # 平滑时间窗口
    smooth_path = time_series_df.rolling(n_steps).mean()
    # 路径偏差
    path_deviation = 1.5*time_series_df.rolling(n_steps).std()
    print(smooth_path)
    # 以正负偏差上下定义界限
    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]
    # 绘制平滑曲线
    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
      path_deviation.index,
      under_line,
      over_line,
      alpha=.125)
    ax.set_title(class_name)
min_max_scaler = preprocessing.MinMaxScaler(copy=True,feature_range=(0,1))
def main():
    data = pd.read_excel("23-03handler.xlsx",usecols=[1,2]).dropna(how='any')
    jiance_data = np.array(np.array(list(data['jiance'])).astype(dtype=float).tolist()).reshape(-1,1)
    print(jiance_data.shape)
    print(min_max_scaler.fit_transform(jiance_data))
    data['jiance'] = jiance_data.squeeze()
    data.reset_index(drop=True, inplace=True)
    alarm_list = []
    noalarm_list = []
    target=[]
    count_1 = 0
    i = 0
    for i in range(len(data['jiance'])-seq_len):
        flag = False
        for state_data in data['state'][i:i+seq_len]:
            if state_data==1:
                target.append(1)
                count_1+=1
                flag=True
                alarm_list.append([data_ for data_ in data['jiance'][i:i+seq_len]])
                break
        if not flag:
            target.append(0)
            noalarm_list.append([data_ for data_ in data['jiance'][i:i+seq_len]])
    print(len(target))
    print(noalarm_list)
    print(len(alarm_list))
    # plot_data = pd.Series([count_1,(len(target)-count_1)],index=['alarm', 'noalarm'])
    plot_data = pd.Series({"target":target})
    classes = data.state.unique()
    print('classed',classes)
    class_names=[0,1]
# 定义画布
    fig, axs = plt.subplots(
        nrows=len(classes) // 3 + 1,
        ncols=2,
        sharey=True,
        figsize=(12, 5))
    # 循环绘制曲线
    for i, cls in enumerate(classes):
        print('cls',cls)
       
        ax = axs.flat[i]
        datad = data[data.state == cls] \
        .drop(labels='state', axis=1) \
        .to_numpy()
        print( 'daad',datad)
        plot_time_series_class(datad, class_names[i], ax)

    fig.delaxes(axs.flat[-1])
    fig.tight_layout();

    # ax = sns.countplot(x="target", data=plot_data)
    plt.show()
    tras_data_noalarm,test_vaildation_noalarm = train_test_split(noalarm_list,train_size=0.8,test_size=0.2,shuffle=False,random_state=1)
    test_data_noalarm,vaildation_data_noalarm = train_test_split(test_vaildation_noalarm,train_size=0.5,shuffle=False,random_state=1)
    handler_tras_data_noalarm = handler_data(data=tras_data_noalarm)
    handler_vaildation_data_noalarm = handler_data(data=vaildation_data_noalarm)
    handler_test_data_noalarm = handler_data(data=test_data_noalarm)
    trans_data_loader_noalarm = DataLoader(handler_tras_data_noalarm,batch_size=batch_size,shuffle=True)
    vaildation_data_loader_noalarm = DataLoader(handler_vaildation_data_noalarm,batch_size=batch_size,shuffle=True)
    test_data_loader_noalarm = DataLoader(handler_test_data_noalarm,batch_size=batch_size,shuffle=True)
    
    tras_data_alarm,test_vaildation_alarm = train_test_split(alarm_list,train_size=0.8,test_size=0.2,shuffle=False,random_state=1)
    vaildation_data_alarm,test_data_alarm = train_test_split(test_vaildation_alarm,train_size=0.5,shuffle=False,random_state=1)
    handler_tras_data_alarm = handler_data(data=tras_data_alarm)
    handler_vaildation_data_alarm = handler_data(data=vaildation_data_alarm)
    handler_test_data_alarm = handler_data(data=test_data_alarm)
    trans_data_loader_alarm = DataLoader(handler_tras_data_alarm,batch_size=batch_size,shuffle=True)
    vaildation_data_loader_alarm = DataLoader(handler_vaildation_data_alarm,batch_size=batch_size,shuffle=True)
    test_data_loader_alarm = DataLoader(handler_test_data_alarm,batch_size=batch_size,shuffle=True)
    
    return trans_data_loader_noalarm,vaildation_data_loader_noalarm,test_data_loader_noalarm,trans_data_loader_alarm,vaildation_data_loader_alarm,test_data_loader_alarm
def predict(model, dataset):
    predictions, losses = [], []
    total_loss = 0
    criterion = nn.L1Loss(reduction='none').to(device)
    with torch.no_grad():
        model.eval()
        for date_item in dataset:
            tras_martix = date_item['tras_martix'].transpose(1,2).to(device)
            src_mask = date_item['src_mask'].to(device)
            outputs = _model(tras_martix=tras_martix,src_mask_= src_mask).to(device)
            output = criterion(tras_martix, outputs).to(device)
            output = output.reshape(output.shape[0],output.shape[1]).sum(dim=1)
            sun_loss = torch.sum(output)
            total_loss += sun_loss
            print('Train Loss {} '.format(torch.sum(output)) )
            # predictions.append(outputs.cpu().numpy().flatten())
            # print(output.item().shape)
            losses.extend(torch.tensor(output))
            # print(losses)
        # print("loss",total_loss)
    return predictions, losses,total_loss
if __name__=='__main__':
    trans_data_loader_noalarm,vaildation_data_loader_noalarm,test_data_loader_noalarm,trans_data_loader_alarm,vaildation_data_loader_alarm,test_data_loader_alarm=main()
    print(len(vaildation_data_loader_noalarm))
    # train(trans_data_loader_noalarm)
    # all_loss = []
    # for i in range(100):
    #     _model = torch.load('./lrm_model/model_'+str(i)+'.pt')  # 18
    #     _, losses,total_loss = predict(_model, vaildation_data_loader_noalarm)
    #     all_loss.append(torch.Tensor.cpu(total_loss))
    # print(all_loss)
    # plt.plot(np.arange(0, 100), all_loss)
    # df = pd.DataFrame(all_loss, columns=['all_loss'])
    # df['epoch'] = np.arange(0, 100)
    # df.to_excel("val_loss_list.xlsx", index=False)
    # plt.show()
    # _, losses = predict(_model, trans_data_loader_alarm)
    # _, losses = predict(_model, vaildation_data_loader_noalarm)
    # _, losses = predict(_model, vaildation_data_loader_alarm)
    _, losses ,total_loss= predict(_model, test_data_loader_noalarm)
    _alarm, losses_alarm,total_loss = predict(_model, test_data_loader_alarm)
    losses = [it.item() for it  in losses]
    losses_alarm = [it.item() for it in losses_alarm]
    threshold = 5
    correct = sum(l <= threshold for l in losses)
    print(f'Correct noalarm predictions: {correct}/{len(losses)}')
    correct_alarm = sum(l >= threshold for l in losses_alarm)
    print(f'Correct alarm predictions: {correct_alarm}/{len(losses_alarm)}')
    sns.displot(torch.tensor(losses), bins=50, kde=True)
    plt.show()
    sns.displot(torch.tensor(losses_alarm), bins=50, kde=True)
    plt.show()
    plt.legend()

    print('losses_alarm,',losses_alarm)
    print('losses_nolarm',losses)
    print('test_data_loader_noalarm',len(test_data_loader_noalarm))
    print('test_data_loader_alarm',len(test_data_loader_alarm))
    print('losses_alarm_len', len(losses_alarm))
    print('losses_noalarm_len', len(losses))
    df = pd.DataFrame(losses_alarm, columns=['losses_alarm'])
    df['epoch'] = np.arange(0, len(losses_alarm))
    df.to_excel("test_alarm_loss_list.xlsx", index=False)

    df = pd.DataFrame(losses, columns=['losses_nolarm'])
    df['epoch'] = np.arange(0, len(losses))
    df.to_excel("test_noalarm_loss_list.xlsx", index=False)
