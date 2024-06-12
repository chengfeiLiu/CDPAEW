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
from  copy import deepcopy
from tcn import TemporalConvNet
plt.rcParams['font.family'] = 'Microsoft YaHei'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        all_hengduan_list = data['all_hengduan_list']
        all_juejinjc_list = data['all_juejinjc_list']
        all_huifen_list = data['all_huifen_list']
        all_huifafen_list = data['all_huifafen_list']
        all_shuifen_list = data['all_shuifen_list']
        all_qingjiao_list = data['all_qingjiao_list']
        all_gouzao_list = data['all_gouzao_list']
        all_yingdu_list = data['all_yingdu_list']
        all_target_list = data['all_target_list']
        # all_fenchen_list_  = []
        # all_fengsu_list_ = []
        # all_target_list_= []
        # print(all_fenchen_list[0])
        # for item in range(len(all_fenchen_list)):
        #     all_fenchen_list_.append(all_fenchen_list[item])
        #     all_fengsu_list_.append(all_fengsu_list[item])
        #     all_target_list_.append(all_target_list[item])
        # # self.all_fenchen_list =  scaler.fit_transform(all_fenchen_list_)
        # self.all_fengsu_list = scaler.fit_transform(all_fengsu_list_)
        # self.all_target_list =  scaler.fit_transform(all_target_list_)
        self.all_fenchen_list =  all_fenchen_list
        self.all_fengsu_list = all_fengsu_list
        self.all_wendu_list = all_wendu_list
        self.all_hengduan_list =  all_hengduan_list
        self.all_juejinjc_list =  all_juejinjc_list
        self.all_huifen_list =  all_huifen_list
        self.all_huifafen_list =  all_huifafen_list
        self.all_shuifen_list =  all_shuifen_list
        self.all_qingjiao_list =  all_qingjiao_list
        self.all_gouzao_list =  all_gouzao_list
        self.all_yingdu_list =  all_yingdu_list
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
                "hengduan":torch.tensor([self.all_hengduan_list[index]]),"juejinjc":torch.tensor([self.all_juejinjc_list[index]]),"huifen":torch.tensor([self.all_huifen_list[index]]),"huifafen":torch.tensor([self.all_huifafen_list[index]]),
                "shuifen":torch.tensor([self.all_shuifen_list[index]]),"qingjiao":torch.tensor([self.all_qingjiao_list[index]]),"gouzao":torch.tensor([self.all_gouzao_list[index]]),"yingdu":torch.tensor([self.all_yingdu_list[index]]),"all_target_list":torch.tensor([self.all_target_list[index]],dtype=torch.float),
                "target":torch.tensor([self.target[index]],dtype=torch.float),"src_mask":~torch.tensor(self.src_mask[index],dtype=torch.bool)}
    def __len__(self):
        return len(self.all_fenchen_list)
batch_size = 16
d_model=64
seq_len =32 
nhead=4
d_k=d_v=d_q=32
n_layers=4*d_model
d_ff = 1024  # FeedForward dimension
fastmode=False
sparse=False
seed=72
epochs=10000
lr=0.005
nfeat = 16
weight_decay=5e-4
hidden=16
nb_heads=8
dropout=0.6
alpha=0.2
patience=100
nb_heads=8
dropout=0.6
alpha=0.2
nclass=9                                                                                                         

class transformers(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(transformers,self).__init__()
        self.linear1 = nn.Linear(1,d_model)
        self.tcn =  TemporalConvNet(num_channels=[32,16,4,5],num_inputs=11,kernel_size=3)
        self.linear2 = nn.Linear(32,1)
    def forward(self,data):
        print('data.shape',data.shape)
        tcndata=self.tcn(data)
        print('cnndata.shape',tcndata.shape)
        # rnndata = rnndata.transpose(0,1).reshape(rnndata.shape[1],rnndata.shape[0]*rnndata.shape[2])
        print('cnndata.shape',tcndata.shape)
        # dec_out = self.decoder(fenchen_matrix,out)
        tgt = self.linear2(tcndata)
        print('tgt',tgt.shape)
        # return torch.exp(tgt)
        return tgt
criterion = nn.L1Loss(reduction='sum').to(device)
# _model = transformers(3,1).to(device)
_model = torch.load('./model/model_tcn.pt')#18
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
            all_fenchen_list = torch.squeeze(date_item['fenchen'].transpose(1,2).to(device))
            all_fengsu_list = torch.squeeze(date_item['fengsu'].transpose(1,2).to(device))
            all_wendu_list = torch.squeeze(date_item['wendu'].transpose(1,2).to(device))
            all_hengduan_list = torch.squeeze(date_item['hengduan'].transpose(1,2).to(device))
            all_juejinjc_list = torch.squeeze(date_item['juejinjc'].transpose(1,2).to(device))
            all_huifen_list = torch.squeeze(date_item['huifen'].transpose(1,2).to(device))
            all_huifafen_list = torch.squeeze(date_item['huifafen'].transpose(1,2).to(device))
            all_shuifen_list = torch.squeeze(date_item['shuifen'].transpose(1,2).to(device))
            all_qingjiao_list = torch.squeeze(date_item['qingjiao'].transpose(1,2).to(device))
            all_gouzao_list = torch.squeeze(date_item['gouzao'].transpose(1,2).to(device))
            all_yingdu_list = torch.squeeze(date_item['yingdu'].transpose(1,2).to(device))
            print('all_fenchen_list.shape',all_fenchen_list.shape)
            print('all_gouzao_list.shape',all_gouzao_list.shape)
            gra_data  = torch.stack((all_fenchen_list,all_fengsu_list,all_wendu_list,all_hengduan_list,all_juejinjc_list,all_huifen_list,all_huifafen_list,all_shuifen_list,all_qingjiao_list,all_gouzao_list,all_yingdu_list),dim=1)# [16, 9, 16]
            print('gra_data.shape',gra_data.shape)
            # print(grap_dim)
            # print('all_hengduan_list.shape',all_hengduan_list.shape) torch.Size([16, 16, 1]
            all_target_list = date_item['all_target_list'].transpose(1,2).to(device)
            target = date_item['target']
            src_mask = date_item['src_mask'].to(device)
            print(src_mask.shape)
            print(all_target_list[0])
            # tras_martix = tras_martix.reshape(4,5)
            # print(tras_martix)
            # print(src_mask.shape)
            outputs = _model(gra_data).to(device)
            print(outputs[0])
            print('outputs.shape',outputs.shape)
            print('all_target_list.shape',all_target_list.shape)
            output = criterion(all_target_list, outputs).to(device)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            acc=0
            items+=1
            total_loss+=output.item()
            print('Epoch {}, Step {} Train Loss {} '.format(Echop, items, output.item()) )
        # torch.save(_model,'.\model\model_'+str(80+Echop)+".pt")
        loss_data.append(total_loss)
        print('Epoch {},  Totle Loss {}'.format(Echop, total_loss) )
        
    torch.save(_model,".\model\model_tcn.pt")
    plt.plot(np.arange(0,Echop+1),loss_data)
    print(loss_data)
    df = pd.DataFrame(loss_data, columns=['loss_data'])
    df['epoch']=np.arange(0,Echop+1)
    df.to_excel("tcn_loss_guiyi_200.xlsx", index=False)
    plt.show()
def vaildation():
    return
# 4136.919990539551    4097.382328033447   4037.812355041504  3917.7808723449707 3695.9693603515625  3572.4541578292847  3548.8950634002686  3520.985252380371 3509.5440545082092
# def plot_time_series_class(data, class_name, ax, n_steps=5): 
#     """
#     param data:数据
#     param class_name: 不同心跳类名
#     param ax:画布
#     """
#     time_series_df = pd.DataFrame(data)
#     # 平滑时间窗口
#     smooth_path = time_series_df.rolling(n_steps).mean()
#     # 路径偏差
#     path_deviation = 1.5*time_series_df.rolling(n_steps).std()
#     print(smooth_path)
#     # 以正负偏差上下定义界限
#     under_line = (smooth_path - path_deviation)[0]
#     over_line = (smooth_path + path_deviation)[0]
#     # 绘制平滑曲线
#     ax.plot(smooth_path, linewidth=2)
#     ax.fill_between(
#       path_deviation.index,
#       under_line,
#       over_line,
#       alpha=.125)
#     ax.set_title(class_name)
def main():
    # data = pd.read_excel("./data/fenchen/31060/22-10_fengsu.xlsx",usecols=[3,5,7]).dropna(how='any')[:5192]
    # data = pd.read_excel("./data/fenchen/23-03_fengsu.xlsx",usecols=[3,5,7]).dropna(how='any')[:400]
    # data_org = pd.read_excel("./data/fenchen/23-03_fengsu_org.xlsx",usecols=[3,5,7]).dropna(how='any')[400:]
    data = pd.read_excel("./data/all/31060_sin/22-10_all_tot.xlsx",usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).dropna(how='any')[:2430]
    data_org = pd.read_excel("./data/all/31060_sin/22-10_all_org.xlsx",usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).dropna(how='any')[400:]
    print(data.info)
    fenchen_data = data['fenchen'].to_list()
    fengsu_data = data['fengsu'].to_list()
    state_data = data['状态'].to_list()
    wendu_data = data['wendu'].to_list()
    # 单个指标
    hengduan_data = data['hengduan'].to_list()
    juejinjc_data = data['juejinjc'].to_list()
    huifen_data = data['huifen'].to_list()
    huifafen_data = data['huifafen'].to_list()
    shuifen_data = data['shuifen'].to_list()
    qingjiao_data = data['qingjiao'].to_list()
    gouzao_data = data['gouzao'].to_list()
    yingdu_data = data['yingdu'].to_list()
    
    
    fenchen_data_org = data_org['fenchen'].to_list()
    fengsu_data_org = data_org['fengsu'].to_list()
    state_data_org = data_org['状态'].to_list()
    wendu_data_org = data_org['wendu'].to_list()
    # 单个指标
    hengduan_data_org = data_org['hengduan'].to_list()
    juejinjc_data_org = data_org['juejinjc'].to_list()
    huifen_data_org = data_org['huifen'].to_list()
    huifafen_data_org = data_org['huifafen'].to_list()
    shuifen_data_org = data_org['shuifen'].to_list()
    qingjiao_data_org = data_org['qingjiao'].to_list()
    gouzao_data_org = data_org['gouzao'].to_list()
    yingdu_data_org = data_org['yingdu'].to_list()
    
    org_data = deepcopy(fenchen_data)
    org_fengsu_data = deepcopy(fengsu_data)
    org_state_data = deepcopy(state_data)
    org_wendu_data = deepcopy(wendu_data)
    org_hengduan_data = deepcopy(hengduan_data)
    org_juejinjc_data = deepcopy(juejinjc_data)
    org_huifen_data = deepcopy(huifen_data)
    org_huifafen_data = deepcopy(huifafen_data)
    org_shuifen_data = deepcopy(shuifen_data)
    org_qingjiao_data = deepcopy(qingjiao_data)
    org_gouzao_data = deepcopy(gouzao_data)
    org_yingdu_data = deepcopy(yingdu_data)
    
    np.random.seed(12345)
    # for i in range(10):
    #     # print("----------")
    #     # print(len(fengsu_data))
    #     # print(len(fenchen_data))
    #     # print(len(wendu_data))
    #     # print(len(hengduan_data))
    #     # print(len(juejinjc_data))
    #     # print(len(huifen_data))
    #     # print(len(huifafen_data))
    #     # print(len(shuifen_data))
    #     # print(len(qingjiao_data))
    #     # print(len(gouzao_data))
    #     # print(len(yingdu_data))
    #     # print(len(state_data))
    #     # print("----------")
    #     white_noise = 0.2*np.random.randn(len(org_data))+0.4
    #     fenchen_data.extend(org_data+white_noise)
    #     fengsu_data.extend(org_fengsu_data)
    #     state_data.extend(org_state_data)
    #     wendu_data.extend(org_wendu_data)
        
    #     hengduan_data.extend(org_hengduan_data)
    #     juejinjc_data.extend(org_juejinjc_data)
    #     huifen_data.extend(org_huifen_data)
    #     huifafen_data.extend(org_huifafen_data)
    #     shuifen_data.extend(org_shuifen_data)
    #     qingjiao_data.extend(org_qingjiao_data)
    #     gouzao_data.extend(org_gouzao_data)
    #     yingdu_data.extend(org_yingdu_data)
        
        
    #     fenchen_data.extend(fenchen_data_org[i*len(org_data):(i+1)*len(org_data)])
    #     fengsu_data.extend(fengsu_data_org[i*len(org_fengsu_data):(i+1)*len(org_fengsu_data)])
    #     state_data.extend(state_data_org[i*len(org_state_data):(i+1)*len(org_state_data)])
    #     wendu_data.extend(wendu_data_org[i*len(org_wendu_data):(i+1)*len(org_wendu_data)])
        
    #     hengduan_data.extend(hengduan_data_org[i*len(org_hengduan_data):(i+1)*len(org_hengduan_data)])
    #     juejinjc_data.extend(juejinjc_data_org[i*len(org_juejinjc_data):(i+1)*len(org_juejinjc_data)])
    #     huifen_data.extend(huifen_data_org[i*len(org_huifen_data):(i+1)*len(org_huifen_data)])
    #     huifafen_data.extend(huifafen_data_org[i*len(org_huifafen_data):(i+1)*len(org_huifafen_data)])
    #     shuifen_data.extend(shuifen_data_org[i*len(org_shuifen_data):(i+1)*len(org_shuifen_data)])
    #     qingjiao_data.extend(qingjiao_data_org[i*len(org_qingjiao_data):(i+1)*len(org_qingjiao_data)])
    #     gouzao_data.extend(gouzao_data_org[i*len(org_gouzao_data):(i+1)*len(org_gouzao_data)])
    #     yingdu_data.extend(yingdu_data_org[i*len(org_yingdu_data):(i+1)*len(org_yingdu_data)])
    #     # print(org_data+white_noise)
    # # plt.plot(fenchen_data)
    # plt.plot(fengsu_data)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('White Noise Signal')
    # plt.show()
    print(len(fengsu_data))
    print(len(fenchen_data))
    print(len(wendu_data))
    print(len(hengduan_data))
    print(len(juejinjc_data))
    print(len(huifen_data))
    print(len(huifafen_data))
    print(len(shuifen_data))
    print(len(qingjiao_data))
    print(len(gouzao_data))
    print(len(yingdu_data))
    print(len(state_data))
    fenchen_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(fenchen_data), 10)]
    fengsu_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(fengsu_data), 10)]
    wendu_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(wendu_data), 10)]
    hengduan_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(hengduan_data), 10)]
    juejinjc_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(juejinjc_data), 10)]
    huifen_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(huifen_data), 10)]
    huifafen_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(huifafen_data), 10)]
    shuifen_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(shuifen_data), 10)]
    qingjiao_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(qingjiao_data), 10)]
    gouzao_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(gouzao_data), 10)]
    yingdu_data_chai = [fengsu_data[i : i + 10] for i in range(0, len(yingdu_data), 10)]
    scaler = MinMaxScaler() #实例化
    scalerfenchen = scaler.fit(fenchen_data_chai) #fit，在这里本质是生成min(x)和max(x)
    fenchen_data = list(np.array(scalerfenchen.transform(fenchen_data_chai)).ravel())
    
    scalerfengsu = scaler.fit(fengsu_data_chai) #fit，在这里本质是生成min(x)和max(x)
    fengsu_data = list(np.array(scalerfengsu.transform(fengsu_data_chai)).ravel())
    
    scalerwendu = scaler.fit(wendu_data_chai) #fit，在这里本质是生成min(x)和max(x)
    wendu_data = list(np.array(scalerwendu.transform(wendu_data_chai)).ravel())
    
    scalerhengduan = scaler.fit(hengduan_data_chai) #fit，在这里本质是生成min(x)和max(x)
    hengduan_data = list(np.array(scalerhengduan.transform(hengduan_data_chai)).ravel())
    
    scalerjuejinjc = scaler.fit(juejinjc_data_chai) #fit，在这里本质是生成min(x)和max(x)
    juejinjc_data = list(np.array(scalerjuejinjc.transform(juejinjc_data_chai)).ravel())
        
    scalerhuifen = scaler.fit(huifen_data_chai) #fit，在这里本质是生成min(x)和max(x)
    huifen_data = list(np.array(scalerhuifen.transform(huifen_data_chai)).ravel())
            
    scalerhuifafen = scaler.fit(huifafen_data_chai) #fit，在这里本质是生成min(x)和max(x)
    huifafen_data = list(np.array(scalerhuifafen.transform(huifafen_data_chai)).ravel())
                
    scalerssshuifen = scaler.fit(shuifen_data_chai) #fit，在这里本质是生成min(x)和max(x)
    shuifen_data = list(np.array(scalerssshuifen.transform(shuifen_data_chai)).ravel())
                    
    scalerqingjiao = scaler.fit(qingjiao_data_chai) #fit，在这里本质是生成min(x)和max(x)
    qingjiao_data = list(np.array(scalerqingjiao.transform(qingjiao_data_chai)).ravel())
            
    scalergouzao = scaler.fit(gouzao_data_chai) #fit，在这里本质是生成min(x)和max(x)
    gouzao_data = list(np.array(scalergouzao.transform(gouzao_data_chai)).ravel())
                
    scaleryingdu = scaler.fit(yingdu_data_chai) #fit，在这里本质是生成min(x)和max(x)
    yingdu_data = list(np.array(scaleryingdu.transform(yingdu_data_chai)).ravel())
    data = pd.DataFrame({'fengsu':fengsu_data,'fenchen':fenchen_data,'wendu':wendu_data,'状态':state_data,
                         'hengduan':hengduan_data,'juejinjc':juejinjc_data,'huifen':huifen_data,
                         'huifafen':huifafen_data,'shuifen':shuifen_data,'qingjiao':qingjiao_data,
                         'gouzao':gouzao_data,'yingdu':yingdu_data})
    data.reset_index(drop=True, inplace=True)
    alarm_list = []
    noalarm_list = []
    all_fenchen_list = []
    all_fengsu_list = []
    all_wendu_list = []
    all_hengduan_list = []
    all_juejinjc_list = []
    all_huifen_list = []
    all_huifafen_list = []
    all_shuifen_list = []
    all_qingjiao_list = []
    all_gouzao_list = []
    all_yingdu_list = []
    all_target_list = []
    print(data.info())
    target=[]
    count_1 = 0
    i = 0
    for i in range(len(data['fenchen'])):
        if (i+seq_len)>=(len(data['fenchen'])-10):
            break
        flag = False
        pre = 0
        cur_fenchen = [data_ for data_ in data['fenchen'][i:i+seq_len]]
        cur_fengsu = [data_ for data_ in data['fengsu'][i:i+seq_len]]
        cur_wendu = [data_ for data_ in data['wendu'][i:i+seq_len]]
        cur_hengduan = [data_ for data_ in data['hengduan'][i:i+seq_len]]
        cur_juejinjc = [data_ for data_ in data['juejinjc'][i:i+seq_len]]
        cur_huifen = [data_ for data_ in data['huifen'][i:i+seq_len]]
        cur_huifafen = [data_ for data_ in data['huifafen'][i:i+seq_len]]
        cur_shuifen = [data_ for data_ in data['shuifen'][i:i+seq_len]]
        cur_qingjiao = [data_ for data_ in data['qingjiao'][i:i+seq_len]]
        cur_gouzao = [data_ for data_ in data['gouzao'][i:i+seq_len]]
        cur_yingdu = [data_ for data_ in data['yingdu'][i:i+seq_len]]
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
        all_hengduan_list.append(cur_hengduan)
        all_juejinjc_list.append(cur_juejinjc)
        all_huifen_list.append(cur_huifen)
        all_huifafen_list.append(cur_huifafen)
        all_shuifen_list.append(cur_shuifen)
        all_qingjiao_list.append(cur_qingjiao)
        all_gouzao_list.append(cur_gouzao)
        all_yingdu_list.append(cur_yingdu)
        all_target_list.append([data_ for data_ in data['fenchen'][i+seq_len:i+seq_len+5]])
    # all_fengsu_list = scaler.fit_transform(pd.DataFrame(all_fengsu_list))
    # all_target_list = scaler.fit_transform(pd.DataFrame(all_target_list))
    data_after = pd.DataFrame({'all_fenchen_list':all_fenchen_list,'all_fengsu_list':all_fengsu_list,
                               'all_target_list':all_target_list,'all_wendu_list':all_wendu_list,
                               'all_hengduan_list':all_hengduan_list,'all_juejinjc_list':all_juejinjc_list,
                               'all_huifen_list':all_huifen_list,'all_huifafen_list':all_huifafen_list,
                               'all_shuifen_list':all_shuifen_list,'all_qingjiao_list':all_qingjiao_list,
                               'all_gouzao_list':all_gouzao_list,'all_yingdu_list':all_yingdu_list,
                               'target':target})
    print(data_after)
    print(len(target))
    print(noalarm_list)
    print(len(alarm_list))
    # plot_data = pd.Series([count_1,(len(target)-count_1)],index=['alarm', 'noalarm'])
    plot_data = pd.Series({"target":target})
    classes = data.状态.unique()
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
        datad = data[data.状态 == cls] \
        .drop(labels='状态', axis=1) \
        .to_numpy()
        print( 'daad',datad)
        # plot_time_series_class(datad, class_names[i], ax)

    fig.delaxes(axs.flat[-1])
    fig.tight_layout();

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
    
    # tras_data_alarm,test_vaildation_alarm = train_test_split(alarm_list,train_size=0.8,test_size=0.2,shuffle=False,random_state=1)
    # vaildation_data_alarm,test_data_alarm = train_test_split(test_vaildation_alarm,train_size=0.5,shuffle=False,random_state=1)
    # handler_tras_data_alarm = handler_data(data=tras_data_alarm)
    # handler_vaildation_data_alarm = handler_data(data=vaildation_data_alarm)
    # handler_test_data_alarm = handler_data(data=test_data_alarm)
    # trans_data_loader_alarm = DataLoader(handler_tras_data_alarm,batch_size=batch_size,shuffle=True)
    # vaildation_data_loader_alarm = DataLoader(handler_vaildation_data_alarm,batch_size=batch_size,shuffle=True)
    # test_data_loader_alarm = DataLoader(handler_test_data_alarm,batch_size=batch_size,shuffle=True)
    
    return trans_data_loader,vaildation_data_loader,test_data_loader,scalerfenchen
def predict(model, dataset,scalerfenchen):
    predictions, losses = [], []
    total_loss = 0
    all_data_true = []
    all_data_pred = []
    i = 0
    criterion = nn.L1Loss(reduction='none').to(device)
    with torch.no_grad():
        model.eval()
        for date_item in dataset:
            all_fenchen_list = torch.squeeze(date_item['fenchen'].transpose(1,2).to(device))
            all_fengsu_list = torch.squeeze(date_item['fengsu'].transpose(1,2).to(device))
            all_wendu_list = torch.squeeze(date_item['wendu'].transpose(1,2).to(device))
            all_hengduan_list = torch.squeeze(date_item['hengduan'].transpose(1,2).to(device))
            all_juejinjc_list = torch.squeeze(date_item['juejinjc'].transpose(1,2).to(device))
            all_huifen_list = torch.squeeze(date_item['huifen'].transpose(1,2).to(device))
            all_huifafen_list = torch.squeeze(date_item['huifafen'].transpose(1,2).to(device))
            all_shuifen_list = torch.squeeze(date_item['shuifen'].transpose(1,2).to(device))
            all_qingjiao_list = torch.squeeze(date_item['qingjiao'].transpose(1,2).to(device))
            all_gouzao_list = torch.squeeze(date_item['gouzao'].transpose(1,2).to(device))
            all_yingdu_list = torch.squeeze(date_item['yingdu'].transpose(1,2).to(device))
            print('all_fenchen_list.shape',all_fenchen_list.shape)
            print('all_gouzao_list.shape',all_gouzao_list.shape)
            gra_data  = torch.stack((all_fenchen_list,all_fengsu_list,all_wendu_list,all_hengduan_list,all_juejinjc_list,all_huifen_list,all_huifafen_list,all_shuifen_list,all_qingjiao_list,all_gouzao_list,all_yingdu_list),dim=1)# [16, 9, 16]
            print('gra_data.shape',gra_data.shape)
            # print(grap_dim)
            # print('all_hengduan_list.shape',all_hengduan_list.shape) torch.Size([16, 16, 1]
            all_target_list = date_item['all_target_list'].transpose(1,2).to(device)
            target = date_item['target']
            src_mask = date_item['src_mask'].to(device)
            all_data_true.extend(all_target_list[:,0,:].T[0].cpu().numpy().tolist())
            print(src_mask.shape)
            print(all_target_list[0])
            outputs = _model(gra_data).to(device)
            print(outputs[0])
            print('outputs.shape',outputs.shape)
            print('all_target_list.shape',all_target_list.shape)
            output = criterion(all_target_list, outputs).to(device)
            all_data_pred.extend(outputs[:,0, :].T[0].cpu().numpy().tolist())
            # print(outputs)
            output = criterion(all_target_list, outputs).to(device)
            output = output.reshape(output.shape[0],output.shape[1]).sum(dim=1)
            print('Train Loss {} '.format(torch.sum(output)) )
            # predictions.append(outputs.cpu().numpy().flatten())
            # print(output.item().shape)
            losses.extend(torch.tensor(output))
            # for i in range(len(all_fenchen_list)):
            #     test_0 = all_fenchen_list[i].reshape(all_fenchen_list[i].shape[0]*all_fenchen_list[i].shape[1])
            #     target_0 = all_target_list[i].reshape( all_target_list[i].shape[0]*all_target_list[i].shape[1])
            #     output_0 = outputs[i].reshape(outputs[i].shape[0],outputs[i].shape[1])
            #     test_0 =  test_0.tolist()
            #     pred = deepcopy(test_0)
            #     test_0.extend(target_0)
            #     pred.extend(output_0)
            #     print(output_0)
            #     plt.title('实际值与预测值')
            #     plt.xlabel("时间步数")
            #     plt.ylabel("值")
            #     plt.plot(np.arange(0,len(test_0)),test_0,'rs:',label='实际值')
            #     plt.plot(np.arange(0,len(pred)),pred,'m<:',label='预测值')
            #     plt.legend()
            #     plt.show()
                
                
            plt.title('实际值与预测值')
            plt.xlabel("时间步数")
            plt.ylabel("值")
            if len(all_data_true)%10==0:
                plt.plot(np.arange(0,len(all_data_true)),list(np.array(scalerfenchen.inverse_transform([all_data_true[i : i + 10] for i in range(0, len(all_data_true), 10)])).ravel()),'rs:',label='实际值')
            # plt.plot(np.arange(0,len(all_target_list[:,0,:].T[0].numpy().tolist())),all_target_list[:,0,:].T[0].numpy().tolist(),'rs:',label='实际值')
                plt.plot(np.arange(0,len(all_data_pred)),list(np.array(scalerfenchen.inverse_transform([all_data_pred[i : i + 10] for i in range(0, len(all_data_pred), 10)])).ravel()),'m<:',label='预测值')
            # plt.plot(np.arange(0,len(outputs[:,0, :].T[0].numpy().tolist())),outputs[:,0, :].T[0].numpy().tolist(),'m<:',label='预测值')
                plt.legend()
                plt.show()
        result = pd.DataFrame({"true":all_data_true,"pred":all_data_pred})
        data_result_write = pd.ExcelWriter("./tcn_predict_result.xlsx")
        result.to_excel(data_result_write)
        data_result_write.close()
            # print(losses)
            # total_loss+=output.item()
        # print("loss",total_loss)
    return predictions, losses
if __name__=='__main__':
    trans_data_loader,vaildation_data_loader,test_data_loader,scalerfenchen=main()
    # print(len(trans_data_loader))
    # train(trans_data_loader)
    _, losses = predict(_model, trans_data_loader,scalerfenchen)
    # _, losses = predict(_model, trans_data_loader_alarm)
    # _, losses = predict(_model, vaildation_data_loader_noalarm)
    # _, losses = predict(_model, trans_data_loader)
    # _, losses = predict(_model, trans_data_loader)
    # _alarm, losses_alarm = predict(_model, test_data_loader_alarm)
    
    # threshold = 5
    # correct = sum(l <= threshold for l in losses)
    # print(f'Correct noalarm predictions: {correct}/{len(losses)}')
    # correct_alarm = sum(l >= threshold for l in losses_alarm)
    # print(f'Correct alarm predictions: {correct_alarm}/{len(losses_alarm)}')
    # sns.displot(torch.tensor(losses), bins=50, kde=True)
    # plt.show()
    # sns.displot(torch.tensor(losses_alarm), bins=50, kde=True)
    # plt.show()
    # plt.legend()