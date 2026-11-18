"""
2025/7/7
CV-RES-ADMM-Net复现代码
"""
from __future__ import print_function, division
import sys
import os
import torch
import argparse
from utils.my_loss import MyLoss
import time
from scipy.io import loadmat
from utils.fftc import *
import numpy as np
import random
from torch.nn import init
from complex_layers.CV_Net import CVnet
from torch.utils.data import Dataset, DataLoader
from utils.dataset import ComplexImageDataset
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


if __name__ == '__main__':

    ###############################################################################
    # parameters
    ###############################################################################
    parser = argparse.ArgumentParser(description='main ')
    parser.add_argument('--data_dir', default='data_sat/', type=str,
                        help='directory of data')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--moshi', default='supervised', type=str, help='train_moshi(supervised/unsupervised)')
    parser.add_argument('--num_epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='logs_isarnet', help='path of log files')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ##############################################################################
    if not os.path.exists(args.outf):
        os.mkdir(args.outf)

    grid_size = 128
    Na = 128
    Nr = 128
    # 系统参数
    fc = 10e9
    c = 3e8
    f = np.linspace(0,1e9,Nr)
    PRF = 1
    omega = 0.1
    T = 1
    xp = np.linspace(0,19.2,grid_size)
    yq = np.linspace(0,19.2,grid_size)
    tr = omega* np.linspace(0,T,Na)/PRF
    Fr = 1/np.sqrt(grid_size)*np.exp(1j*4*np.pi*f[:,None]@xp[None,:]/c)
    Fa = 1/np.sqrt(grid_size)*np.exp(1j*4*np.pi*fc*yq[:,None]@tr[None,:]/c)
    
    
    model_c = CVnet(1,32).cuda()
    init_weights(model_c, init_type='normal',init_gain=0.02)
    
    # 数据集参数
    num_samples = 1000
    img_size = 128
    min_points = 100
    max_points = 300
    batch_size = 32
    num_epochs = 30
    lr = 0.001
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成数据集
    full_dataset = ComplexImageDataset(num_samples, img_size, min_points, max_points)
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(range(num_samples), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ###############################################################################
    # self-define loss
    ###############################################################################
    criterion1 = MyLoss().cuda()
    ###############################################################################
    # train
    ###############################################################################
    print("start training...")
    total_loss_org = 0
    train_batches = 0
    train_psnr = 0
    rho0 = torch.tensor([1.0]).cuda()
    rho0.requires_grad = True
    params = []
    params = [
        {'params': model_c.parameters()},
        {'params': [rho0]}  # 可调大一些
    ]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    batch_idx = 0
    I = torch.eye(grid_size).cuda()
    start_time = time.time()

    model_c.train()
    for epoch in range(args.num_epoch + 1):
        mask_size = 40 + random.randint(1, 60)
        index = random.sample(range(Na), mask_size)
        mask = torch.ones(Na, dtype=bool)  # 初始化为全 True
        mask[index] = False  # 将要删除的行标记为 False
        Fr_t, Fa_t = torch.tensor(Fr,dtype=torch.complex64).cuda(),torch.tensor(Fa,dtype=torch.complex64).cuda()[:,mask]
        for label_new in train_loader:
            label_new = label_new.cuda()
            Z = torch.zeros_like(label_new).cuda()
            U = torch.zeros_like(label_new).cuda()
            data_echo = torch.einsum('km,ijml,ln->ijkn', Fr_t,label_new,Fa_t)
            data_echo += (0.01+0.05*np.random.rand())*torch.randn_like(data_echo)  #加噪声
            for iter in range(6):
                # 前向计算
                X = Z-U - 1/(1+rho0)*torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr_t.T),torch.einsum('km,ijml,ln->ijkn', Fr_t,Z-U, Fa_t)-data_echo, torch.conj(Fa_t.T))
                Z = model_c(X + U.detach())  # 切断历史梯度
                U = U + X.detach() - Z.detach()  # 不追踪这步梯度

            optimizer.zero_grad()
            loss_normal = criterion1(X, label_new)
            loss_normal.backward()
            optimizer.step()
        print(f'epoch:{epoch},train_loss:{loss_normal.data.item()}')
    torch.save(model_c.state_dict(), os.path.join(args.outf, 'CV_Net1.pth'))