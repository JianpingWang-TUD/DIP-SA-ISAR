"""
    基于DIP的ISAR成像
    2025/7/15 建立完整的ISAR前向模型
    预训练模型
"""
from __future__ import print_function, division
import sys
import os
import torch
import argparse
from utils.unet import Unet,MediumUNet
from utils.my_loss import MyLoss
import time
from tensorboardX import SummaryWriter
import gc
import torchvision.utils as utils
from scipy.io import loadmat
from utils.fftc import *
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import random
from Main_simulation import init_weights
from utils.common_utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':

    ###############################################################################
    # parameters
    ###############################################################################
    parser = argparse.ArgumentParser(description='main ')
    parser.add_argument('--data_dir', default='data_sat/', type=str,
                        help='directory of data')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--moshi', default='supervised', type=str, help='train_moshi(supervised/unsupervised)')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='logs_isarnet', help='path of log files')
    args = parser.parse_args()

    ###############################################################################
    # callable methods
    ###############################################################################
    def adjust_learning_rate(opt, epo, lr):
        """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
        lr = lr * (0.5 ** (epo // 2000))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    if not os.path.exists(args.outf):
        os.mkdir(args.outf)
    ###############################################################################
    # mask
    ###############################################################################
    # ### 定义观测矩阵（部分傅里叶矩阵）
    # 目标成像网格
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
    
    Fr, Fa = torch.tensor(Fr,dtype=torch.complex64).cuda(),torch.tensor(Fa,dtype=torch.complex64).cuda()
    
    from torch.utils.data import Dataset, DataLoader
    from utils.dataset import ComplexImageDataset
    from sklearn.model_selection import train_test_split
    
    num_samples = 1000
    img_size = 128
    min_points = 50
    max_points = 200
    batch_size = 32
    num_epochs = 30
    lr = 0.001
    full_dataset = ComplexImageDataset(num_samples, img_size, min_points, max_points)
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(range(num_samples), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MediumUNet(2,2).cuda()
    init_weights(model, init_type='normal',init_gain=0.02)
    ###############################################################################
    # Adam optimizer
    ###############################################################################
    params = []
    params = [
        {'params': model.parameters()},
    ]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    ###############################################################################
    # self-define loss
    ###############################################################################
    criterion1 = MyLoss().cuda()
    writer = SummaryWriter(args.outf)
    ###############################################################################
    # train
    ###############################################################################
    print("start training...")
    start_time = time.time()
    total_loss_org = 0
    train_batches = 0
    train_psnr = 0
    model.train()
    mask_size = 64
    index = random.sample(range(Na), mask_size)
    mask = torch.ones(Na, dtype=bool)  # 初始化为全 True
    mask[index] = False  
    for epoch in range(args.num_epoch + 1):
        adjust_learning_rate(optimizer, epoch, lr=0.00001)
        for label_new in train_loader:
            label_new = label_new.cuda()
            kspace_out_f = torch.einsum('km,ijml,ln->ijkn', Fr,label_new,Fa)[:,:,:,mask]
            Ahy = torch.einsum('km,ijml,ln->ijkn',torch.conj(Fr.T),kspace_out_f,torch.conj(Fa[:,mask].T))
            input = 0.1*torch.zeros(label_new.shape[0],2,grid_size,grid_size)
            input[:,0] = Ahy.real.squeeze()
            input[:,1] = Ahy.imag.squeeze()
            output = model(input.cuda())
            loss_normal = criterion1(torch.complex(output[:,0],output[:,1]).unsqueeze(1), label_new)
            loss_normal.backward()
            optimizer.step()
            total_loss_org += loss_normal.data.item()
            train_batches += 1
        print(f'epoch:{epoch},train_loss:{loss_normal.data.item()}')
        plt.figure(figsize=(10, 10))
        plt.imshow(20*np.log10(np.abs(torch.complex(output[:,0],output[:,1]).cpu().detach().numpy())[0]),vmin = -30, vmax = 0)  # 显示幅度
        plt.colorbar(label='Magnitude')
        plt.savefig('test3.png')
        plt.close()
    torch.save(model.state_dict(), os.path.join(args.outf, '1.pth'))