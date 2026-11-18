"""
2025/7/7
复现Hnet
"""
from __future__ import print_function, division
import sys
import os
import torch
import argparse
from models import Hynet
from utils.my_loss import MyLoss,MyLoss_MSE,evaluate_ssim
import time
from utils.metric import complex_psnr
from tensorboardX import SummaryWriter
import gc
import torchvision.utils as utils
from scipy.io import loadmat
from utils.fftc import *
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.ADMM import SOFT
from models import get_net
from torch.nn import init
from complex_layers.CV_Net import CVnet

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

    def entropy_zw(X):
        X1 = torch.sum(torch.abs(X))
        eps = torch.finfo(X.dtype).eps  # 防止除0和log(0)
        P = torch.abs(X) / (X1 + eps)
        E = -torch.sum(P * torch.log(P + eps))
        return E
    
    # get the dft_matrix
    def generate_Fa_matrix(N, P):
        omega = np.exp(-1j * 2 * np.pi / P)  # 计算ω值
        # 创建指数矩阵
        rows = np.arange(N).reshape(-1, 1)
        cols = np.arange(P)
        exponents = rows * cols  # 指数部分 (k-1)(l-1)
        # 计算矩阵元素
        F_a = omega ** exponents
        return torch.tensor(F_a, dtype = torch.complex64)
    # 更高效的各向同性TV实现
    def tv(image):
        """更高效的各向同性TV计算"""
        y = image[:, :, :-1, :] - image[:, :, 1:, :]
        x = image[:, :, :, :-1] - image[:, :, :, 1:]
        TV_loss = (torch.sum(torch.pow(x, 2)) + torch.sum(torch.pow(y, 2))) / x.numel()
        return TV_loss
    
    def hessian(image):
        y = image[:, :, :-1, :] - image[:, :, 1:, :]
        x = image[:, :, :, :-1] - image[:, :, :, 1:]
        xx = x[:, :, :, :-1] - x[:, :, :, 1:]
        yy = y[:, :, :-1, :] - y[:, :, 1:, :]
        xy = y[:, :, :, :-1] - y[:, :, :, 1:]
        yx = x[:, :, :-1, :] - x[:, :, 1:, :]
        Hess_loss = (torch.sum(torch.pow(xx, 2)) + torch.sum(torch.pow(yy, 2)) + torch.sum(torch.pow(yx, 2)) + torch.sum(torch.pow(xy, 2))) / xx.numel()
        return Hess_loss
    def generate_specified_blocks(N, blocks):
        sequence = []
        for start, end in blocks:
            start = max(0, min(start, N-1))
            end = max(0, min(end, N))
            for i in range(start, end):
                sequence.append(i)
        return sequence
    
    if not os.path.exists(args.outf):
        os.mkdir(args.outf)
    ###############################################################################
    # mask
    ###############################################################################
    # ### 定义观测矩阵（部分傅里叶矩阵）
    # import random

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
    
    # np.random.seed(2)  # 保证结果可重复
    # random.seed(2)  # 可以是任意整数
    # torch.manual_seed(0)    


    criterion = MyLoss()

    model_c = Hynet.CSWinTransformerNet(45).cuda()

    num_samples = 1000
    img_size = 128
    min_points = 100
    max_points = 200
    batch_size = 32
    num_epochs = 30
    lr = 0.001
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from torch.utils.data import Dataset, DataLoader
    from utils.dataset import ComplexImageDataset
    from sklearn.model_selection import train_test_split
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
    # Adam optimizer
    ###############################################################################
    
    ###############################################################################
    # self-define loss
    ###############################################################################
    criterion1 = MyLoss().cuda()
    criterion2 = MyLoss_MSE().cuda()
    ###############################################################################
    # train
    ###############################################################################
    print("start training...")

    params = []
    params = [
        {'params': model_c.parameters()},
    ]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    batch_idx = 0
    I = torch.eye(grid_size).cuda()
    start_time = time.time()

    model_c.train()
    for epoch in range(args.num_epoch + 1):
        mask_size = 64
        index = random.sample(range(Na), mask_size)
        mask = torch.ones(Na, dtype=bool)  # 初始化为全 True
        mask[index] = False  # 将要删除的行标记为 False    
        Fr_t, Fa_t = torch.tensor(Fr,dtype=torch.complex64).cuda(),torch.tensor(Fa,dtype=torch.complex64).cuda()[:,mask]
        for label_new in train_loader:
            label_new = label_new.cuda()
            Z = torch.zeros_like(label_new).cuda()
            U = torch.zeros_like(label_new).cuda()
            echo = torch.einsum('km,ijml,ln->ijkn', Fr_t,label_new, Fa_t)
            echo += (0.01+0.05*np.random.rand())*torch.randn_like(echo) #加噪声
            Ahy = torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr_t.T),echo, torch.conj(Fa_t.T))
            hyper = model_c(abs(Ahy).cuda())
            for iter in range(15):
                # 前向计算
                X = Z-U - 1/(1+hyper[:,0+3*iter]).unsqueeze(1).unsqueeze(1).unsqueeze(1)*torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr_t.T),torch.einsum('km,ijml,ln->ijkn', Fr_t,Z-U, Fa_t)-echo, torch.conj(Fa_t.T))
                Z = SOFT(X + U,hyper[:,1+3*iter].unsqueeze(1).unsqueeze(1).unsqueeze(1)/10)  # 切断历史梯度
                U = U + hyper[:,2+3*iter].unsqueeze(1).unsqueeze(1).unsqueeze(1)*(X - Z)

            optimizer.zero_grad()
            loss_normal = criterion1(X, label_new)
            loss_normal.backward()
            optimizer.step()
        print(f'epoch:{epoch},train_loss:{loss_normal.data.item()}')

    torch.save(model_c.state_dict(), os.path.join(args.outf, 'HNet.pth'))