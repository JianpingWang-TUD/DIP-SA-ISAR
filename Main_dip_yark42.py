"""
    基于DIP的ISAR成像
    2025/7/15 建立完整的ISAR前向模型
    2025/7/26
    yark42 测试的专门代码
"""
from __future__ import print_function, division
import sys
import os
import torch
import argparse
from utils.unet import Unet,MediumUNet,ExtraDeepUNet,SmallUNet,Resnet
from utils.my_loss import MyLoss,MyLoss_MSE,evaluate_ssim,correlation_coefficient,grad_ssim,iw_ssim
import time
from utils.metric import complex_psnr
from tensorboardX import SummaryWriter
import gc
import torchvision.utils as utils
from scipy.io import loadmat, savemat
from utils.fftc import *
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.ADMM import ADMM,ADMM_2D,ADMM_2D_Error,ADMM_2D_weight,ISTA_2d
from models import get_net
from torch.nn import init
from Main_simulation import init_weights
from utils.common_utils import *
from Main_simulation import generate_specified_blocks
from models import All_Net
from utils.plot import plot_cv_fuc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from utils.stopping_criterion import EarlyStop, myMetric



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
    parser.add_argument('--num_epoch', default=20000, type=int, help='number of epochs')
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
    def isotropic_tv(image):
        """更高效的各向同性TV计算"""
        dx = torch.diff(image, axis=0)
        dy = torch.diff(image, axis=1)
        # 填充使dx和dy尺寸相同
        dx = torch.vstack((dx, torch.zeros((1, dx.shape[1])).cuda()))
        dy = torch.hstack((dy, torch.zeros((dy.shape[0], 1)).cuda()))
        return torch.sum(torch.sqrt(dx**2 + dy**2+1e-10))
    
    def add_awgn_noise_torch(signal, snr_db):
    # 信号功率（复信号模平方平均）
        signal_power = torch.mean(torch.abs(signal) ** 2)

        # SNR 换算成线性值
        snr_linear = 10 ** (snr_db / 10)

        # 噪声功率 = 信号功率 / SNR
        noise_power = signal_power / snr_linear

        # 生成复数噪声（实部与虚部为独立的 N(0, noise_power / 2)）
        noise_real = torch.randn_like(signal.real) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(signal.imag) * torch.sqrt(noise_power / 2)
        complex_noise = noise_real + 1j * noise_imag
        return complex_noise    
    
    if not os.path.exists(args.outf):
        os.mkdir(args.outf)
    ###############################################################################
    # mask
    ###############################################################################
    # ### 定义观测矩阵（部分傅里叶矩阵）
    # 目标成像网格

    
    grid_size = 256
    Na = 256
    Nr = 256
    # 系统参数
    fc = 5.52e9
    c = 3e8
    f = np.linspace(0,5e8,Nr)
    PRF = 1
    omega = 0.05
    T = 1
    xp = np.linspace(0,19.2*4,grid_size)
    yq = np.linspace(0,19.2*4,grid_size)
    tr = omega* np.linspace(0,T,Na)/PRF
    Fr = 1/np.sqrt(grid_size)*np.exp(1j*4*np.pi*f[:,None]@xp[None,:]/c)
    Fa = 1/np.sqrt(grid_size)*np.exp(1j*4*np.pi*fc*yq[:,None]@tr[None,:]/c)    


    np.random.seed(2)  # 保证结果可重复
    random.seed(2)  # 可以是任意整数
    torch.manual_seed(0)    

    grid = loadmat('data_sat/Yak42_image.mat')['y_d']

    if grid.shape[0] == 1:
        grid = grid[0]/np.max(np.max(abs(grid)))
    else:
        grid = grid/np.max(np.max(abs(grid)))
    
    echo = Fr@ grid@ Fa
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(grid), cmap='gray')  # 显示幅度
    plt.imshow(20*np.log10(np.abs(grid)+1e-10),vmin = -30, vmax = 0, cmap='hot')  # 显示幅度
    # plt.colorbar(label='Magnitude')
    plt.savefig('test.png')
    
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(np.conj(Fr.T)@echo@np.conj(Fa.T)), cmap='hot')  # 显示幅度
    plt.colorbar(label='Magnitude')
    plt.savefig('test1.png')
    
    criterion = MyLoss()
    sample_mode = 'sparse' # 'Gap'
    # 稀疏成像模式
    if sample_mode == 'sparse':
        mask_size = 128+25
        index = random.sample(range(Na), mask_size)
    else:
        index = generate_specified_blocks(Na, [(5,60),(120,180),(210,240)])

    mask = torch.ones(Na, dtype=bool)  # 初始化为全 True
    mask[index] = False  # 将要删除的行标记为 False

    # 添加噪声
    echo_e_t = torch.tensor(echo, dtype = torch.complex64)
    noise_a = 0.01
    noise = noise_a*torch.randn_like(echo_e_t)
    echo_sparse = (echo_e_t+noise)[:,mask]
    SNR = 10*torch.log10(torch.mean(abs(echo_e_t)**2)/torch.mean(abs(noise)**2))
    print(f"信号SNR: {SNR.cpu().numpy().item():.2f} dB")
    
    # 估计噪声系数
    maiya = np.abs(torch.tensor(np.conj(Fr.T), dtype = torch.complex64)@(echo_sparse))
    xishu = ((torch.mean(maiya[:50])+torch.mean(maiya[200:]))/2/torch.mean(maiya[80:170])).numpy()  # 大概估计噪声强度
    echo_show = np.abs(torch.tensor(np.conj(Fr.T), dtype = torch.complex64)@(echo_e_t+noise))
    echo_show[:,index]=0
    
    plt.figure(figsize=(10,10))
    plt.imshow(echo_show,cmap='hot')
    plt.xticks(np.arange(0, grid_size, 50))
    plt.tick_params(labelsize=30)  # 设置字体大小
    plt.yticks(np.arange(0, grid_size, 50))
    plt.tick_params(labelsize=30)  # 设置字体大小
    plt.ylabel("Range cell",fontsize=40)
    plt.xlabel("Cross-range cell",fontsize=40)
    plt.tight_layout()# 显示幅度
    plt.savefig('test_1.png')
    

    # RD 成像
    Fr, Fa = torch.tensor(Fr,dtype=torch.complex64).cuda(),torch.tensor(Fa,dtype=torch.complex64).cuda()
    start_time = time.time()
    RD = (torch.conj(Fr.T)@echo_sparse.cuda()@torch.conj(Fa[:,mask].T)).cpu()
    RD = RD/torch.max(abs(RD))
    print(time.time()-start_time,'s')
    plt.figure(figsize=(10, 10))
    plt.imshow(20*np.log10(abs(RD.numpy())),vmin = -30, vmax = 0, cmap='hot')  # 显示幅度
    plt.colorbar(label='Magnitude')
    plt.savefig('test1.png')

    Psnr_RD = complex_psnr(abs(RD).numpy(), abs(grid),
                                        peak='normalized')
    Nrmse_RD = criterion(abs(RD), torch.tensor(abs(grid)))
    Ssim_RD = evaluate_ssim(abs(RD).numpy(), abs(grid))
    Ent_RD = entropy_zw(abs(RD))
    CC_RD = correlation_coefficient(abs(RD).cpu().detach(), torch.tensor(abs(grid)))
    grad_ssim_RD = grad_ssim(abs(RD).numpy(), abs(grid))
    initw_ssim_RD = iw_ssim(abs(RD).numpy(), abs(grid))
    print(f"RD_PSNR: {Psnr_RD:.4f} "
      f"RD_NRMSE: {Nrmse_RD.numpy():.4f} "
      f"RD_SSIM: {Ssim_RD:.4f} "
      f"RD_ent: {Ent_RD.numpy():.4f} "
      f"RD_CC: {CC_RD:.4f}",
      f"RD_giwSSIM: {(grad_ssim_RD+initw_ssim_RD)/2:.4f} ")
    
    # ADMM 方法
    start_time = time.time()
    Rec_ADMM = ADMM_2D(Fr,Fa[:,mask],echo_sparse.cuda(),noise_a,2,1000)
    Rec_ADMM = Rec_ADMM/torch.max(abs(Rec_ADMM))
    print(time.time()-start_time,'s')
    plt.figure(figsize=(10, 10))
    plt.imshow(20*np.log10(abs(Rec_ADMM+1e-10).cpu().numpy()),vmin = -30, vmax = 0, cmap='hot')  # 显示幅度
    plt.savefig('test_2.png')   
    train_psnr_admm = complex_psnr(abs(Rec_ADMM).cpu().numpy(), abs(grid),
                                        peak='normalized')
    NRMSE_admm = criterion(abs(Rec_ADMM).cpu().detach(), torch.tensor(abs(grid)))
    Ssim_admm = evaluate_ssim(abs(Rec_ADMM).cpu().detach().numpy(), abs(grid))
    ent_admm = entropy_zw(abs(Rec_ADMM).cpu().detach())
    CC_admm = correlation_coefficient(abs(Rec_ADMM).cpu().detach(), torch.tensor(abs(grid)))
    grad_ssim_admm = grad_ssim(abs(Rec_ADMM).cpu().detach().numpy(), abs(grid))
    initw_ssim_admm = iw_ssim(abs(Rec_ADMM).cpu().detach().numpy(), abs(grid))
    print(f"ADMM_PSNR: {train_psnr_admm:.4f} "
      f"ADMM_NRMSE: {NRMSE_admm.numpy():.4f} "
      f"ADMM_SSIM: {Ssim_admm:.4f} "
      f"ADMM_ent: {ent_admm.numpy():.4f} "
      f"ADMM_CC: {CC_admm:.4f}",
      f"ADMM_giwSSIM: {(grad_ssim_admm+initw_ssim_admm)/2:.4f} ")
    
    start_time = time.time()
    Rec_ISTA = ISTA_2d(Fr,Fa[:,mask],(echo_sparse).cuda(),noise_a,0.3,1000)
    Rec_ISTA = Rec_ISTA/torch.max(abs(Rec_ISTA))
    print(time.time()-start_time,'s')
    Psnr_ista = complex_psnr(abs(Rec_ISTA).cpu().detach().numpy(), abs(grid),peak='normalized')
    NRMSE_ista  = criterion(abs(Rec_ISTA).cpu().detach(), torch.tensor(abs(grid)))
    Ssim_ista = evaluate_ssim(abs(Rec_ISTA).cpu().detach().numpy(), abs(grid))
    ent_ISTA = entropy_zw(abs(Rec_ISTA).cpu().detach())
    CC_ista = correlation_coefficient(abs(Rec_ISTA).cpu().detach(), torch.tensor(abs(grid)))
    grad_ssim_ista = grad_ssim(abs(Rec_ISTA).cpu().detach().numpy(), abs(grid))
    initw_ssim_ista = iw_ssim(abs(Rec_ISTA).cpu().detach().numpy(), abs(grid))
    print(f"ISTA_PSNR: {Psnr_ista:.4f} "
        f"ISTA_NRMSE: {NRMSE_ista.numpy():.4f} "
        f"ISTA_SSIM: {Ssim_ista:.4f} "
        f"ISTA_ent: {ent_ISTA.numpy():.4f} "
        f"ISTA_CC: {CC_ista:.4f}",
        f"ISTA_giwSSIM: {(grad_ssim_ista+initw_ssim_ista)/2:.4f} ")
     
    # Rec_CVres方法
    start_time = time.time()
    Rec_CVres = All_Net.CVnet_test(Fr,Fa[:,mask],(echo_sparse).unsqueeze(0).unsqueeze(0).cuda(),RD).squeeze().cpu().detach()
    Rec_CVres = Rec_CVres/torch.max(abs(Rec_CVres))
    print(time.time()-start_time,'s')
    Psnr_CVres = complex_psnr(abs(Rec_CVres).numpy(), abs(grid),
                                        peak='normalized')
    Nrmse_CVres = criterion(abs(Rec_CVres), torch.tensor(abs(grid)))
    Ssim_CVres = evaluate_ssim(abs(Rec_CVres).numpy(), abs(grid))
    ent_CVres = entropy_zw(abs(Rec_CVres))
    CC_CVres = correlation_coefficient(abs(Rec_CVres).cpu().detach(), torch.tensor(abs(grid)))
    grad_ssim_CVres = grad_ssim(abs(Rec_CVres).cpu().detach().numpy(), abs(grid))
    initw_ssim_CVres = iw_ssim(abs(Rec_CVres).cpu().detach().numpy(), abs(grid))
    print(f"CVres_PSNR: {Psnr_CVres:.4f} "
        f"CVres_NRMSE: {Nrmse_CVres.numpy():.4f} "
        f"CVres_SSIM: {Ssim_CVres:.4f} "
        f"CVres_ent: {ent_CVres.numpy():.4f} "
        f"CVres_CC: {CC_CVres:.4f}",
        f"CVres_giwSSIM: {(grad_ssim_CVres+initw_ssim_CVres)/2:.4f} ")
    
    # Rec_Hnet方法
    start_time = time.time()
    Rec_Hnet = All_Net.Hnet_test(Fr,Fa[:,mask],(echo_sparse).unsqueeze(0).unsqueeze(0).cuda(),RD).squeeze().cpu().detach()
    Rec_Hnet = Rec_Hnet/torch.max(abs(Rec_Hnet))
    print(time.time()-start_time,'s')
    Psnr_Hnet = complex_psnr(abs(Rec_Hnet).numpy(), abs(grid),
                                        peak='normalized')
    Nrmse_Hnet = criterion(abs(Rec_Hnet), torch.tensor(abs(grid)))
    Ssim_Hnet = evaluate_ssim(abs(Rec_Hnet).numpy(), abs(grid))
    Ent_Hnet = entropy_zw(abs(Rec_Hnet))
    CC_Hnet = correlation_coefficient(abs(Rec_Hnet).cpu().detach(), torch.tensor(abs(grid)))
    grad_ssim_Hnet = grad_ssim(abs(Rec_Hnet).cpu().detach().numpy(), abs(grid))
    initw_ssim_Hnet = iw_ssim(abs(Rec_Hnet).cpu().detach().numpy(), abs(grid))
    print("Hnet_PSNR: {:.4f}".format(Psnr_Hnet),
        "Hnet_NRMSE: {:.4f}".format(Nrmse_Hnet.numpy()),
        "Hnet_SSIM: {:.4f}".format(Ssim_Hnet),
        "Hnet_ent: {:.4f}".format(Ent_Hnet.numpy()),
        "HNet_CC: {:.4f}".format(CC_Hnet),
        "Hnet_giwSSIM: {:.4f}".format((grad_ssim_Hnet+initw_ssim_Hnet)/2))
    
    
    
    # 引入DIP网络模型
    model = MediumUNet(2,2).cuda()
    init_weights(model, init_type='normal',init_gain=0.02)
    
    ###############################################################################
    # Adam optimizer
    ###############################################################################
    params = []
    params = [
        {'params': model.parameters()},
    ]
    optimizer = torch.optim.Adam(params, lr=0.001)
    model.load_state_dict(torch.load(os.path.join(args.outf, '2.pth'), map_location=lambda storage, loc: storage))
    ###############################################################################
    # self-define loss
    ###############################################################################
    criterion1 = MyLoss().cuda()
    writer = SummaryWriter(args.outf)
    # loss_admm = criterion1(image_admm,label)
    ###############################################################################
    # train
    ###############################################################################
    print("start training...")
    start_time = time.time()
    total_loss_org = 0
    train_batches = 0
    train_psnr = 0

    label_new = torch.tensor(grid,dtype=torch.complex64).unsqueeze(0).unsqueeze(1)
    data_kspace_label = (echo_sparse).unsqueeze(0).unsqueeze(1)
    input = 0.1*torch.rand_like(torch.zeros(1,2,grid_size,grid_size))
    AHy = torch.conj(Fr.T).cpu()@(echo_sparse)@torch.conj(Fa[:,mask].T).cpu()
    AHy = AHy / torch.max(abs(AHy))
    input[:,0] = AHy.real
    input[:,1] = AHy.imag
    # lamda = torch.mean(abs(AHy))*grid_size**2
    batch_idx = 0
    log_ent = []
    log_psnr = []
    variance_history  = []
    x_axis = []
    buffer_size = 100
    earlystop = EarlyStop(size=buffer_size,patience=1000)
    for epoch in range(1, args.num_epoch + 1):

        adjust_learning_rate(optimizer, epoch, lr=0.001)
        optimizer.zero_grad()
        output = model(input.cuda())
            
        
        kspace_out = torch.einsum('km,ijml,ln->ijkn', Fr, torch.complex(output[:,0],output[:,1]).unsqueeze(1),Fa)[:,:,:,mask]
        loss_normal = criterion1(kspace_out, data_kspace_label.cuda())+0*(3e-4*xishu+5e-5)*torch.sum(torch.log10(abs(torch.complex(output[:,0],output[:,1])[0])+5e-2)) +0.0*(3e-4*xishu+5e-5)*isotropic_tv(abs(torch.complex(output[:,0],output[:,1])[0]))
        # +0.0/lamda*torch.sum(abs(torch.complex(output[:,0],output[:,1])[0])+1e-10)
        loss_normal.backward()
        optimizer.step()

        total_loss_org += loss_normal.data.item()
        train_batches += 1
        r_img_np = torch_to_np(abs(torch.complex(output[:,0],output[:,1])))
        #variance hisotry
        r_img_np = r_img_np.reshape(-1)
        earlystop.update_img_collection(r_img_np)
        img_collection = earlystop.get_img_collection()
        if len(img_collection) == buffer_size:
            ave_img = np.mean(img_collection,axis = 0)
            variance = []
            for tmp in img_collection:
                variance.append(myMetric(ave_img, tmp))
            cur_var = np.mean(variance)
            cur_epoch = epoch
            variance_history.append(cur_var)
            x_axis.append(cur_epoch)
            if earlystop.stop == False:
                earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)



        if epoch % 100 == 0:
            input[:, 0] = 0.95*input[:,0] + 0.05*output[:,0].cpu().detach()
            input[:, 1] = 0.95*input[:,1] + 0.05*output[:,1].cpu().detach()
            print(time.time()-start_time,'s')
            with torch.no_grad():  # 显式禁用梯度跟踪

                Rec_Dip = torch.complex(output[:,0],output[:,1])[0]
                plt.figure(figsize=(10, 10))
                plt.imshow(20*np.log10(abs((Rec_Dip/torch.max(abs(Rec_Dip))).cpu().squeeze().detach().numpy())), cmap='hot',vmin=-30,vmax=0) # 显示幅度
                plt.xticks(np.arange(0, grid_size, 50))
                plt.tick_params(labelsize=30)  # 设置字体大小
                plt.yticks(np.arange(0, grid_size, 50))
                plt.tick_params(labelsize=25)  # 设置字体大小
                plt.ylabel("Range cell",fontsize=30)
                plt.xlabel("Cross-range cell",fontsize=30)
                plt.tight_layout()
                plt.savefig('test3.png')
                plt.close()
                
                data = 20 * np.log10(abs((Rec_Dip / torch.max(abs(Rec_Dip))).cpu().squeeze().detach().numpy()))
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(data, cmap='hot', vmin=-30, vmax=0)

                # 主图刻度与标签
                ax.set_xticks(np.arange(0, grid_size, 50))
                ax.set_yticks(np.arange(0, grid_size, 50))
                ax.tick_params(labelsize=25)
                ax.set_ylabel("Range cell", fontsize=30)
                ax.set_xlabel("Cross-range cell", fontsize=30)

                # 要放大的区域范围（示例：中心部分）
                x1, x2 = 135, 175  # 横向范围
                y1, y2 = 100, 140  # 纵向范围

                # 嵌入放大图
                axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=1)
                axins.imshow(data, cmap='hot', vmin=-30, vmax=0)
                axins.set_xlim(x1, x2)
                axins.set_ylim(y2, y1)  # y轴方向反过来
                axins.set_xticks([])
                axins.set_yticks([])
                # 在原图画矩形框
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
                # 连接线
                mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="cyan")

                plt.tight_layout()
                plt.savefig('test3_zoom.png', dpi=300)
                plt.close()
                result = (RD.cuda(),Rec_ADMM,Rec_ISTA,Rec_CVres.cuda(),Rec_Hnet.cuda(),Rec_Dip)    
                plot_cv_fuc(result,-30,50)
            DIP_psnr = complex_psnr(abs(torch.complex(output[:,0],output[:,1])).unsqueeze(1).cpu().detach().numpy(), abs(label_new).detach().numpy(),
                                            peak='normalized')
            NRMSE_dip = criterion(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach(), torch.tensor(abs(grid)))
            Ssim_dip = evaluate_ssim(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach().numpy(), abs(grid))
            ent_dip = entropy_zw(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach())
            CC_dip = correlation_coefficient(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach(), torch.tensor(abs(grid)))
            grad_ssim_dip = grad_ssim(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach().numpy(), abs(grid))
            initw_ssim_dip = iw_ssim(abs(torch.complex(output[:,0],output[:,1])[0]).cpu().detach().numpy(), abs(grid))
            print("[epoch %d]" %(epoch + 1))
            print("train_loss: ", loss_normal.data.item())
            print(f"DIP_psnr: {DIP_psnr:.4f} "
                f"DIP_NRMSE: {NRMSE_dip.numpy():.4f} "
                f"DIP_SSIM: {Ssim_dip:.4f} "
                f"DIP_Ent: {ent_dip.numpy():.4f} "
                f"DIP_CC: {CC_dip:.4f}",
                f"DIP_giwSSIM: {(grad_ssim_dip+initw_ssim_dip)/2:.4f} ")
            log_ent.append(ent_dip.numpy())
            log_psnr.append(DIP_psnr)
    # savemat("psnr.mat",{'psnr':log_psnr})  
    # psnr_pre = loadmat("psnr.mat") ['psnr']     
    plt.figure(figsize=(10, 10))
    plt.plot(10*np.array(log_ent))  # 显示幅度
    # plt.plot(np.array(psnr_pre))  # 显示幅度
    plt.plot(np.array(log_psnr))  # 显示幅度
    plt.savefig('test4.png')
    plt.close()