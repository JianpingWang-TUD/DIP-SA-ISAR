import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch
from scipy.io import loadmat,savemat

def plot_cv_fuc(output,db_min,step):# 创建图形
    
    fig = plt.figure(figsize=(60, 10))
    # 使用 GridSpec 来设置布局
    gs = GridSpec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.05])  # 最后一列给 colorbar

    # 绘制第一个子图
    ax1 = fig.add_subplot(gs[0])
    cax1 = ax1.imshow(20*np.log10(abs((output[0]/torch.max(abs(output[0]))).cpu().squeeze().detach().numpy())), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax1.set_title("MF")
    ax1.set_xticks(np.arange(0, output[0].shape[0], step))
    ax1.tick_params(labelsize=30)  # 设置字体大小
    ax1.set_yticks(np.arange(0, output[0].shape[1], step))
    ax1.tick_params(labelsize=30)  # 设置字体大小
    ax1.set_ylabel("Range cell",fontsize=40)
    ax1.set_xlabel("Cross-range cell",fontsize=40)
    # ax1.text(40,198,'PSNR:'+str(round(test_psnr_value[0],2))+'dB',fontsize=30,color ='r')

    # 绘制第二个子图
    ax2 = fig.add_subplot(gs[1])
    cax2 = ax2.imshow(20*np.log10(abs((output[1]/torch.max(abs(output[1]))).cpu().squeeze().detach().numpy())+0.0000001), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax2.set_title("LADMM")
    ax2.set_xlabel("Cross-range cell",fontsize=40)
    ax2.set_xticks(np.arange(0, output[0].shape[0], step))
    ax2.tick_params(labelsize=30)  # 设置字体大小
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[2])
    cax3 = ax3.imshow(20*np.log10(abs((output[2]/torch.max(abs(output[2]))).cpu().squeeze().detach().numpy())+0.000001), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax3.set_title("SwinT-ADMM(Sup)")
    ax3.set_xlabel("Cross-range cell",fontsize=40)
    ax3.set_xticks(np.arange(0, output[0].shape[0], step))
    ax3.tick_params(labelsize=30)  # 设置字体大小
    ax3.set_yticks([])
    # ax3.text(40,198,'PSNR:'+str(round(test_psnr_value[2],2))+'dB',fontsize=30,color ='r')
    # 绘制第四个子图
    ax4 = fig.add_subplot(gs[3])
    cax4 = ax4.imshow(20*np.log10(abs((output[3]/torch.max(abs(output[3]))).cpu().squeeze().detach().numpy())), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax4.set_title("SFH-ADMM-Net(Sup)")
    ax4.set_xlabel("Cross-range cell",fontsize=40)
    ax4.set_xticks(np.arange(0, output[0].shape[0], step))
    ax4.tick_params(labelsize=30)  # 设置字体大小
    ax4.set_yticks([])
    # ax4.text(40,198,'PSNR:'+str(round(test_psnr_value[3],2))+'dB',fontsize=30,color ='r')
    # 绘制第四个子图
    ax5 = fig.add_subplot(gs[4])
    cax5 = ax5.imshow(20*np.log10(abs((output[4]/torch.max(abs(output[4]))).cpu().squeeze().detach().numpy())), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax5.set_title("ZS-ISAR-Net")
    ax5.set_xlabel("Cross-range cell",fontsize=40)
    ax5.set_xticks(np.arange(0, output[0].shape[0], step))
    ax5.tick_params(labelsize=30)  # 设置字体大小
    ax5.set_yticks([])
    # ax5.text(40,198,'PSNR:'+str(round(test_psnr_value[4],2))+'dB',fontsize=30,color ='r')
    
    ax6 = fig.add_subplot(gs[5])
    cax6 = ax6.imshow(20*np.log10(abs((output[5]/torch.max(abs(output[5]))).cpu().squeeze().detach().numpy())), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # ax6.set_title("Sup-ISAR-Net")
    ax6.set_xlabel("Cross-range cell",fontsize=40)
    ax6.set_xticks(np.arange(0, output[0].shape[0], step))
    ax6.tick_params(labelsize=30)  # 设置字体大小
    ax6.set_yticks([])
    # ax6.text(40,198,'PSNR:'+str(round(test_psnr_value[5],2))+'dB',fontsize=30,color ='r')
    # 绘制第四个子图
    # ax7 = fig.add_subplot(gs[6])
    # cax7 = ax7.imshow(20*np.log10(abs((output[6]/torch.max(abs(output[6]))).cpu().squeeze().detach().numpy())+0.001), cmap='hot', aspect='auto',vmin=db_min,vmax=0)
    # # cax6 = ax6.imshow(abs(label[0]).cpu().squeeze().detach().numpy(), cmap='hot', aspect='auto')
    # # ax7.set_title("Ground Truth")
    # ax7.set_xlabel("Cross-range cell",fontsize=30)
    # ax7.set_xticks(np.arange(0, output[0].shape[0], step))
    # ax7.set_yticks([])    
    # 添加 colorbar
    cbar_ax = fig.add_subplot(gs[6])  # colorbar 放在第五个位置
    cbar = fig.colorbar(cax6, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=30)

    plt.tight_layout()
    # 显示图形
    plt.savefig('test1.png')
    plt.close()