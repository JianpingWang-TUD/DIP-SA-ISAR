"""
2025/7/24
添加ADMM,CV-Res-admm和HNet的测试对比代码

"""

from __future__ import print_function, division
import torch
from utils.fftc import *
from utils.ADMM import SOFT
from complex_layers.CV_Net import CVnet
from models import Hynet



def CVnet_test(Fr,Fa,echo,RD):

    model_c = CVnet(1,32).cuda()
    model_c.load_state_dict(torch.load('logs_isarnet/CV_Net1.pth'))
    Z = torch.zeros_like(RD).cuda().unsqueeze(0).unsqueeze(0)
    U = torch.zeros_like(Z).cuda()
    model_c.eval()
    for iter in range(6):
        # 前向计算
        X = Z-U - 1/(1+1.2)*torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr.T),torch.einsum('km,ijml,ln->ijkn', Fr,Z-U, Fa)-echo, torch.conj(Fa.T))
        Z = model_c(X + U.detach())  # 切断历史梯度
        U = U + X.detach() - Z.detach()  # 不追踪这步梯度
    return X

def Hnet_test(Fr,Fa,echo,RD):

    model_c = Hynet.CSWinTransformerNet(45).cuda()
    model_c.load_state_dict(torch.load('logs_isarnet/HNet.pth'))
    Z = torch.zeros_like(RD).cuda().unsqueeze(0).unsqueeze(0)
    U = torch.zeros_like(Z).cuda()
    Ahy = torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr.T),echo, torch.conj(Fa.T))
    model_c.eval()
    hyper = model_c(abs(Ahy).cuda())
    for iter in range(15):
        # 前向计算
        X = Z-U - 1/(1+hyper[0,0+3*iter])*torch.einsum('km,ijml,ln->ijkn', torch.conj(Fr.T),torch.einsum('km,ijml,ln->ijkn', Fr,Z-U, Fa)-echo, torch.conj(Fa.T))
        Z = SOFT(X + U,hyper[0,1+3*iter]/10)  # 切断历史梯度
        U = U + hyper[0,2+3*iter]*(X - Z)
    return X