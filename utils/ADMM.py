import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def SOFT(z,threshold):
    out = torch.exp(1j*torch.angle(z))* torch.maximum(torch.abs(z)-threshold,0*abs(z))   
    return out

def ADMM(F, S, thr,rho0,epochs):

    Nn = F.shape[1]

    Z = torch.zeros_like(torch.conj(F.T) @ S,dtype=torch.complex64)
    U = torch.zeros_like(torch.conj(F.T) @ S,dtype=torch.complex64)
    I = torch.eye(Nn)
    if S.device.type == 'cuda':
        I = I.cuda()
        Z = Z.cuda()
        U = U.cuda()
    AHA = (torch.conj(F.T) @ F + rho0*I).inverse()
    for epoch in range(epochs):
        # updata_X
        mm =  (torch.einsum('km,iml->ikl', torch.conj(F.T), S)+rho0*(Z-U))
        X = torch.einsum('km,iml->ikl', AHA, mm)
        # updata_Z sparse约束
        Z = SOFT(X+U , thr)
        # # updata_U
        U = U + X - Z
        # print(epoch)
    return Z 


def ADMM_2D(F1,F2, S, thr,rho0,epochs):

    Z = torch.zeros_like(torch.conj(F1.T) @ S@ torch.conj(F2.T),dtype=torch.complex64)
    U = torch.zeros_like(Z,dtype=torch.complex64)

    if S.device.type == 'cuda':
        Z = Z.cuda()
        U = U.cuda()
    for epoch in range(epochs):
        # updata_X
        X = Z-U - 1/(1+rho0)*torch.conj(F1.T) @(F1@(Z-U)@F2-S)@torch.conj(F2.T)
        # updata_Z sparse约束
        Z = SOFT(X+U , thr)
        # # updata_U
        U = U + X - Z
        # print(epoch)
    return X 

def ADMM_2D_weight(F1,F2, S, thr,rho0,epochs):

    Z = torch.zeros_like(torch.conj(F1.T) @ S@ torch.conj(F2.T),dtype=torch.complex64)
    U = torch.zeros_like(Z,dtype=torch.complex64)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False).cuda()
    # 设置权重为全 1
    with torch.no_grad():
        conv.weight[:] = 1.0  # shape: [1, 1, 3, 3]
    if S.device.type == 'cuda':
        Z = Z.cuda()
        U = U.cuda()
    for epoch in range(epochs):
        # updata_X
        X = Z-U - 1/(1+rho0)*torch.conj(F1.T) @(F1@(Z-U)@F2-S)@torch.conj(F2.T)
        # updata_Z sparse约束
        Z = SOFT(X+U , thr/(conv(abs(X).unsqueeze(0).unsqueeze(0)).squeeze()+1e-5))
        # # updata_U
        U = U + X - Z
        # print(epoch)
    return X.detach()


def ADMM_2D_Error(F1,F2, S, thr,rho0,epochs):

    Z = torch.zeros_like(torch.conj(F1.T) @ S@ torch.conj(F2.T),dtype=torch.complex64)
    U = torch.zeros_like(Z,dtype=torch.complex64)

    if S.device.type == 'cuda':
        Z = Z.cuda()
        U = U.cuda()
    for epoch in range(epochs):
        for epoch1 in range(30):
            # 成像
            X = Z-U - 1/(1+rho0)*torch.conj(F1.T) @(F1@(Z-U)@F2-S)@torch.conj(F2.T)
            Z = SOFT(X+U , thr)
            U = U + X - Z
        
        # 误差估计
        W = torch.conj(F1@X@F2)*S
        l = torch.argmax (torch.abs(torch.exp(2j * torch.pi * torch.linspace(-64, 63, 128, dtype=torch.complex64)[:,None]@torch.linspace(1, 128, 128,dtype=torch.complex64)[None,:]).cuda() @  W),0)-63
        fai = -torch.angle(torch.sum(torch.exp(2j * torch.pi * torch.linspace(1, 128, 128, dtype=torch.complex64)[:, None].cuda() @ l[None, :].type(torch.complex64)) *  W,0))
        S = S * torch.exp(1j*(2*torch.pi*torch.linspace(1, 128, 128, dtype=torch.complex64)[:, None].cuda()*l[None, :].type(torch.complex64)/128+fai))
    return X 

def ISTA_2d(Fr,Fa, S, thr,ruo,epochs):
    X = (torch.conj(Fr.T) @ S@torch.conj(Fa.T) )
    for epoch in range(epochs):
        X = SOFT(X+ruo*torch.conj(Fr.T)@(S- Fr@X@Fa)@torch.conj(Fa.T),thr)
    return X 

def ISTA(F, S, thr,ruo,epochs):
    X = (torch.conj(F.T) @ S)
    for epoch in range(epochs):
        X = SOFT(X-ruo*torch.conj(F.T) @(F@X-S),thr)
    return X 

