import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):

        return torch.norm((output - target),'fro') / torch.norm(target,'fro')


class MyLoss_MSE(torch.nn.Module):
    def __init__(self):
        super(MyLoss_MSE, self).__init__()

    def forward(self, output, target):

        return torch.norm((output - target),'fro')

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
def evaluate_ssim(images, image2):

    images = images/np.max(abs(images))
    ssim_table = ssim(np.where(abs(images)<0.01,0,images), image2, data_range= 1)
    return ssim_table

def correlation_coefficient(X, X_ref):
    # 取模值（如果是复数就取复数模）
    X_abs = torch.abs(X)
    X_ref_abs = torch.abs(X_ref)
    
    # 分子：对应元素相乘再求和
    numerator = torch.sum(X_abs * X_ref_abs)
    
    # 分母：Frobenius 范数的乘积
    denominator = torch.norm(X, p='fro') * torch.norm(X_ref, p='fro')
    
    cc = numerator / denominator
    return cc.item()

"""
Gradient-based Structural Similarity Index (G-SSIM)计算

"""


import torch
import torch.nn.functional as F
from torchvision import transforms

def sobel_grad(x):
    # x: [B,1,H,W]
    sobel_x = torch.tensor([[1,0,-1],
                            [2,0,-2],
                            [1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)


def grad_ssim(x, y):
    # x,y: numpy 或 torch 都可
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    if x.ndim == 2:
        x = x[None, None]
        y = y[None, None]

    gx = sobel_grad(x)
    gy = sobel_grad(y)
    gx = gx/torch.max(gx)
    gy = gy/torch.max(gy)
    grad_ssim_value = ssim(gx.squeeze().numpy(), gy.squeeze().numpy(), data_range=1)
    return grad_ssim_value  # 直接用官方SSIM即可


"""
Information Weighted Structural Similarity Index (IW-SSIM)计算

"""

# -------------------------------------------------------------
# Gaussian kernel
# -------------------------------------------------------------
def gaussian_kernel(win_size=11, sigma=1.5):
    coords = torch.arange(win_size).float() - win_size // 2
    x_grid = coords.repeat(win_size, 1)
    y_grid = x_grid.t()
    g = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g.view(1, 1, win_size, win_size)


# -------------------------------------------------------------
# local mean and variance
# -------------------------------------------------------------
def local_stats(x, kernel):
    mu = F.conv2d(x, kernel, padding=kernel.shape[-1]//2)
    mu_sq = mu * mu

    sigma_sq = F.conv2d(x * x, kernel, padding=kernel.shape[-1]//2) - mu_sq
    sigma_sq = torch.clamp(sigma_sq, min=0)

    return mu, sigma_sq


# -------------------------------------------------------------
# single-scale SSIM (no luminance term unless top level)
# -------------------------------------------------------------
def ssim_map(x, y, kernel, C1=1e-4, C2=9e-4, include_luminance=False):

    mu_x, var_x = local_stats(x, kernel)
    mu_y, var_y = local_stats(y, kernel)

    cov_xy = (
        F.conv2d(x * y, kernel, padding=kernel.shape[-1]//2)
        - mu_x * mu_y
    )

    var_x = torch.clamp(var_x, min=0)
    var_y = torch.clamp(var_y, min=0)

    if include_luminance:
        L = (2 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1)
    else:
        L = 1.0

    C = (2 * torch.sqrt(var_x * var_y) + C2) / (var_x + var_y + C2 + 1e-12)
    S = (cov_xy + C2/2) / (torch.sqrt(var_x * var_y) + C2/2 + 1e-12)

    return L * C * S


# -------------------------------------------------------------
# downsample function for pyramid
# -------------------------------------------------------------
def downsample(x):
    return x[:, :, ::2, ::2]

def to_torch_img(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    else:
        img = img.float()

    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)
    return img


def iw_ssim(img1, img2, levels=1, win_size=11, sigma=1.5):

    img1 = to_torch_img(img1)
    img2 = to_torch_img(img2)

    kernel = gaussian_kernel(win_size, sigma).to(img1.device)

    imgs1 = [img1]
    imgs2 = [img2]

    for _ in range(levels - 1):
        imgs1.append(downsample(imgs1[-1]))
        imgs2.append(downsample(imgs2[-1]))

    ssim_maps = []
    info_maps = []

    for i in range(levels):
        is_top = (i == levels - 1)
        smap = ssim_map(imgs1[i], imgs2[i], kernel,
                        include_luminance=is_top)
        ssim_maps.append(smap)

        # information weight = local variance sum
        _, v1 = local_stats(imgs1[i], kernel)
        _, v2 = local_stats(imgs2[i], kernel)
        info = v1 + v2
        info_maps.append(info)

    # normalize weights per scale
    final_scores = []
    for w, s in zip(info_maps, ssim_maps):
        w_norm = w / (w.mean() + 1e-12)
        final_scores.append((w_norm * s).mean())

    return float(torch.stack(final_scores).mean().cpu().item())

