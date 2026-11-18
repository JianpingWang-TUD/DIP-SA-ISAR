import numpy as np
import torch
import math
from skimage.metrics import structural_similarity as ssim

def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    # a = np.abs(x - y) ** 2
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse + 1e-5)

def complex_psnr_many(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    # a = np.abs(x - y) ** 2
    mse = np.mean(np.abs(x - y)**2,axis=(1,2))
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse + 1e-5)

def nrmse(outputs, targets):
    """
    Normalized root-mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Normalized root-mean square deviation
    """
    # Flatten tensors
    outputs = outputs.reshape(-1)[:,None]
    targets = targets.reshape(-1)[:,None]

    # Check dim
    if outputs.size() != targets.size():
        raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Normalization with N-1
    var = torch.std(targets) ** 2

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(math.sqrt(torch.mean(error) / var))

def nrmse_many(outputs, targets):
    """
    Normalized root-mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Normalized root-mean square deviation
    """
    
    # Error
    NMSE = np.sum((targets - outputs)**2,axis=(1, 2))/np.sum((targets.squeeze() ** 2))
    # Return
    return NMSE

def evaluate_ssim(images, image2):
    nums,_,_ = images.shape
    ssim_table = np.zeros(nums)
    for i in range(nums):
        ssim_table[i] = ssim(np.where(images[i] < 0.001, 0, images[i]), image2[0],data_range= 1)
    return ssim_table


"""
Gradient-based Structural Similarity Index (G-SSIM)计算

"""

from scipy.ndimage import sobel

def compute_gradients(img):
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return np.sqrt(gx**2 + gy**2)

def gaussian_filter(img, size=11, sigma=1.5):
    """
    Fast separable Gaussian filter
    """
    radius = size // 2
    x = np.arange(-radius, radius + 1)
    g = np.exp(-(x**2) / (2*sigma*sigma))
    g /= g.sum()

    tmp = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'),
                              axis=0, arr=img)
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode='same'),
                              axis=1, arr=tmp)
    return out

def g_ssim(X, Y, C1=0.01**2, C2=0.03**2, window_size=11, sigma=1.5):
    # Gradient magnitude maps
    Gx = compute_gradients(X)
    Gy = compute_gradients(Y)

    # Gaussian window smoothing
    mu_x = gaussian_filter(Gx, window_size, sigma)
    mu_y = gaussian_filter(Gy, window_size, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(Gx*Gx, window_size, sigma) - mu_x2
    sigma_y2 = gaussian_filter(Gy*Gy, window_size, sigma) - mu_y2
    sigma_xy = gaussian_filter(Gx*Gy, window_size, sigma) - mu_xy

    numerator = (2 * mu_xy + C1) * (2*sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean()

"""
Information Weighted Structural Similarity Index (IW-SSIM)计算

"""
import numpy as np
from scipy.ndimage import gaussian_filter

def downsample(img):
    return img[::2, ::2]

def compute_local_stats(X, Y, sigma=1.5):
    mu_x = gaussian_filter(X, sigma)
    mu_y = gaussian_filter(Y, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(X*X, sigma) - mu_x2
    sigma_y2 = gaussian_filter(Y*Y, sigma) - mu_y2
    sigma_xy = gaussian_filter(X*Y, sigma) - mu_xy

    return mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy

def iw_ssim(X, Y, levels=5, C1=0.01**2, C2=0.03**2):
    Xs = [X]
    Ys = [Y]

    # build pyramid
    for _ in range(levels-1):
        Xs.append(downsample(Xs[-1]))
        Ys.append(downsample(Ys[-1]))

    weight_list = []
    ssim_list = []

    for i in range(levels):
        imgX = Xs[i]
        imgY = Ys[i]

        mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy = compute_local_stats(imgX, imgY)

        # information weight (local contrast)
        info = sigma_x2 + sigma_y2 + 1e-12
        weight_list.append(info)

        if i == levels - 1:
            # highest level includes luminance term
            ssim = ((2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)) * \
                   ((2*np.sqrt(sigma_x2*sigma_y2) + C2) /
                    (sigma_x2 + sigma_y2 + C2))
        else:
            # others: contrast+structure only
            ssim = ((2*np.sqrt(sigma_x2*sigma_y2) + C2) /
                    (sigma_x2 + sigma_y2 + C2)) * \
                   ((sigma_xy + C2/2) /
                    (np.sqrt(sigma_x2*sigma_y2) + C2/2))

        ssim_list.append(np.clip(ssim, 0, 1))

    # normalize weights
    weight_list = [w / (w.mean() + 1e-12) for w in weight_list]

    # final IW-SSIM score
    iw_score = 1.0
    for w, s in zip(weight_list, ssim_list):
        iw_score *= (s.mean() ** w.mean())

    return float(iw_score)
