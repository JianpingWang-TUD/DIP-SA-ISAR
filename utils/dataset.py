import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 复数图像数据集生成和封装（回归任务）
class ComplexImageDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=128, min_points=200, max_points=400):
        self.num_samples = num_samples
        self.img_size = img_size
        self.min_points = min_points
        self.max_points = max_points
        self.data, self.targets = self._generate_data()
        
    def _generate_data(self):
        """生成复数图像数据和回归目标（点数）"""
        data_real = torch.zeros((self.num_samples, self.img_size, self.img_size))
        data_imag = torch.zeros((self.num_samples, self.img_size, self.img_size))
        targets = torch.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            num_points = torch.randint(self.min_points, self.max_points+1, (1,)).item()
            targets[i] = num_points / self.max_points  # 归一化到[0,1]
            
            coords = torch.randint(10, self.img_size-10, (num_points, 2))
            real_vals = 0.9*torch.rand(num_points)+0.1
            imag_vals = 0.9*torch.rand(num_points)+0.1
            
            for j in range(num_points):
                x, y = coords[j]
                data_real[i, x, y] += real_vals[j]
                data_imag[i, x, y] += imag_vals[j]
            data_real[i] = data_real[i]/torch.max(abs(torch.complex(data_real[i], data_imag[i])))
            data_imag[i] = data_imag[i]/torch.max(abs(torch.complex(data_real[i], data_imag[i])))
        # 合并为复数并添加通道维度 (N, 1, H, W)
        complex_data = torch.complex(data_real, data_imag).unsqueeze(1)
        return complex_data, targets
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.data[idx]
        return img