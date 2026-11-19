# DIP-SA-ISAR: Training-Free Sparse-Aperture ISAR Imaging via Deep Image Prior

本仓库给出论文 *DIP-SA-ISAR: Training-Free Sparse-Aperture ISAR Imaging via Deep Image Prior*（IEEE Transactions on Aerospace and Electronic Systems）的成像代码。网络把单幅散射图像当作待求变量，在距离–方位可分离的前向模型下拟合观测回波，无需成对训练数据，并与 RD、ADMM、ISTA、CV-Res-ADMM-Net 和 HNet 对比。

## 引用

如果本仓库或论文对你的工作有帮助，请引用：

Z. Wang, J. Wang, P. Li, H. Li, and Z. Ding, “DIP-SA-ISAR: Training-Free Sparse-Aperture ISAR Imaging via Deep Image Prior,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. 62, pp. 18387–18395, 2026, doi: [10.1109/TAES.2026.3738533](https://doi.org/10.1109/TAES.2026.3738533).

```bibtex
@ARTICLE{11714471,
  author={Wang, Ziwen and Wang, Jianping and Li, Pucheng and Li, Han and Ding, Zegang},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  title={DIP-SA-ISAR: Training-Free Sparse-Aperture ISAR Imaging via Deep Image Prior},
  year={2026},
  volume={62},
  number={},
  pages={18387-18395},
  keywords={Electronics packaging;Apertures;Optimization;Distance measurement;Renewable Portfolio Standard;Simulation;Gratings;TV;Measurement;Training;Deep image prior (DIP);neural network;sparse-aperture inverse synthetic aperture radar (SA-ISAR);sparse sampling},
  doi={10.1109/TAES.2026.3738533}}
```

IEEE Xplore: https://ieeexplore.ieee.org/document/11714471/

## 目录

```
├── Main_simulation.py          # 仿真飞机目标（128×128）
├── Main_dip_yark42.py          # Yak-42 目标（256×256）
├── train_dip_pretrain.py       # DIP 网络预训练
├── Comparison methods/         # HNet、CV-Res-ADMM-Net 的训练脚本
├── models/                     # HNet、Deep Image Prior 网络等
├── complex_layers/             # 复数卷积与 CV-Net
├── utils/                      # 前向模型、ADMM/ISTA、损失与指标
├── data_sat/                   # 示例散射图像（.mat）
├── logs_isarnet/               # 预训练权重
└── results/                    # 已有成像结果与指标曲线
```

## 环境

需要 NVIDIA GPU 和 CUDA 版 PyTorch。脚本里默认使用 2 号卡（`CUDA_VISIBLE_DEVICES=2`），没有这张卡时改成可用编号，或在运行前设置 `CUDA_VISIBLE_DEVICES`。

```bash
pip install torch torchvision tensorboardX numpy scipy matplotlib scikit-image scikit-learn opencv-python pillow packaging
```

在项目根目录下运行。

## 数据与权重

| 文件 | 用途 |
| --- | --- |
| `data_sat/data_feiji.mat` | 仿真目标，键名 `data_feiji` |
| `data_sat/Yak42_image.mat` | Yak-42 图像，键名 `y_d` |
| `logs_isarnet/2.pth` | 仿真实验加载的 DIP 权重 |
| `logs_isarnet/CV_Net1.pth` | CV-Res-ADMM-Net 权重 |
| `logs_isarnet/HNet.pth` | HNet 权重 |
| `logs_isarnet/1.pth` | `train_dip_pretrain.py` 的保存结果 |

## 运行

仿真对比（RD、ADMM、ISTA、CV-Net、HNet、DIP）：

```bash
python Main_simulation.py --data_dir data_sat/ --num_epoch 20000 --outf logs_isarnet
```

Yak-42：

```bash
python Main_dip_yark42.py --data_dir data_sat/ --num_epoch 20000 --outf logs_isarnet
```

重新预训练 DIP（随机点目标，默认 100 个 epoch，结果写入 `logs_isarnet/1.pth`）：

```bash
python train_dip_pretrain.py --num_epoch 100 --outf logs_isarnet
```

对比方法的训练脚本在 `Comparison methods/` 下，训练完成后权重需放到 `logs_isarnet/`，文件名与上表一致。

成像过程中会在当前目录写出 `test.png`、`test1.png`、`test3.png` 等中间图；每 100 个 epoch 调用 `utils/plot.py` 把六种方法画在一张图上。评价指标包括 PSNR、NRMSE、SSIM、熵和相关系数。已整理的曲线在 `results/`。

## 方法说明

回波按$\mathbf{S}=\mathbf{F}_r\mathbf{X}\mathbf{F}_a$生成。方位向按给定索引抽掉一部分脉冲，得到稀疏孔径观测，再加复高斯噪声。DIP 使用 `MediumUNet`，输入为稀疏孔径距离–多普勒图的实部与虚部，损失是回波拟合项加上幅度对数稀疏项和各向同性 TV。迭代中用方差早停（`utils/stopping_criterion.py`）观察是否过拟合。

## 已知问题

- `utils/unet.py` 依赖 `unet_parts.py`（`DoubleConv`、`Down`、`Up`、`OutConv`）。源文件不在仓库里，只剩下 `__pycache__/unet_parts.cpython-311.pyc`。
- `Main_simulation.py`、`Main_dip_yark42.py` 和 `train_dip_pretrain.py` 都写了 `from Main_simulation import init_weights`，但 `init_weights` 只定义在对比方法的训练脚本里。`generate_specified_blocks` 写在 `Main_simulation.py` 的 `if __name__ == '__main__'` 内部，别的脚本无法导入。
