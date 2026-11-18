import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Multi-Layer Perceptron
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossShapedWindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape

        # 横向 attention：每一行做注意力
        x_h = x  # [B, H, W, C]
        qkv_h = self.qkv(x_h).reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv_h = qkv_h.permute(3, 0, 1, 4, 2, 5)  # [3, B, H, heads, W, head_dim]
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale  # [B, H, heads, W, W]
        attn_h = attn_h.softmax(dim=-1)
        out_h = (attn_h @ v_h)  # [B, H, heads, W, head_dim]
        out_h = out_h.permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        # 纵向 attention：每一列做注意力
        x_v = x.permute(0, 2, 1, 3)  # [B, W, H, C]
        qkv_v = self.qkv(x_v).reshape(B, W, H, 3, self.num_heads, self.head_dim)
        qkv_v = qkv_v.permute(3, 0, 1, 4, 2, 5)  # [3, B, W, heads, H, head_dim]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
        attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale  # [B, W, heads, H, H]
        attn_v = attn_v.softmax(dim=-1)
        out_v = (attn_v @ v_v)  # [B, W, heads, H, head_dim]
        out_v = out_v.permute(0, 3, 1, 2, 4).reshape(B, H, W, C)

        # 融合横纵注意力
        out = out_h + out_v
        out = self.proj(out)
        return out

# -------------------------
# Cross-Shaped Window Self-Attention
# -------------------------
class CSWinBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = nn.Conv2d(dim,dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # attention + residual
        x = x + self.mlp(self.norm2(x))   # mlp + residual
        return x


# -------------------------
# Patch Embedding Block
# -------------------------
class ConvEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.proj(x)                     # [B, C, H/4, W/4]   # [B, HW/16, C]
        return x

# -------------------------
# Stage Block (downsample + CSWin blocks)
# -------------------------
class Stage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, heads):
        super().__init__()
        self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[CSWinBlock(out_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.downsample(x)              # downsample spatial
        return self.blocks(x)

# -------------------------
# CSWin Transformer Network
# -------------------------
class CSWinTransformerNet(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()
        self.embed = ConvEmbedding(1, 64)

        self.stage1 = nn.Sequential(
            *[CSWinBlock(64) for _ in range(2)]
        )
        self.stage2 = Stage(64, 128, num_blocks=1, heads=6)
        self.stage3 = Stage(128, 256, num_blocks=1, heads=12)
        self.stage4 = Stage(256, 512, num_blocks=1, heads=24)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.outlayer = nn.Sigmoid()
    def forward(self, x):
        x = self.embed(x)           # [B, N, C]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.flatten(2)      # [B, C, N]
        x = self.pool(x).squeeze(-1)
        return self.outlayer(self.fc(x))
