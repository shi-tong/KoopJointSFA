import matplotlib.pylab as plt
import numpy as np
from typing import Literal
import seaborn as sns
def visualize_feature_surface(attn_weight, 
                            batch_idx=0, 
                            mode:Literal["mean", 'batch'] = 'mean',
                            figsize = (10, 8), 
                            cmap="jet", 
                            save_name=None, 
                            show_label=False):
    B, T, C = attn_weight.shape
    if mode == 'mean':
        X = attn_weight.mean(dim=0)          # (T, C)
    else:
        X = attn_weight[batch_idx]
    
    weights = X.detach().cpu().numpy()               # (T, C)

    # 4) 网格 & 绘图
    Xg, Yg = np.meshgrid(np.linspace(0, T-1, T),
                         np.linspace(0, C-1, C),
                         indexing="ij")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, weights, cmap=cmap,
                        rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.view_init(elev=30, azim=30)
    if show_label:
        ax.set_xlabel('Temporal index')
        ax.set_ylabel('Channel index')
        ax.set_zlabel('Value')
    else:
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    fig.colorbar(surf, shrink=0.5, aspect=10)
    if save_name is None:
        save_name = '3d_surface.png'
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Union
import matplotlib.ticker as ticker

def HeatMap(
    data: Union[np.ndarray, torch.Tensor],
    fontsize: int = 16,
    num_ticks = 8,
    figsize = (9, 5),
    batch_idx: int = 0,
    mode: Literal["mean", "batch"] = "mean",
    colormap: Literal[
        "viridis","plasma","magma","cividis",
        "coolwarm","jet","hsv","rainbow"
    ] = "viridis",
    normalize: bool = False,
    xlabel: str = "Time step",
    colorbar_label: str = "Value",
    save_name: str = None
):
    """
    data shape = (B, T)

    mode = 'mean'  -> 显示 Batch 均值
    mode = 'batch' -> 显示指定 batch 样本
    """

    # -------- 0) 转成 torch tensor --------
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    assert isinstance(data, torch.Tensor), "data 必须是 torch.Tensor 或 numpy.ndarray"
    assert data.ndim == 2, "data 必须是 (B, T) 形状"

    B, T = data.shape

    # -------- 1) 选择数据模式 --------
    if mode == "mean":
        X = data.mean(dim=0, keepdim=True)   # (1, T)
    else:
        batch_idx = min(max(batch_idx, 0), B - 1)
        X = data[batch_idx].unsqueeze(0)     # (1, T)

    # -------- 2) 归一化（可选）--------
    if normalize:
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    X = X.detach().cpu().numpy()  # -> (1, T)

    # -------- 3) 全局字体 --------
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots(figsize=figsize)

    # -------- 4) 绘制热力图 (T 在横轴) --------
    hm = sns.heatmap(
        X,
        cmap=colormap,
        cbar_kws={"label": colorbar_label},
        ax=ax
    )

    # -------- 5) 坐标轴标签 --------
    num_ticks = min(num_ticks, T)
    xticks = np.linspace(1, T, num=num_ticks, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontname="Times New Roman")
    ax.set_xlabel(xlabel, fontname="Times New Roman", fontsize=fontsize)
    ax.set_yticks([])
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # -------- 6) 坐标刻度字体 --------
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")

    # -------- 7) colorbar 字体（legend）--------
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontname("Times New Roman")

    for t in cbar.ax.get_yticklabels():
        t.set_fontname("Times New Roman")

    plt.tight_layout()

    # -------- 8) 保存 --------
    if save_name is None:
        save_name = "Heatmap.png"

    plt.savefig(save_name, dpi=400, bbox_inches="tight")
    plt.close(fig)

