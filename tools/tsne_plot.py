import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Literal
import seaborn as sns
import matplotlib as mpl

def plot_tsne(
    x: torch.Tensor,
    labels: torch.Tensor, # (B, )
    figsize:tuple = (15, 12),
    color_map:Literal['husl', 'hls', 'rainbow'] = 'husl',
    mode:Literal['mean', 'downsample'] = 'mean',
    target_length = None,
    default_ratio = 0.01, # for downsample
    perplexity: int = 30,
    n_iter: int = 1000,
    font_size: int = 25,
    legend_marker_scale:float = 2.5, 
    s: int = 20,
    alpha: float = 0.7,
    random_state: int = 42,
    save_path = None
):
    """
    x: Tensor [B, T, N]
    label: B
    """
    assert x.dim() == 3, "Input must be B x T x N"

    # -------- reshape --------
    if mode == 'mean':
        data = x.mean(dim = 1) # (B, N)
    elif mode == 'downsample':
        x_transpose = x.transpose(1, 2) # B, N, T
        B, N, T = x_transpose.shape
        if target_length is None:
            target_length = int(T * default_ratio)
        x_downsample = torch.nn.functional.interpolate(
            x_transpose,
            size=target_length,
            mode="linear",
            align_corners=False)
        data = x_downsample.transpose(1, 2).reshape(-1, N) # (B*T_downsample, N)
        labels = labels.unsqueeze(1).repeat(1, target_length).reshape(-1) # (B*T_downsample, )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data = data.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # -------- t-SNE --------
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    print("Start fitting")
    emb = tsne.fit_transform(data)
    print("Over fitting")

    unique_labels = np.unique(labels)
    palette = sns.color_palette(color_map, len(unique_labels))

    # -------- plot --------
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=figsize)
    print("Starting ploting")
    
    for idx, lab in enumerate(unique_labels):
        mask = (labels == lab)
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            color=palette[idx],
            label = f"Fault {lab+1}",
            alpha = alpha,
            s = s
        )
    print("Over ploting")

    plt.tight_layout()
    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), markerscale=legend_marker_scale , frameon=True)
    if save_path is None:
        plt.savefig("TSNE.png", dpi = 300, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

from umap import UMAP
def plot_umap(
    x: torch.Tensor,
    labels: torch.Tensor,
    figsize=(15, 12),
    color_map: Literal['husl', 'hls', 'rainbow'] = 'husl',
    mode: Literal['BT', 'B'] = 'BT',
    n_neighbors: int = 15,
    n_iter:int = 500,
    min_dist: float = 0.1,
    spread = 1.0,
    font_size: int = 25,
    legend_marker_scale: float = 2.5,
    legend_font_size = 30, 
    s: int = 15,
    alpha: float = 0.6,
    random_state: int = 42,
    save_path=None
):
    """
    x: Tensor [B, T, N]
    """

    assert x.dim() == 3

    if labels.dim() == 2:
        labels = labels.squeeze(1)

    # -------- reshape --------
    if mode == 'BT':
        data = x.reshape(-1, x.size(-1))           # (B*T, N)
        labels = labels.repeat_interleave(x.size(1))
    elif mode == 'B':
        data = x.mean(dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data = data.cpu().numpy()
    labels = labels.cpu().numpy()

    # -------- UMAP --------
    reducer = UMAP(
        n_components=2,
        n_epochs=n_iter,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric='euclidean',
        random_state=random_state,
        verbose=True
    )
    emb = reducer.fit_transform(data)

    # -------- plot --------
    unique_labels = np.unique(labels)
    palette = sns.color_palette(color_map, len(unique_labels))

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=figsize)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=s,
            alpha=alpha,
            color=palette[i],
            edgecolors='k',          # black edge
            linewidths=0.3,           
            label=f"Fault {lab}"
        )

    plt.legend(
        loc='center right',
        bbox_to_anchor=(1.2, 0.5),
        markerscale=legend_marker_scale,
        fontsize = legend_font_size,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path or "UMAP.png", dpi=300, bbox_inches='tight')
    plt.close()
