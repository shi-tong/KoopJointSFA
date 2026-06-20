from decimal import Decimal
import os
import pandas as pd
from typing import Literal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
def to_int_scientific(x):
    d = Decimal(str(x)).normalize()
    sign, digits, exponent = d.as_tuple()

    int_part = ''.join(map(str, digits))
    if sign:
        int_part = '-' + int_part

    return f"{int_part}e{exponent}"


def Confusion_matrix_shrink(
        test_trues,
        test_predictions,
        num_classes:int,
        figsize=(10, 8),
        cmap:Literal['Blues', 'Greens', 'Purples', 'Reds', 'Greys', 'Oranges'] = 'Blues',
        fontsize=28,
        num_font_size = 23,
        save_name = None,
        norm_mode: Literal["row", "col", "none"] = "row"
    ):
    """
    norm_mode = 'row'  -> 每一行归一化（真实类别误差去向）
    norm_mode = 'col'  -> 每一列归一化（误分类来源）
    norm_mode = 'none' -> 不归一化（真实数量）
    """

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = fontsize

    # ---- 1) 原始混淆矩阵（未归一化）----
    cm = confusion_matrix(
        test_trues,
        test_predictions,
        labels=np.arange(num_classes)
    )

    # ---- 2) 找出“存在误分类”的类别 ----
    row_sums = cm.sum(axis=1)
    correct = np.diag(cm)
    mask = correct < row_sums          # 仅保留 misclassified 类

    if not mask.any():
        print("All classes predicted correctly — nothing to shrink.")
        return

    cm_sub = cm[np.ix_(mask, mask)]

    # ---- 3) 构造类别标签 ----
    labels = [str(i+1) for i in range(num_classes)]
    
    sub_labels = [labels[i] for i in range(num_classes) if mask[i]]

    # ---- 4) 归一化方式选择 ----
    if norm_mode == "row":
        denom = cm_sub.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm_sub / denom

    elif norm_mode == "col":
        denom = cm_sub.sum(axis=0, keepdims=True).clip(min=1)
        cm_plot = cm_sub / denom

    else:   # "none"
        cm_plot = cm_sub.astype(float)

    # ---- 5) 绘图 ----
    plt.figure(figsize=figsize)

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=".3f" if norm_mode != "none" else "g",
        cmap=cmap,
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        annot_kws={"size": num_font_size}
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("Confused_matrix.png" if not save_name else f"{save_name}", dpi=600)
    plt.close()


def Confusion_matrix_plot(
        test_trues, 
        test_predictions,
        figsize = (20, 16),
        fontsize = 28,
        num_font_size = 22,
        colormap:Literal['Blues', 'Greens', 'Purples', 'Reds', 'Greys', 'Oranges'] = 'Blues'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import matplotlib as mpl
    import numpy as np
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = fontsize
    labels = [str(i+1) for i in np.unique(test_trues)]

    cm = confusion_matrix(test_trues, test_predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            eps = 1e-6
            if v < eps:
                annot[i, j] = f"{v:.0f}"          # 0 -> 0
            else:
                annot[i, j] = f"{v:.3f}"  # 非 0 正常显示

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=colormap,
        xticklabels=labels
    )
    # 创建热图后，根据数值设置字体颜色
    for text_obj in ax.texts:
        value = float(text_obj.get_text())
        eps = 1e-6
        if value < eps:
            text_obj.set_color("black")  # 0值
            text_obj.set_weight("normal")
        else:
            # text_obj.set_color("")  # 非0值
            text_obj.set_weight("bold")
            text_obj.set_fontsize(num_font_size)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi = 300)

def Confusion_matrix_plot_split(test_trues, 
                                test_predictions, 
                                num_classes=21, 
                                block_size=7, 
                                figsize = (10, 8),
                                fontsize = 25,
                                colormap:Literal['Blues', 'Greens', 'Purples', 'Reds', 'Greys', 'Oranges'] = 'Blues'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = fontsize

    cm = confusion_matrix(test_trues, test_predictions, labels=np.arange(num_classes))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm_norm, dtype=object)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            eps = 1e-6
            if v < eps:
                annot[i, j] = f"{v:.1f}"          # 0 -> 0.0
            else:
                annot[i, j] = f"{v:.3f}"  # 非 0 正常显示

    num_blocks = num_classes // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size

        sub_cm = cm_norm[start:end, start:end]
        sub_annot = annot[start:end, start:end]

        plt.figure(figsize=figsize)
        sns.heatmap(
            sub_cm,
            annot=sub_annot,
            fmt="",
            cmap=colormap,
            cbar=True,
            xticklabels=np.arange(start, end),
            yticklabels=np.arange(start, end)
        )

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{start}_{end-1}.png", dpi=300)
        plt.close()



def write_acc_loss(
    acc_list,
    loss_list,
    flag:Literal['Train', 'Test', 'Valid'] = 'Train',
    save_path = None
):
    """
    acc_list  : list[float]
    loss_list : list[float]
    """

    assert len(acc_list) == len(loss_list), \
        "acc_list and loss_list must have same length"

    df = pd.DataFrame({
        'epoch': range(1, len(acc_list) + 1),
        'acc': acc_list,
        'loss': loss_list
    })

    # save
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{flag}.csv")
        df.to_csv(file_path, index=False)
        print('Write Over!')

def PR_Curve(test_trues, test_probs, num_class, model):
        from sklearn.metrics import precision_recall_curve, auc
        from sklearn.preprocessing import label_binarize
        import seaborn as sns
        """
        绘制多分类的 Precision-Recall 曲线

        参数:
        - test_trues: 一维数组，真实标签
        - test_probs: 二维数组,shape = [样本数, 类别数],每类的预测概率(经过softmax)
        - num_class: 类别总数
        """
        # 设置字体样式
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 35

        # 将真实标签转为 one-hot 编码
        y_true_bin = label_binarize(test_trues, classes=list(range(num_class)))

        # 初始化
        precision = dict()
        recall = dict()
        pr_auc = dict()

        # 为每一类绘制 PR 曲线
        for i in range(num_class):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], test_probs[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # 绘图
        plt.figure(figsize=(20, 15))
        colors = sns.color_palette("husl", num_class)

        for i in range(num_class):
            plt.plot(recall[i], precision[i],
                     lw=2,
                     color=colors[i],
                     label=f'Fault {i+1} (AUC = {pr_auc[i]:.2f})' if i >= 9 else f'Fault 0{i+1} (AUC = {pr_auc[i]:.2f})')

        plt.xlabel('')
        plt.ylabel('')
        # 隐藏x轴刻度标签但保留刻度线
        plt.tick_params(axis='x', labelbottom=False)
        # 隐藏y轴刻度标签但保留刻度线
        plt.tick_params(axis='y', labelleft=False)
        plt.legend(loc='center left', 
           fontsize=30,
           bbox_to_anchor=(1.02, 0.5),  # 放在图外右上角
           borderaxespad=0.0)  # 图例与坐标轴之间的间距
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{model}_PR_Curve.png")
