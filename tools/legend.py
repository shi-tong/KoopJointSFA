import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

# 自定义圆形处理器
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Circle(xy=center, radius=min(width, height)/2)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def plot_husl_legend(unique_labels, color_map='husl'):
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 45
    
    # 生成调色板
    palette = sns.color_palette(color_map, len(unique_labels))
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(4, 0.5 * len(unique_labels) + 1))
    ax.axis('off')
    
    # 创建圆形patch
    handles = []
    for col in palette:
        circle = mpatches.Circle((0, 0), 1, facecolor=col, edgecolor='black', linewidth=1.5)
        handles.append(circle)
    
    # 创建图例
    legend = ax.legend(
        handles=handles,
        labels=['Fault 0' + str(lab) if lab < 10 else 'Fault ' + str(lab) for lab in unique_labels],
        fontsize = 50,
        loc='center left',
        frameon=False,
        handler_map={mpatches.Circle: HandlerCircle()},
        handlelength=2.0,
        handleheight=1.0,
        handletextpad=0.05,
        labelspacing=0.5
    )

    plt.savefig('legend.png', dpi=600, bbox_inches='tight', transparent = True)

unique_labels = [i+1 for i in range(21)]
plot_husl_legend(unique_labels)