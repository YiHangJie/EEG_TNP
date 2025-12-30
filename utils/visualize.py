import numpy as np
import matplotlib.pyplot as plt

def plot_eeg(
    data,
    sfreq=None,
    ch_names=None,
    color=None,
    title=None,
    figsize=None,
    linewidth=0.8,
    grid=False,
    save_path=None,
):
    data = np.asarray(data)
    n_channels, n_times = data.shape

    if sfreq is not None and sfreq > 0:
        t = np.arange(n_times) / float(sfreq)
    else:
        t = np.arange(n_times)

    if ch_names is None:
        ch_names = [f"Ch{i}" for i in range(n_channels)]

    if figsize is None:
        # Keep a reasonable aspect ratio: scale height per channel but cap it to avoid overly tall stacks
        height = min(max(0.3 * n_channels, 3.5), 8)
        figsize = (12, height)

    fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=figsize, gridspec_kw={'hspace': 0.0})
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        ax = axes[i]
        y = data[i]
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.05 * (y_max - y_min if y_max != y_min else abs(y_min) + 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.plot(t, y, color=color, linewidth=linewidth)
        # Remove channel labels and y-axis ticks for a cleaner, compact view
        ax.set_ylabel("")
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 取消y 轴刻度
        # ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 隐藏所有坐标轴、刻度、边框
        if i == 0:
            # 第一个子图：隐藏 bottom 轴
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        elif i == n_channels - 1:
            # 最后一个子图：隐藏 top 轴
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, labeltop=False)
        else:
            # 中间子图：隐藏 top 和 bottom 轴
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if grid:
            ax.grid(False)

    if title:
        fig.suptitle(title, y=0.99, fontsize=12)
        # Tight layout with a small top margin so the title sits closer to the traces
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.965])
    else:
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

    # return fig, axes

def plot_double_eeg(
    data1,
    data2,
    sfreq=None,
    ch_names=None,
    color1=None,
    color2=None,
    title=None,
    figsize=None,
    linewidth=0.8,
    grid=False,
    save_path=None,
):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    n_channels, n_times = data1.shape

    if sfreq is not None and sfreq > 0:
        t = np.arange(n_times) / float(sfreq)
    else:
        t = np.arange(n_times)

    if ch_names is None:
        ch_names = [f"Ch{i}" for i in range(n_channels)]

    if figsize is None:
        height = min(max(0.3 * n_channels, 3.5), 8)
        figsize = (12, height)

    fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=figsize, gridspec_kw={'hspace': 0.0})
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        ax = axes[i]
        y1 = data1[i]
        y2 = data2[i]
        y1_min, y1_max = float(np.nanmin(y1)), float(np.nanmax(y1))
        y2_min, y2_max = float(np.nanmin(y2)), float(np.nanmax(y2))
        y_min = min(y1_min, y2_min)
        y_max = max(y1_max, y2_max)
        pad = 0.05 * (y_max - y_min if y_max != y_min else abs(y_min) + 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.plot(t, y1, color=color1, linewidth=linewidth)
        ax.plot(t, y2, color=color2, linewidth=linewidth)
        ax.set_ylabel("")
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 取消y 轴刻度
        # ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 隐藏所有坐标轴、刻度、边框
        if i == 0:
            # 第一个子图：隐藏 bottom 轴
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        elif i == n_channels - 1:
            # 最后一个子图：隐藏 top 轴
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, labeltop=False)
        else:
            # 中间子图：隐藏 top 和 bottom 轴
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if grid:
            ax.grid(False)

    if title:
        fig.suptitle(title, y=0.995, fontsize=12)

    # plt.subplots_adjust(hspace=0.05, top=0.95)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

    # return fig, axes
