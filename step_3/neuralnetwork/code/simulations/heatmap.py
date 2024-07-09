import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict




def plot_stacked_bar(row_name, col_name, data, xlabel, ylabel, name):
    fig, axs = plt.subplots(1, len(row_name), sharex=False, sharey=True)
    divnorm = colors.TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.8)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i in range(len(row_name)):
        vec1 = np.array(data[row_name[i]])
        vec2 = vec1.reshape(vec1.shape[0], 1)
        sns.heatmap(vec2, ax=axs[i],
                    annot=False,
                    cbar=i == 0,
                    cmap="bwr",
                    norm = divnorm,
                    xticklabels=[row_name[i]],
                    yticklabels=col_name,
                    vmin=0.8, vmax=1.8,
                    cbar_ax=None if i else cbar_ax)

    # fig.tight_layout(rect=[0, 0, .9, 1])
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    plt.savefig(name, format="pdf", transparent=True)

    plt.show()
    plt.close()