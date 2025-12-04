import numpy as np
import matplotlib.pyplot as plt

def figsize(scale, nplots=1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                     # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0          # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = nplots * fig_width * golden_mean        # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

# 更新figure的大小设置
plt.rcParams["figure.figsize"] = figsize(1.0)

def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop=True):
    if crop:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))
