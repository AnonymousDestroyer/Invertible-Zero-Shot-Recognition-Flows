import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.linalg import norm
import wandb
import torch.nn.functional as F


def make_toy_graph(x, epoch, fit=False, show=False, save=False, path="plots/"):
    """
    Used to make graphs for groundtruth and generated
    """
    plt.title(rf'Toy Data {epoch}')
    plt.xlabel(r'')
    plt.ylabel(r'')
    a = plt.scatter(x[0][:, 0], x[0][:, 1], alpha=0.5)
    b = plt.scatter(x[1][:, 0], x[1][:, 1], alpha=0.5)
    c = plt.scatter(x[2][:, 0], x[2][:, 1], alpha=0.5)
    u = plt.scatter(x[3][:, 0], x[3][:, 1], alpha=0.5)
    if fit:
        plt.axis((-2, 2, -2, 2))
    plt.legend((a, b, c, u),
               ('Seen A', 'Seen B', 'Seen C', 'Unseen'),
               scatterpoints=1,
               fontsize=12,
               bbox_to_anchor=(1.04, 1),
               borderaxespad=0,
               fancybox=True,
               shadow=True)
    if show:
        plt.show()
    if save:
        # _, _, filenames = next(walk(rf'{path}'))
        #
        # increment = max([int(''.join(filter(str.isdigit, name))) for name in filenames]) + 1 if filenames else 0

        # plt.savefig(f'{path}plot_{increment}.png')
        wandb.log({f'plot_{epoch}.png': plt})

        #   wandb.log({f'{path}plot_{increment}.png': plt})
    plt.clf()
