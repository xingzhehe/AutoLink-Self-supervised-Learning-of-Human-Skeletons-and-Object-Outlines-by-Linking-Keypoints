import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from matplotlib import colors


def get_part_color(n_parts):
    colormap = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle',
                'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle')[:n_parts]
    part_color = []
    for i in range(n_parts):
        part_color.append(colors.to_rgb(colormap[i]))
    part_color = np.array(part_color)

    return part_color


def denormalize(img):
    mean = torch.tensor((0.5, 0.5, 0.5), device=img.device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5), device=img.device).reshape(1, 3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, min=0, max=1)
    return img


def draw_matrix(mat):
    fig = plt.figure()
    sns.heatmap(mat, annot=True, fmt='.2f', cmap="YlGnBu")

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot


def draw_kp_grid(img, kp):
    kp_color = get_part_color(kp.shape[1])
    img = img[:64].permute(0, 2, 3, 1).detach().cpu()
    kp = kp.detach().cpu()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(sample, vmin=0, vmax=1)
        ax.scatter(kp[i, :, 1], kp[i, :, 0], c=kp_color, s=20, marker='+')

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot


def draw_kp_grid_unnorm(img, kp):
    kp_color = get_part_color(kp.shape[1])
    img = img[:64].permute(0, 2, 3, 1).detach().cpu()
    kp = kp.detach().cpu()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(sample)
        ax.scatter(kp[i, :, 1], kp[i, :, 0], c=kp_color, s=20, marker='+')

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot


def draw_img_grid(img):
    img = img[:64].detach().cpu()
    nrow = min(8, img.shape[0])
    img = torchvision.utils.make_grid(img[:64], nrow=nrow).permute(1, 2, 0)
    return torch.clamp(img * 255, min=0, max=255).numpy().astype(np.uint8)
