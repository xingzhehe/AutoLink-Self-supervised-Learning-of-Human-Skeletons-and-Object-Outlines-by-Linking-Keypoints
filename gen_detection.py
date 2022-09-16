import argparse
import importlib
import os

import pytorch_lightning as pl
import torch.nn.functional as F
from PIL import Image

from datasets.base_dataset import DataModule
from visualization import *

torch.set_grad_enabled(False)
device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')


def save_img_kp_skeleton(img, damaged_img, kp, heatmap, kp_color, folder_name, index):
    os.makedirs(os.path.join('det', folder_name, str(index)), exist_ok=True)
    # draw image
    Image.fromarray(np.uint8(img * 255)).save(os.path.join('det', folder_name, str(index), 'img.png'))
    Image.fromarray(np.uint8(damaged_img * 255)).save(os.path.join('det', folder_name, str(index), 'damaged_img.png'))

    # draw kp
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.imshow(img)
    plt.scatter(kp[:, 1], kp[:, 0], c=kp_color, s=20, marker='o')
    plt.savefig(os.path.join('det', folder_name, str(index), 'kp.png'), dpi=128)
    plt.close(fig)

    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.imshow(heatmap)
    plt.savefig(os.path.join('det', folder_name, str(index), 'heatmap.png'), dpi=128)
    plt.close(fig)

    # draw skeleton
    heatmap_overlaid = torch.stack([heatmap] * 3, dim=2) / heatmap.max()
    heatmap_overlaid = torch.clamp(heatmap_overlaid + img * 0.5, min=0, max=1)
    Image.fromarray(np.uint8(heatmap_overlaid * 255)).save(os.path.join('det', folder_name, str(index), 'structure.png'))

    print(index)


def draw_img_kp_skeleton(img, kp, heatmap, kp_color):
    fig = plt.figure(figsize=(3, 1), dpi=128)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0, hspace=0)

    # draw image
    ax = plt.subplot(gs[0])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(img)

    # draw kp
    ax = plt.subplot(gs[1])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(img)
    plt.scatter(kp[:, 1], kp[:, 0], c=kp_color, s=20, marker='o')

    # draw skeleton
    ax = plt.subplot(gs[2])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    heatmap_overlaid = torch.stack([heatmap] * 3, dim=2) / heatmap.max()
    heatmap_overlaid = torch.clamp(heatmap_overlaid + img * 0.5, min=0, max=1)
    plt.imshow(heatmap_overlaid)

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.tight_layout(pad=0)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='horse/horse_k32_m0.8_b16_t0.00025_sklr512')
    parser.add_argument('--folder_name', type=str, default='horse_test')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--data_root', type=str, default='../../data/horse')

    args = parser.parse_args()

    model = importlib.import_module('models.' + args.model).Model.load_from_checkpoint(os.path.join('checkpoints', args.log, 'model.ckpt'))
    model = model.to(device)
    model.eval()

    model.decoder.thick = 5e-4  # for visualization only

    if 'deepfashion' in args.log or 'h36m' in args.log or 'zebra' in args.log or 'horse' in args.log or 'afhq' in args.log:
        skeleton_threshold = 0.01
        chosen_skeleton_idx = torch.triu(F.softplus(model.decoder.skeleton_scalar * 512), diagonal=1) > skeleton_threshold
        chosen_skeleton = model.decoder.skeleton_scalar[chosen_skeleton_idx]
        model.decoder.skeleton_scalar[chosen_skeleton_idx] = chosen_skeleton + 0.01

    kp_color = get_part_color(model.hparams.n_parts)

    datamodule = DataModule(model.hparams.dataset, args.data_root, model.hparams.image_size, batch_size=1).test_dataloader()[1]

    pl.utilities.seed.seed_everything(0)

    for batch_index, batch in enumerate(datamodule):
        encoded = model.encoder({'img': batch['img'].to(device)})
        decoded = model.decoder(encoded)
        scaled_kp = decoded['keypoints'][0].cpu() * model.hparams.image_size / 2 + model.hparams.image_size / 2
        # draw_img_kp_skeleton(img=batch['img'].squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5,
        #                      kp=scaled_kp,
        #                      heatmap=decoded['heatmap'][0, 0].cpu(),
        #                      kp_color=kp_color)
        save_img_kp_skeleton(img=batch['img'].squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5,
                             damaged_img=encoded['damaged_img'].squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5,
                             kp=scaled_kp,
                             heatmap=decoded['heatmap'][0, 0].cpu(),
                             kp_color=kp_color,
                             folder_name=args.folder_name,
                             index=batch_index)

        if batch_index > 200:
            break
