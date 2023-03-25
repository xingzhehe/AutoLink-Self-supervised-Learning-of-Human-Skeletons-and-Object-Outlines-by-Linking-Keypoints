import argparse
import importlib
import os

import imageio
import pytorch_lightning as pl
import torch.nn.functional as F

from utils_.visualization import *

torch.set_grad_enabled(False)
device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')


image_size = 128
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def draw_img_kp_heatmap_skeleton(img, kp, heatmap, kp_color):
    fig = plt.figure(figsize=(4, 1), dpi=128)
    gs = gridspec.GridSpec(1, 4)
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

    # draw heatmap
    ax = plt.subplot(gs[3])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(heatmap)

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.tight_layout(pad=0)

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_log_small', type=str, default='celeba_wild/celeba_wild_k16_m0.8_b16_t0.001_sklr512')
    parser.add_argument('--face_log_large', type=str, default='celeba_wild/celeba_wild_k32_m0.8_b16_t0.00075_sklr512')
    parser.add_argument('--body_log_small', type=str, default='taichi/taichi_k10_m0.8_b16_t0.0005_sklr512')
    parser.add_argument('--body_log_large', type=str, default='taichi/taichi_k32_m0.8_b16_t0.0005_sklr512')
    parser.add_argument('--model', type=str, default='model')

    args = parser.parse_args()

    pl.utilities.seed.seed_everything(0)

    face_videos = [
        imageio.get_reader('../../data/voxceleb_raw/test/mp4/id01593/_gyaAyVi6SA/00344.mp4'),
        imageio.get_reader('../../data/voxceleb_raw/test/mp4/id08392/CRFu2BvRzVI/00068.mp4'),
        imageio.get_reader('../../data/voxceleb_raw/test/mp4/id02542/gfgfOn-MeS4/00062.mp4'),
        imageio.get_reader('../../data/voxceleb_raw/test/mp4/id01228/FiIjEyg3qe0/00108.mp4'),
    ]
    print(*[face_video.count_frames() for face_video in face_videos])

    body_videos = [
        '../../data/taichi/test/GQ0ef8nh9H8#002530#002818.mp4',
        '../../data/taichi/test/mndSqTrxpts#002234#002406.mp4',
        '../../data/taichi/test/OiblkvkAHWM#002280#002440.mp4',
        '../../data/taichi/test/gaccfn5JB4Y#001713#001845.mp4',
    ]

    models = {
        'face_model_small': importlib.import_module('models.' + args.model).Model.load_from_checkpoint(
            os.path.join('checkpoints', args.face_log_small, 'model.ckpt')).to(device),
        'face_model_large': importlib.import_module('models.' + args.model).Model.load_from_checkpoint(
            os.path.join('checkpoints', args.face_log_large, 'model.ckpt')).to(device),
        'body_model_small': importlib.import_module('models.' + args.model).Model.load_from_checkpoint(
            os.path.join('checkpoints', args.body_log_small, 'model.ckpt')).to(device),
        'body_model_large': importlib.import_module('models.' + args.model).Model.load_from_checkpoint(
            os.path.join('checkpoints', args.body_log_large, 'model.ckpt')).to(device)
    }

    for key, model in models.items():
        model.eval()
        model.decoder.thick = 5e-4  # for visualization only

    kp_color = {key: get_part_color(model.hparams.n_parts) for key, model in models.items()}

    os.makedirs('gif', exist_ok=True)

    writer = imageio.get_writer(os.path.join('gif', 'detection.mp4'.format()), mode='I', fps=25)

    for i in range(100):
        face_drawns = []
        for j in range(len(face_videos)):
            face_img = torch.from_numpy(np.asarray(face_videos[j].get_data(i)) / 255).permute(2, 0, 1).float()
            face_img256 = F.interpolate(face_img.unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1, 2, 0)
            face_img = transform(face_img)
            encoded = models['face_model_large'].encoder({'img': face_img.unsqueeze(0).to(device)})
            decoded = models['face_model_large'].decoder(encoded)
            scaled_kp = decoded['keypoints'][0].cpu() * 128 + 128
            face_drawn = draw_img_kp_heatmap_skeleton(img=face_img256,
                                                      kp=scaled_kp,
                                                      heatmap=F.interpolate(decoded['heatmap'], size=256, mode='bilinear')[0, 0].cpu(),
                                                      kp_color=kp_color['face_model_large'])
            face_drawns.append(face_drawn)
        face_drawns = np.concatenate(
            [np.concatenate([face_drawns[0], face_drawns[1]], axis=0),
             np.concatenate([face_drawns[2], face_drawns[3]], axis=0)],
            axis=1)

        body_drawns = []
        for j in range(len(body_videos)):
            body_img256 = imageio.imread(os.path.join(body_videos[j], '{}.png'.format(str(i).zfill(7))))
            body_img256 = torch.from_numpy(np.asarray(body_img256) / 255).float()
            body_img = body_img256.permute(2, 0, 1)
            body_img = transform(body_img)
            encoded = models['body_model_large'].encoder({'img': body_img.unsqueeze(0).to(device)})
            decoded = models['body_model_large'].decoder(encoded)
            scaled_kp = decoded['keypoints'][0].cpu() * 128 + 128
            body_drawn = draw_img_kp_heatmap_skeleton(img=body_img256,
                                                      kp=scaled_kp,
                                                      heatmap=F.interpolate(decoded['heatmap'], size=256, mode='bilinear')[0, 0].cpu(),
                                                      kp_color=kp_color['body_model_large'])
            body_drawns.append(body_drawn)
        body_drawns = np.concatenate(
            [np.concatenate([body_drawns[0], body_drawns[1]], axis=0),
             np.concatenate([body_drawns[2], body_drawns[3]], axis=0)],
            axis=1)
        drawn = np.concatenate([face_drawns, body_drawns], axis=0)
        writer.append_data(drawn)
    writer.close()
