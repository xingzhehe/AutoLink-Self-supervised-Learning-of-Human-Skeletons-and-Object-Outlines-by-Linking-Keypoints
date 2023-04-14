import importlib
import PIL
import pytorch_lightning as pl
import torch.utils.data
import wandb
from typing import Union
from torchvision import transforms
from utils_.loss import VGGPerceptualLoss
from utils_.visualization import *
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = importlib.import_module('models.' + self.hparams.encoder).Encoder(self.hparams)
        self.decoder = importlib.import_module('models.' + self.hparams.decoder).Decoder(self.hparams)
        self.batch_size = self.hparams.batch_size
        self.test_func = importlib.import_module('datasets.' + self.hparams.dataset).test_epoch_end

        self.vgg_loss = VGGPerceptualLoss()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def forward(self, x: PIL.Image.Image) -> PIL.Image.Image:
        """
        :param x: a PIL image
        :return: an edge map of the same size as x with values in [0, 1] (normalized by max)
        """
        w, h = x.size
        x = self.transform(x).unsqueeze(0)
        x = x.to(self.device)
        kp = self.encoder({'img': x})['keypoints']
        edge_map = self.decoder.rasterize(kp, output_size=64)
        bs = edge_map.shape[0]
        edge_map = edge_map / (1e-8 + edge_map.reshape(bs, 1, -1).max(dim=2, keepdim=True)[0].reshape(bs, 1, 1, 1))
        edge_map = torch.cat([edge_map] * 3, dim=1)
        edge_map = F.interpolate(edge_map, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.clamp(edge_map + (x * 0.5 + 0.5)*0.5, min=0, max=1)
        x = transforms.ToPILImage()(x[0].detach().cpu())

        fig = plt.figure(figsize=(1, h/w), dpi=w)
        fig.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(x)
        kp = kp[0].detach().cpu() * 0.5 + 0.5
        kp[:, 1] *= w
        kp[:, 0] *= h
        plt.scatter(kp[:, 1], kp[:, 0], s=min(w/h, min(1, h/w)), marker='o')
        ncols, nrows = fig.canvas.get_width_height()
        fig.canvas.draw()
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
        plt.close(fig)
        return plot

    def training_step(self, batch, batch_idx):
        self.vgg_loss.eval()
        out_batch = self.decoder(self.encoder(batch, need_masked_img=True))

        perceptual_loss = self.vgg_loss(out_batch['img'], batch['img'])

        self.log("perceptual_loss", perceptual_loss)
        self.log("alpha", self.decoder.alpha.detach().cpu())
        return perceptual_loss

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, outputs):
        self.log("val_loss", -self.global_step*1.0)
        imgs = denormalize(outputs[0]['img']).cpu()
        recon_batch = self.decoder(self.encoder(outputs[0], need_masked_img=True))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)

        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(outputs[0]['img']).cpu()), caption='original_image'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(imgs, scaled_kp)), caption='keypoints'),
                                                  wandb.Image(draw_matrix(self.decoder.skeleton_scalar_matrix().detach().cpu().numpy()), caption='skeleton_scalar')]})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        kp = self.encoder(batch)['keypoints']
        out_batch = {'keypoints': batch['keypoints'].cpu(), 'det_keypoints': kp.cpu()}
        return out_batch

    def test_epoch_end(self, outputs):
        outputs = self.test_func(outputs)
        self.print("test_loss", outputs['val_loss'])
        self.log("test_loss", outputs['val_loss'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        return optimizer
