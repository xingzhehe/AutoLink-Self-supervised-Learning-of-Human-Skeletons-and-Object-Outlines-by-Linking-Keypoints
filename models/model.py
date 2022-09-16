import importlib

import pytorch_lightning as pl
import torch.utils.data
import wandb

from utils.loss import VGGPerceptualLoss
from visualization import *


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = importlib.import_module('models.' + self.hparams.encoder).Encoder(self.hparams)
        self.decoder = importlib.import_module('models.' + self.hparams.decoder).Decoder(self.hparams)
        self.batch_size = self.hparams.batch_size
        self.test_func = importlib.import_module('datasets.' + self.hparams.dataset).test_epoch_end

        self.vgg_loss = VGGPerceptualLoss()

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        self.vgg_loss.eval()
        out_batch = self.decoder(self.encoder(batch))

        perceptual_loss = self.vgg_loss(out_batch['img'], batch['img'])

        self.log("perceptual_loss", perceptual_loss)
        self.log("alpha", self.decoder.alpha.detach().cpu())
        return perceptual_loss

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, outputs):
        self.log("val_loss", -self.global_step)
        imgs = denormalize(outputs[0]['img']).cpu()
        recon_batch = self.decoder(self.encoder(outputs[0]))
        scaled_kp = recon_batch['keypoints'] * self.hparams.image_size / 2 + self.hparams.image_size / 2

        heatmap = recon_batch['heatmap'].cpu()
        heatmap_overlaid = torch.cat([heatmap] * 3, dim=1) / heatmap.max()
        heatmap_overlaid = torch.clamp(heatmap_overlaid + imgs * 0.5, min=0, max=1)

        self.logger.experiment.log({'generated': [wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='original_image'),
                                                  wandb.Image(draw_img_grid(denormalize(recon_batch['img']).cpu()), caption='reconstructed'),
                                                  wandb.Image(draw_img_grid(heatmap_overlaid.cpu()), caption='heatmap_overlaid'),
                                                  wandb.Image(draw_kp_grid_unnorm(recon_batch['heatmap'], scaled_kp), caption='heatmap'),
                                                  wandb.Image(wandb.Image(draw_kp_grid(imgs, scaled_kp)), caption='keypoints'),
                                                  wandb.Image(draw_matrix(recon_batch['skeleton_scalar_matrix'].detach().cpu().numpy()), caption='skeleton_scalar')]})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        kp = self.encoder(batch)['keypoints']
        out_batch = batch.copy()
        out_batch['det_keypoints'] = kp
        return out_batch

    def test_epoch_end(self, outputs):
        outputs = self.test_func(outputs)
        self.print("test_loss", outputs['val_loss'])
        self.log("test_loss", outputs['val_loss'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        return optimizer
