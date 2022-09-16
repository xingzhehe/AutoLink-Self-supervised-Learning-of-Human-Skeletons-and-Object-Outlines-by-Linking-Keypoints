import argparse
import importlib
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datasets.base_dataset import DataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='cub_k4/cub_k4_m0.8_b16_t0.05_sklr512')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--data_root', type=str, default='../data/cub')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    pl.utilities.seed.seed_everything(0)

    model = importlib.import_module('models.' + args.model).Model.load_from_checkpoint(os.path.join('checkpoints', args.log, 'model.ckpt'))

    trainer = pl.Trainer(gpus=1, num_processes=6,
                         precision=16, sync_batchnorm=True, weights_summary=None, checkpoint_callback=False,
                         logger=WandbLogger(name=args.log, project="AutoLink_test"),
                         )
    # datamodule = DataModule(model.hparams.dataset, args.data_root, model.hparams.image_size, args.batch_size)
    datamodule = DataModule('cub_three', args.data_root, model.hparams.image_size, args.batch_size)
    trainer.test(model, datamodule=datamodule)
