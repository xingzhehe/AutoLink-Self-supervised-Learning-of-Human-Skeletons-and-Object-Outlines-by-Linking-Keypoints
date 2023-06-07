import argparse
import importlib
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datasets.base_dataset import DataModule

## 1722_95
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--project', type=str, default='AutoLink')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--encoder', type=str, default='encoder')
    parser.add_argument('--decoder', type=str, default='decoder')
    parser.add_argument('--n_parts', type=int, default=4)
    parser.add_argument('--missing', type=float, default=0.8)
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--thick', type=float, default=1e-3)
    parser.add_argument('--sklr', type=float, default=512)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='../data/celeba_wild')
    parser.add_argument('--dataset', type=str, default='celeba_wild')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)

    args = parser.parse_args()

    args.log = '{0}_k{1}_m{2}_b{3}_t{4}_sklr{5}'.format(args.dataset, args.n_parts, args.missing, args.block, args.thick, args.sklr)
    wandb_logger = WandbLogger(name=args.log, project=args.project)
    model = importlib.import_module('models.' + args.model).Model(**vars(args))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                       dirpath=os.path.join('checkpoints', model.hparams.log),
                                                       filename='model')
    datamodule = DataModule(model.hparams.dataset, model.hparams.data_root, model.hparams.image_size, model.hparams.batch_size,
                            model.hparams.num_workers)

    val_check_every_iter = 1000
    val_check_every_epoch = val_check_every_iter * model.hparams.batch_size / len(datamodule.train_dataset)
    if val_check_every_epoch < 1:
        val_check_interval = val_check_every_iter
        val_check_every_epoch = 1
    else:
        val_check_interval = 1.0
        val_check_every_epoch = round(val_check_every_epoch)

    trainer = pl.Trainer(accelerator='gpu', gpus=model.hparams.gpus, num_nodes=model.hparams.num_nodes,
                         fast_dev_run=model.hparams.debug,
                         max_steps=20001, precision=16, sync_batchnorm=True, #strategy='ddp',
                         limit_val_batches=1,
                         val_check_interval=val_check_interval,
                         check_val_every_n_epoch=val_check_every_epoch,
                         callbacks=checkpoint_callback, logger=wandb_logger,
                         )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
