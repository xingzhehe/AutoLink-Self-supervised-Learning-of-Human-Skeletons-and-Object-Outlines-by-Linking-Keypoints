import importlib

import pytorch_lightning as pl
import torch
import torch.utils.data


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_root, image_size, batch_size, num_workers=6):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_root = data_root
        self.image_size = image_size
        self.num_workers = num_workers

        dataset = importlib.import_module('datasets.' + self.dataset)
        self.train_dataset = dataset.TrainSet(self.data_root, self.image_size)
        self.train_reg_dataset = dataset.TrainRegSet(self.data_root, self.image_size)
        self.test_dataset = dataset.TestSet(self.data_root, self.image_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return [torch.utils.data.DataLoader(self.train_reg_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False),
                torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)]
