import os

import h5py
import torch
import torch.utils.data
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'flower.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])

        self.imgs = torch.cat((self.imgs, self.imgs, self.imgs), dim=0)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255)}

    def __len__(self):
        return len(self.imgs)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'flower.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255), 'keypoints': torch.tensor(0)}

    def __len__(self):
        return len(self.imgs)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        data_file = 'flower.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test_img'][...])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        return {'img': self.transform(self.imgs[index].float() / 255), 'keypoints': torch.tensor(0)}

    def __len__(self):
        return len(self.imgs)


def test_epoch_end(batch_list_list):
    raise NotImplementedError()
