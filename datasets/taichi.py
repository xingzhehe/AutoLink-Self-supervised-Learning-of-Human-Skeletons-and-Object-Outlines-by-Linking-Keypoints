import os

import numpy as np
import pandas
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.imgs = []
        self.poses = []

        with open(os.path.join(data_root, 'landmark', 'taichi_train_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(data_root, 'eval_images', 'taichi-256', 'train', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.poses.append(pose_file.value[i])  # [0, 255]

        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torch.tensor(np.array(self.imgs)).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.imgs = []
        self.segs = []
        self.poses = []

        with open(os.path.join(data_root, 'landmark', 'taichi_test_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(data_root, 'eval_images', 'taichi-256', 'test', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            seg = Image.open(os.path.join(data_root, 'taichi-test-masks', image_file))
            seg = seg.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.segs.append(np.asarray(seg) / 255)
            self.poses.append(pose_file.value[i])  # [0, 255]

        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torch.tensor(np.array(self.imgs)).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.segs = torch.tensor(self.segs).int()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'seg': self.segs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)


def regress_kp(batch_list):
    train_X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    train_X = train_X * 255
    train_y = torch.cat([batch['keypoints'] for batch in batch_list])
    scores = []
    num_gnd_kp = 18
    betas = []
    for i in range(num_gnd_kp):
        for j in range(2):
            index = (train_y[:, i, j] + 1).abs() > 1e-6
            features = train_X[index]
            features = features.reshape(features.shape[0], -1)
            label = train_y[index, i, j]
            features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
            try:
                beta = (features.T @ features).inverse() @ features.T @ label
            except:
                beta = (features.T @ features + torch.eye(features.shape[-1]).to(features)).inverse() @ features.T @ label
            betas.append(beta)

            pred_label = features @ beta
            score = (pred_label - label).abs().mean()
            scores.append(score.item())
    return {'val_loss': np.sum(scores), 'beta': betas}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    betas = regress_kp(valid_list)['beta']
    num_gnd_kp = 18
    scores = []

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in test_list])

    beta_index = 0

    for i in range(num_gnd_kp):
        for j in range(2):
            index_test = (y[:, i, j] + 1).abs() > 1e-6
            features = X[index_test]
            features = features.reshape(features.shape[0], -1)
            features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
            label = y[index_test, i, j]
            pred_label = features @ betas[beta_index]
            score = (pred_label - label).abs().mean()
            scores.append(score.item())
            beta_index += 1

    return {'val_loss': np.sum(scores)}
