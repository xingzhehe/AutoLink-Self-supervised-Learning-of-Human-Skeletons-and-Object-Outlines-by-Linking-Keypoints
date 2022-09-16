import os

import numpy as np
import scipy.io
import torch
import torch.utils.data
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

        self.data_root = data_root

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        return {'img': self.transform(img)}

    def __len__(self):
        return len(self.samples)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.data_root = data_root

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(self.data_root, 'S{}'.format(subject_index), 'Landmarks',
                                      folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        return {'img': self.transform(img), 'keypoints': keypoints}

    def __len__(self):
        return len(self.samples)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.data_root = data_root

        self.samples = []

        for subject_index in [11]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(self.data_root, 'S{}'.format(subject_index), 'Landmarks',
                                                  folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        return {'img': self.transform(img), 'keypoints': keypoints}

    def __len__(self):
        return len(self.samples)


correspondences = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (17, 25), (18, 26), (19, 27), (20, 28), (21, 28), (22, 30), (23, 31)]


def swap_points(points):
    """
    points: B x N x D
    """
    permutation = list(range((points.shape[1])))
    for a, b in correspondences:
        permutation[a] = b
        permutation[b] = a
    new_points = points[:, permutation, :]
    return new_points


def regress_kp(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)
    XTXXT = (X.T @ X).inverse() @ X.T

    while True:
        beta = XTXXT @ y
        pred_y = X @ beta

        dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        swaped_y = swap_points(y.reshape(batch_size, n_gt_kp, 2)).reshape(batch_size, n_gt_kp * 2)
        swaped_dist = (pred_y - swaped_y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        should_swap = dist > swaped_dist

        if should_swap.sum() > 10:
            y[should_swap] = swaped_y[should_swap]
        else:
            break

    dist_mean = dist.mean()
    dist_std = dist.std()
    chosen = dist < dist_mean + 3 * dist_std
    X, y = X[chosen], y[chosen]

    beta = (X.T @ X).inverse() @ X.T @ y
    pred_y = X @ beta
    dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

    return {'val_loss': dist.mean(), 'beta': beta}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    beta = regress_kp(valid_list)['beta']

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in test_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)

    pred_y = X @ beta

    while True:
        dist = (pred_y - y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)
        swaped_y = swap_points(y.reshape(batch_size, n_gt_kp, 2)).reshape(batch_size, n_gt_kp * 2)
        swaped_dist = (pred_y - swaped_y).reshape(X.shape[0], n_gt_kp, 2).norm(dim=2).mean(dim=1)

        should_swap = dist > swaped_dist

        if should_swap.sum() > 10:
            y[should_swap] = swaped_y[should_swap]
        else:
            break

    return {'val_loss': dist.mean(), 'beta': beta}
