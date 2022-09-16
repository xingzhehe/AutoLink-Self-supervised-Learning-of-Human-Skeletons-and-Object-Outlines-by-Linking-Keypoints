import json
import os

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_train.csv'))][1:]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img)}
        return sample

    def __len__(self):
        return len(self.img_file)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_train.csv'))][1:]

        with open(os.path.join(data_root, 'data_train.json'), 'r') as f:
            self.keypoints = json.load(f)
        self.keypoints = [self.keypoints[i]['keypoints'] for i in range(len(self.keypoints))]
        self.keypoints = torch.tensor(self.keypoints).roll(shifts=1, dims=-1)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img), 'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return len(self.img_file)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_test.csv'))][1:]

        with open(os.path.join(data_root, 'data_test.json'), 'r') as f:
            self.keypoints = json.load(f)
        self.keypoints = [self.keypoints[i]['keypoints'] for i in range(len(self.keypoints))]
        self.keypoints = torch.tensor(self.keypoints).roll(shifts=1, dims=-1)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img), 'keypoints': self.keypoints[idx]}
        return sample

    def __len__(self):
        return len(self.img_file)


def regress_kp(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp*2)
    y = y.reshape(batch_size, n_gt_kp*2)
    try:
        beta = (X.T @ X).inverse() @ X.T @ y
    except:
        print('use penalty in linear regression')
        beta = (X.T @ X + 1e-3 * torch.eye(n_det_kp*2).to(X)).inverse() @ X.T @ y
    scaled_difference = (X @ beta - y).reshape(X.shape[0], n_gt_kp, 2)
    eval_acc = (scaled_difference.norm(dim=2) < 6).float().mean()
    return {'val_loss': -eval_acc, 'beta': beta}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    beta = regress_kp(valid_list)['beta']

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in test_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)
    scaled_difference = (X @ beta - y).reshape(X.shape[0], n_gt_kp, 2)
    eval_acc = (scaled_difference.norm(dim=2) < 6).float().mean()
    return {'val_loss': -eval_acc, 'beta': beta}
