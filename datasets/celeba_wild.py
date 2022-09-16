import os

import h5py
import torch
import torch.utils.data
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_landmark'][...])     # [0, 1]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255)}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['mafl_train_img'][...])
            self.keypoints = torch.from_numpy(hf['mafl_train_landmark'][...])   # [0, 1]
        self.eye_distance = (self.keypoints[:, 0, :] - self.keypoints[:, 1, :]).norm(dim=-1)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255),
                  'keypoints': self.keypoints[idx],
                  'eye_distance': self.eye_distance[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'celeba_wild.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['mafl_test_img'][...])
            self.keypoints = torch.from_numpy(hf['mafl_test_landmark'][...])   # [0, 1]
        self.eye_distance = (self.keypoints[:, 0, :] - self.keypoints[:, 1, :]).norm(dim=-1)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255),
                  'keypoints': self.keypoints[idx],
                  'eye_distance': self.eye_distance[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


def regress_kp(batch_list):
    X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in batch_list])
    eye_distance = torch.cat([batch['eye_distance'] for batch in batch_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp*2)
    y = y.reshape(batch_size, n_gt_kp*2)
    try:
        beta = (X.T @ X).inverse() @ X.T @ y
    except:
        beta = (X.T @ X + torch.eye(n_det_kp*2).to(X)).inverse() @ X.T @ y
    unnormalized_loss = (X @ beta - y).reshape(batch_size, n_gt_kp, 2).norm(dim=-1)
    normalized_loss = (unnormalized_loss / eye_distance.unsqueeze(1)).mean()
    return {'val_loss': normalized_loss, 'beta': beta}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    beta = regress_kp(valid_list)['beta']

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    y = torch.cat([batch['keypoints'] for batch in test_list])
    eye_distance = torch.cat([batch['eye_distance'] for batch in test_list])
    n_det_kp = X.shape[1]
    n_gt_kp = y.shape[1]
    batch_size = X.shape[0]
    X = X.reshape(batch_size, n_det_kp * 2)
    y = y.reshape(batch_size, n_gt_kp * 2)
    unnormalized_loss = (X @ beta - y).reshape(batch_size, n_gt_kp, 2).norm(dim=-1)
    normalized_loss = (unnormalized_loss / eye_distance.unsqueeze(1)).mean()
    return {'val_loss': normalized_loss, 'beta': beta}
