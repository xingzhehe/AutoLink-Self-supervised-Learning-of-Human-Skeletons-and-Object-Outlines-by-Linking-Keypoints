import torch
import torch.utils.data
import torchvision
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0], 'keypoints': torch.tensor(0)}
        return sample

    def __len__(self):
        return len(self.imgs)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0], 'keypoints': torch.tensor(0)}
        return sample

    def __len__(self):
        return len(self.imgs)


def test_epoch_end(batch_list_list):
    raise NotImplementedError()
