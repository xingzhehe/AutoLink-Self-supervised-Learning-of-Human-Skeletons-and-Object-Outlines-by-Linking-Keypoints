import torch
import torch.nn.functional as F
from torch import nn
from typing import Union
import pytorch_lightning as pl


def gen_grid2d(grid_size: int, left_end: float=-1, right_end: float=1) -> torch.Tensor:
    """
    Generate a grid of size (grid_size, grid_size, 2) with coordinate values in the range [left_end, right_end]
    """
    x = torch.linspace(left_end, right_end, grid_size)
    x, y = torch.meshgrid([x, x], indexing='ij')
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid


def draw_lines(paired_joints: torch.Tensor, heatmap_size: int=16, thick: Union[float, torch.Tensor]=1e-2) -> torch.Tensor:
    """
    Draw lines on a grid.
    :param paired_joints: (batch_size, n_points, 2, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    bs, n_points, _, _ = paired_joints.shape
    start = paired_joints[:, :, 0, :]   # (batch_size, n_points, 2)
    end = paired_joints[:, :, 1, :]     # (batch_size, n_points, 2)
    paired_diff = end - start           # (batch_size, n_points, 2)
    grid = gen_grid2d(heatmap_size).to(paired_joints.device).reshape(1, 1, -1, 2)
    diff_to_start = grid - start.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)
    # (batch_size, n_points, heatmap_size**2)
    t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / (1e-8+paired_diff.square().sum(dim=-1, keepdim=True))

    diff_to_end = grid - end.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)

    before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
    after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
    between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

    squared_dist = (before_start + after_end + between_start_end).reshape(bs, n_points, heatmap_size, heatmap_size)
    heatmaps = torch.exp(- squared_dist / thick)
    return heatmaps


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hyper_paras: pl.LightningModule.hparams) -> None:
        super().__init__()
        self.n_parts = hyper_paras['n_parts']
        self.thick = hyper_paras['thick']
        self.sklr = hyper_paras['sklr']
        self.skeleton_idx = torch.triu_indices(self.n_parts, self.n_parts, offset=1)
        self.n_skeleton = len(self.skeleton_idx[0])

        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        skeleton_scalar = (torch.randn(self.n_parts, self.n_parts) / 10 - 4) / self.sklr
        self.skeleton_scalar = nn.Parameter(skeleton_scalar, requires_grad=True)

        self.down0 = nn.Sequential(
            nn.Conv2d(3 + 1, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down1 = DownBlock(64, 128)  # 64
        self.down2 = DownBlock(128, 256)  # 32
        self.down3 = DownBlock(256, 512)  # 16
        self.down4 = DownBlock(512, 512)  # 8

        self.up1 = UpBlock(512, 512)  # 16
        self.up2 = UpBlock(512 + 512, 256)  # 32
        self.up3 = UpBlock(256 + 256, 128)  # 64
        self.up4 = UpBlock(128 + 128, 64)  # 64

        self.conv = nn.Conv2d(64+64, 3, kernel_size=(3, 3), padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def skeleton_scalar_matrix(self) -> torch.Tensor:
        """
        Give the skeleton scalar matrix
        :return: (n_parts, n_parts)
        """
        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        skeleton_scalar = skeleton_scalar + skeleton_scalar.transpose(1, 0)
        return skeleton_scalar

    def rasterize(self, keypoints: torch.Tensor, output_size: int=128) -> torch.Tensor:
        """
        Generate edge heatmap from keypoints, where edges are weighted by the learned scalars.
        :param keypoints: (batch_size, n_points, 2)
        :return: (batch_size, 1, heatmap_size, heatmap_size)
        """

        paired_joints = torch.stack([keypoints[:, self.skeleton_idx[0], :2], keypoints[:, self.skeleton_idx[1], :2]], dim=2)

        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        skeleton_scalar = skeleton_scalar[self.skeleton_idx[0], self.skeleton_idx[1]].reshape(1, self.n_skeleton, 1, 1)

        skeleton_heatmap_sep = draw_lines(paired_joints, heatmap_size=output_size, thick=self.thick)
        skeleton_heatmap_sep = skeleton_heatmap_sep * skeleton_scalar.reshape(1, self.n_skeleton, 1, 1)
        skeleton_heatmap = skeleton_heatmap_sep.max(dim=1, keepdim=True)[0]
        return skeleton_heatmap

    def forward(self, input_dict: dict) -> dict:
        skeleton_heatmap = self.rasterize(input_dict['keypoints'])

        x = torch.cat([input_dict['damaged_img'] * self.alpha, skeleton_heatmap], dim=1)

        down_128 = self.down0(x)
        down_64 = self.down1(down_128)
        down_32 = self.down2(down_64)
        down_16 = self.down3(down_32)
        down_8 = self.down4(down_16)
        up_8 = down_8
        up_16 = torch.cat([self.up1(up_8), down_16], dim=1)
        up_32 = torch.cat([self.up2(up_16), down_32], dim=1)
        up_64 = torch.cat([self.up3(up_32), down_64], dim=1)
        up_128 = torch.cat([self.up4(up_64), down_128], dim=1)
        img = self.conv(up_128)

        input_dict['heatmap'] = skeleton_heatmap
        input_dict['img'] = img
        return input_dict


if __name__ == '__main__':
    model = Decoder({'z_dim': 256, 'n_parts': 10, 'n_embedding': 128, 'tau': 0.01})
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
