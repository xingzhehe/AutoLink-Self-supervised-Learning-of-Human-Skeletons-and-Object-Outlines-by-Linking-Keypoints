import torch
import torch.nn.functional as F
from torch import nn


def gen_grid2d(grid_size, device='cpu', left_end=-1, right_end=1):
    x = torch.linspace(left_end, right_end, grid_size).to(device)
    x, y = torch.meshgrid([x, x])
    grid = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).reshape(grid_size, grid_size, 2)
    return grid


def draw_lines(paired_joints, heatmap_size=16, thick=1e-2):
    """
    :param paired_joints: (batch_size, n_points, 2, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    bs, n_points, _, _ = paired_joints.shape
    start = paired_joints[:, :, 0, :]   # (batch_size, n_points, 2)
    end = paired_joints[:, :, 1, :]     # (batch_size, n_points, 2)
    paired_diff = end - start           # (batch_size, n_points, 2)
    grid = gen_grid2d(heatmap_size, device=paired_joints.device).reshape(1, 1, -1, 2)
    diff_to_start = grid - start.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)
    # (batch_size, n_points, heatmap_size**2)
    t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / paired_diff.square().sum(dim=-1, keepdim=True)

    diff_to_end = grid - end.unsqueeze(-2)  # (batch_size, n_points, heatmap_size**2, 2)

    before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
    after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
    between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

    squared_dist = (before_start + after_end + between_start_end).reshape(bs, n_points, heatmap_size, heatmap_size)
    heatmaps = torch.exp(- squared_dist / thick)
    return heatmaps


def gen_keypoints(joints, heatmap_size=16, thick=1e-2):
    """
    :param joints: (batch_size, n_points, 2)
    :return: (batch_size, n_points, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||
    """
    batch_size, n_points, _ = joints.shape
    grid = gen_grid2d(heatmap_size, device=joints.device).reshape(1, -1, 2)
    diff = joints.unsqueeze(-2) - grid.unsqueeze(-3)  # (batch_size, n_points, 4**2, 2)
    dist = diff.square().sum(dim=-1)
    heatmaps = torch.exp(-dist / thick)
    return heatmaps.reshape(batch_size, n_points, heatmap_size, heatmap_size)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.net(x)


class Conv2DMod(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_channel, in_channel, 1, 1)))
        self.bias = nn.Parameter(torch.zeros(out_channel))
        nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        weight = self.weight / self.weight.std(dim=0, keepdim=True)
        x = F.conv2d(x, weight, bias=self.bias)
        return x


class Decoder(nn.Module):
    def __init__(self, hyper_paras):
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
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down1 = DownBlock(64, 128)  # 64
        self.down2 = DownBlock(128, 256)  # 32
        self.down3 = DownBlock(256, 512)  # 16
        self.down4 = DownBlock(512, 512)  # 8

        # self.res = nn.Sequential(
        #     ResBlock(512, 512),
        #     ResBlock(512, 512),
        #     ResBlock(512, 512),
        #     ResBlock(512, 512),
        # )
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

    def forward(self, input_dict):
        joints = input_dict['keypoints']
        bs = joints.shape[0]

        paired_joints = torch.cat([
            joints.reshape(bs, self.n_parts, 1, 2).expand(-1, -1, self.n_parts, -1).reshape(bs, self.n_parts ** 2, 1, 2),
            joints.reshape(bs, 1, self.n_parts, 2).expand(-1, self.n_parts, -1, -1).reshape(bs, self.n_parts ** 2, 1, 2)
        ], dim=2).reshape(bs, self.n_parts, self.n_parts, 2, 2)[:, self.skeleton_idx[0], self.skeleton_idx[1], :, :]

        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        input_dict['skeleton_scalar_matrix'] = skeleton_scalar + skeleton_scalar.transpose(1, 0)
        skeleton_scalar = skeleton_scalar[self.skeleton_idx[0], self.skeleton_idx[1]].reshape(1, self.n_skeleton, 1, 1)

        skeleton_heatmap_sep = draw_lines(paired_joints, heatmap_size=input_dict['damaged_img'].shape[-1], thick=self.thick)
        skeleton_heatmap_sep = skeleton_heatmap_sep * skeleton_scalar.reshape(1, self.n_skeleton, 1, 1)
        skeleton_heatmap = skeleton_heatmap_sep.max(dim=1, keepdim=True)[0]

        # import matplotlib.pyplot as plt
        # for i in range(skeleton_heatmap.shape[1]):
        #     plt.imshow(skeleton_heatmap[0, i].detach().cpu())
        #     plt.colorbar()
        #     paired_joints_ = paired_joints.detach().cpu() * 63.5 + 63.5
        #     plt.scatter(paired_joints_[0, i, :, 1], paired_joints_[0, i, :, 0])
        #     plt.show()

        x = torch.cat([input_dict['damaged_img'] * self.alpha, skeleton_heatmap], dim=1)

        down_128 = self.down0(x)
        down_64 = self.down1(down_128)
        down_32 = self.down2(down_64)
        down_16 = self.down3(down_32)
        down_8 = self.down4(down_16)
        # up_8 = self.res(down_8)
        up_8 = down_8
        up_16 = torch.cat([self.up1(up_8), down_16], dim=1)
        up_32 = torch.cat([self.up2(up_16), down_32], dim=1)
        up_64 = torch.cat([self.up3(up_32), down_64], dim=1)
        up_128 = torch.cat([self.up4(up_64), down_128], dim=1)
        img = self.conv(up_128)

        input_dict['heatmap'] = skeleton_heatmap
        input_dict['heatmap_sep'] = skeleton_heatmap_sep
        input_dict['img'] = img
        return input_dict


if __name__ == '__main__':
    model = Decoder({'z_dim': 256, 'n_parts': 10, 'n_embedding': 128, 'tau': 0.01})
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))