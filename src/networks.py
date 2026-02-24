import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --- [1] SPADE Block ---
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.mlp_gamma = nn.Conv2d(
            nhidden, norm_nc, kernel_size=3, padding=1
        )
        self.mlp_beta = nn.Conv2d(
            nhidden, norm_nc, kernel_size=3, padding=1
        )

    def forward(self, x, segmap):
        # segmap을 현재 feature map 크기에 맞춰 리사이즈 (Nearest Neighbor)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")

        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = spectral_norm(
            nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        )
        self.conv_1 = spectral_norm(
            nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        )
        if self.learned_shortcut:
            self.conv_s = spectral_norm(
                nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            )

        self.spade_0 = SPADE(fin, semantic_nc)
        self.spade_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.spade_s = SPADE(fin, semantic_nc)

        # inplace=False 로 고정
        self.actvn = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x, seg):
        # Shortcut branch
        if self.learned_shortcut:
            x_s = self.spade_s(x, seg)
            x_s = self.conv_s(self.actvn(x_s))
        else:
            x_s = x

        # Residual branch
        dx = self.spade_0(x, seg)
        dx = self.actvn(dx)
        dx = self.conv_0(dx)

        dx = self.spade_1(dx, seg)
        dx = self.actvn(dx)
        dx = self.conv_1(dx)

        out = x_s + dx
        return out


# --- [2] SPADE Generator (256x256 Optimized) ---
class SPADEGenerator(nn.Module):
    """
    256x256 해상도 기준 Generator.
    - Encoder: 256 -> 128 -> 64 -> 32 -> 16 (4번 downsample)
    - Decoder: 16 -> 32 -> 64 -> 128 -> 256 (4번 upsample)
    """

    def __init__(self, input_nc=3, label_nc=4, z_dim=256):
        super().__init__()

        # 1. Encoder (입력 256px 기준)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, 3, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 -> 64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 64 -> 32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 32 -> 16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            # 16x16 유지
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # 16 -> 16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            # 256px 입력 시 16x16x512 = 131072
            nn.Linear(16 * 16 * 512, z_dim),
        )

        # Z -> Feature Map (Decoding 시작)
        self.fc = nn.Linear(z_dim, 16 * 16 * 1024)

        # 2. SPADE Decoder (출력 256px 기준)
        # Start: 16x16
        self.head_0 = SPADEResnetBlock(1024, 1024, label_nc)

        self.G_middle_0 = SPADEResnetBlock(1024, 1024, label_nc)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024, label_nc)

        self.up_0 = SPADEResnetBlock(1024, 512, label_nc)  # -> 128 이후 256까지
        self.up_1 = SPADEResnetBlock(512, 256, label_nc)
        self.up_2 = SPADEResnetBlock(256, 128, label_nc)
        self.up_3 = SPADEResnetBlock(128, 64, label_nc)

        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, img, segmap):
        # Encoder
        z = self.encoder(img)

        # Decoder
        x = self.fc(z)
        x = x.view(-1, 1024, 16, 16)  # Start 16x16

        x = self.head_0(x, segmap)
        x = self.up(x)  # 32

        x = self.G_middle_0(x, segmap)
        x = self.up(x)  # 64

        x = self.G_middle_1(x, segmap)
        x = self.up(x)  # 128

        x = self.up_0(x, segmap)
        x = self.up(x)  # 256 (최종 해상도)

        # 256px에서 크기 유지하면서 채널만 줄이는 블록들
        x = self.up_1(x, segmap)  # 256x256
        x = self.up_2(x, segmap)  # 256x256
        x = self.up_3(x, segmap)  # 256x256

        # 최종 Image 생성
        x = F.leaky_relu(x, 2e-1, inplace=False)
        x = self.conv_img(x)
        x = torch.tanh(x)

        return x, z


# --- [3] Discriminator (Multi-Scale PatchGAN) ---
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3 + 4, num_D=2):
        """
        input_nc = 이미지(3) + 조건맵(4) = 7 채널이 기본.
        """
        super().__init__()
        self.input_nc = input_nc
        self.num_D = num_D
        self.n_layers = 4
        self.ndf = 64

        for i in range(num_D):
            netD = self.build_discriminator()
            self.add_module(f"discriminator_{i}", netD)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def build_discriminator(self):
        # inplace=False 로 전부 고정 (gradient 안전)
        model = [
            nn.Conv2d(
                self.input_nc, self.ndf, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        nf_mult = 1
        for n in range(1, self.n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(
                    self.ndf * nf_mult_prev,
                    self.ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(self.ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False),
            ]
        model += [
            nn.Conv2d(
                self.ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1
            )
        ]
        return nn.Sequential(*model)

    def forward(self, x):
        """
        x: (B, input_nc, H, W)
        """
        result = []
        cur_input = x
        for i in range(self.num_D):
            model = getattr(self, f"discriminator_{i}")
            out = model(cur_input)
            result.append(out)
            if i != (self.num_D - 1):
                cur_input = self.downsample(cur_input)
        return result
