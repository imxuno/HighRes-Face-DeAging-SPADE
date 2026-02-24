import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import kornia.filters as kf
import kornia.color as kc
from torch.nn import functional as F


# ============================================================
# [1] Identity Loss (ResNet50 embedding, "ArcFace-like")
# ============================================================
class IdentityLoss(nn.Module):
    """
    ⚠️ 주의:
    - model_ir_se50.pth는 원래 IR-SE50(ArcFace) 계열 가중치일 가능성이 큽니다.
    - 여기서는 torchvision resnet50에 strict=False로 "맞는 것만" 로드하므로
      '진짜 ArcFace' 수준의 성능을 보장하진 않습니다.
    - 그래도 ID 보존 경향을 주는 regularizer로는 쓸 수 있습니다.

    개선:
    - ResNet 입력에 ImageNet mean/std 정규화를 적용 (매우 중요)
    - AMP에서도 안정적으로 돌아가게 float32 강제
    """

    def __init__(self, pretrained_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        print(f"Loading Identity backbone from {pretrained_path}")

        self.loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)

        # ImageNet normalize (ResNet50에 필요)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        try:
            self.net = self._load_backbone(pretrained_path, device)
        except Exception as e:
            print(f"[IdentityLoss] Error loading backbone: {e}")
            self.net = None

    def _load_backbone(self, path: str, device: torch.device | str):
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(2048, 512)

        try:
            state_dict = torch.load(path, map_location="cpu")
            missing, unexpected = net.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[IdentityLoss] missing keys: {len(missing)}")
            if unexpected:
                print(f"[IdentityLoss] unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"[IdentityLoss] Warning: failed to load weights strictly. ({e})")

        net = net.to(device).eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) in [-1,1] -> [0,1] -> ImageNet normalize
        """
        x = (x + 1.0) / 2.0
        x = x.float()
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, img_real: torch.Tensor, img_fake: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return img_real.new_tensor(0.0)

        real = self._preprocess(img_real)
        fake = self._preprocess(img_fake)

        real_resize = F.interpolate(real, size=(112, 112), mode="bilinear", align_corners=False)
        fake_resize = F.interpolate(fake, size=(112, 112), mode="bilinear", align_corners=False)

        with torch.no_grad():
            emb_real = self.net(real_resize)
        emb_fake = self.net(fake_resize)

        return 1.0 - self.loss_fn(emb_real, emb_fake).mean()


# ============================================================
# [2] Cycle Consistency Loss (Wrinkle / Pore / Redness)
# ============================================================
class CycleConsistencyLoss(nn.Module):
    """
    생성 이미지에서 [Redness, Wrinkle, Pore] 맵을 추출하여 target_maps와 L1로 맞춤.

    개선:
    - 정규화를 'mask 내부 기준 min/max'로 수행 (피부 밖 0값에 의해 min/max가 깨지는 문제 방지)
    - 내부 연산은 float32 강제 (AMP에서도 안정)
    """

    def __init__(self):
        super().__init__()
        kernels = self._create_gabor_kernels()  # (N,1,K,K) float32
        self.register_buffer("gabor_kernels", kernels)

    def _create_gabor_kernels(self) -> torch.Tensor:
        kernels = []
        sigmas = [2.0, 3.0]
        lambdas = [4.0, 8.0]
        thetas = [0, 45, 90, 135]
        ksize = 31

        for theta_deg in thetas:
            theta = np.deg2rad(theta_deg)
            for lam in lambdas:
                for sigma in sigmas:
                    k = cv2.getGaborKernel(
                        (ksize, ksize),
                        sigma,
                        theta,
                        lam,
                        0.5,
                        0,
                        ktype=cv2.CV_32F,
                    )
                    kernels.append(torch.from_numpy(k))

        return torch.stack(kernels).unsqueeze(1).float()

    def _masked_minmax(self, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        x: (B,C,H,W)
        mask: (B,1,H,W) 0/1 (float)
        -> mask 내부에서만 min/max를 구해 0~1 normalize
        """
        B, C, H, W = x.shape
        m = (mask > 0.5).float()

        # (B,C,H,W)로 broadcast
        m_bc = m.expand(B, C, H, W)

        # mask 밖은 min 계산에 +inf, max 계산에 -inf로 처리
        x_min_src = torch.where(m_bc > 0.5, x, torch.full_like(x, float("inf")))
        x_max_src = torch.where(m_bc > 0.5, x, torch.full_like(x, float("-inf")))

        x_flat_min = x_min_src.view(B, C, -1)
        x_flat_max = x_max_src.view(B, C, -1)

        min_val = x_flat_min.min(dim=2, keepdim=True).values.view(B, C, 1, 1)
        max_val = x_flat_max.max(dim=2, keepdim=True).values.view(B, C, 1, 1)

        # mask가 너무 작아서 inf/-inf가 남는 케이스 방지
        min_val = torch.where(torch.isfinite(min_val), min_val, torch.zeros_like(min_val))
        max_val = torch.where(torch.isfinite(max_val), max_val, torch.ones_like(max_val))

        denom = (max_val - min_val).clamp_min(eps)
        out = (x - min_val) / denom
        out = out.clamp(0.0, 1.0)

        # mask 밖은 0
        out = out * m_bc
        return out

    def get_wrinkle_torch(self, img_gray: torch.Tensor) -> torch.Tensor:
        lap = kf.laplacian(img_gray, kernel_size=3).abs()
        # gabor conv
        kernels = self.gabor_kernels.to(img_gray.device, dtype=img_gray.dtype)
        gabor_feat = F.conv2d(img_gray, kernels, padding=15)
        gabor_response = gabor_feat.abs().mean(dim=1, keepdim=True)
        return 0.6 * lap + 0.4 * gabor_response

    def get_pore_torch(self, img_gray: torch.Tensor) -> torch.Tensor:
        g1 = kf.gaussian_blur2d(img_gray, (5, 5), (1.0, 1.0))
        g2 = kf.gaussian_blur2d(img_gray, (9, 9), (2.5, 2.5))
        return (g1 - g2).abs()

    def get_redness_torch(self, img_rgb01: torch.Tensor) -> torch.Tensor:
        img_lab = kc.rgb_to_lab(img_rgb01)
        _, a, _ = torch.chunk(img_lab, 3, dim=1)
        redness = F.relu(a)
        redness = kf.gaussian_blur2d(redness, (15, 15), (3.0, 3.0))
        return redness

    def forward(
        self,
        img_gen: torch.Tensor,     # (B,3,H,W) [-1,1]
        target_maps: torch.Tensor, # (B,3,H,W) [0,1]
        mask: torch.Tensor,        # (B,1,H,W) 0/1
    ) -> torch.Tensor:
        # float32 강제
        img_gen = img_gen.float()
        target_maps = target_maps.float()
        mask = mask.float()

        # [-1,1] -> [0,1]
        img01 = (img_gen + 1.0) / 2.0

        img_gray = kc.rgb_to_grayscale(img01)

        raw_wrinkle = self.get_wrinkle_torch(img_gray)
        raw_pore = self.get_pore_torch(img_gray)
        raw_redness = self.get_redness_torch(img01)

        # mask 내부 기준 normalize
        pred_wrinkle = self._masked_minmax(raw_wrinkle, mask)
        pred_pore = self._masked_minmax(raw_pore, mask)
        pred_redness = self._masked_minmax(raw_redness, mask)

        pred_stack = torch.cat([pred_redness, pred_wrinkle, pred_pore], dim=1)

        # mask는 (B,1,H,W)라 채널 broadcast로 작동
        loss = F.l1_loss(pred_stack * mask, target_maps[:, :3] * mask)
        return loss


# ============================================================
# [3] VGG Perceptual Loss
# ============================================================
class VGGLoss(nn.Module):
    """
    VGG19 앞부분 feature 기반 perceptual loss.
    - 입력: [-1,1]
    - 내부에서 [0,1] 변환 + ImageNet mean/std 정규화
    - AMP에서도 float32로 강제하는 편이 안전
    """

    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
            self.register_buffer("mean", torch.tensor(weights.meta["mean"]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(weights.meta["std"]).view(1, 3, 1, 1))
        except Exception:
            vgg = models.vgg19(pretrained=True).features
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])

        for p in self.parameters():
            p.requires_grad = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        x = x.float()
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        y = self._preprocess(y)

        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)

        return F.l1_loss(h1_x, h1_y) + F.l1_loss(h2_x, h2_y)
