import torch
import torch.nn as nn
import torchvision.models as models
import kornia.filters as kf
import kornia.color as kc
from torch.nn import functional as F
import cv2
import numpy as np


# --- [1] Identity Loss (ArcFace-like embedding) ---
class IdentityLoss(nn.Module):
    """
    얼굴 임베딩 간 코사인 거리로 ID 보존을 강제하는 Loss.
    현재는 torchvision resnet50 기반의 512-dim 임베딩을 사용한다.
    (model_ir_se50.pth의 구조와 100% 일치하진 않을 수 있지만,
     strict=False 로 가능한 부분만 로드해서 사용.)
    """

    def __init__(self, pretrained_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        print(f"Loading ArcFace from {pretrained_path}")
        self.loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)

        try:
            self.net = self._load_arcface(pretrained_path, device)
        except Exception as e:
            print(f"Error loading ArcFace: {e}")
            self.net = None

    def _load_arcface(self, path: str, device: torch.device | str):
        # 기본 backbone: torchvision resnet50
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(2048, 512)

        try:
            # CPU로 로드 후 원하는 device로 이동 (환경 호환성↑)
            state_dict = torch.load(path, map_location="cpu")
            net.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load ArcFace weights strictly. ({e})")

        net = net.to(device).eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def forward(self, img_real: torch.Tensor, img_fake: torch.Tensor) -> torch.Tensor:
        """
        img_real, img_fake: (B,3,H,W), [-1,1] 범위 가정
        """
        if self.net is None:
            # 학습 그래프에 안전하게 올라가도록 같은 device/type의 상수 사용
            return img_real.new_tensor(0.0)

        # [-1,1] -> [0,1]
        real = (img_real + 1.0) / 2.0
        fake = (img_fake + 1.0) / 2.0

        # AMP 환경에서도 이 Loss는 float32로 고정해서 안정성 확보
        real = real.float()
        fake = fake.float()

        real_resize = F.interpolate(
            real, size=(112, 112), mode="bilinear", align_corners=False
        )
        fake_resize = F.interpolate(
            fake, size=(112, 112), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            emb_real = self.net(real_resize)
        emb_fake = self.net(fake_resize)

        # 1 - cos(·,·) => 0이면 동일, 2에 가까울수록 다른 얼굴
        return 1.0 - self.loss_fn(emb_real, emb_fake).mean()


# --- [2] Cycle Consistency Loss (Wrinkle / Pore / Redness) ---
class CycleConsistencyLoss(nn.Module):
    """
    생성된 이미지에서 주름 / 모공 / 홍조 맵을 추출해서,
    주어진 target condition map과 L1로 맞추는 손실.
    """

    def __init__(self):
        super().__init__()
        kernels = self._create_gabor_kernels()
        # 모델 이동 시 함께 device가 이동하도록 buffer로 등록
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
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize),
                        sigma,
                        theta,
                        lam,
                        0.5,
                        0,
                        ktype=cv2.CV_32F,
                    )
                    kernels.append(torch.from_numpy(kernel))

        # (N,1,K,K) 형태, float32
        return torch.stack(kernels).unsqueeze(1)

    def normalize_minmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        배치 내 각 이미지/채널별로 0~1 Min-Max 정규화
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        min_val = x_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        max_val = x_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)

        eps = 1e-6
        return (x - min_val) / (max_val - min_val + eps)

    def get_wrinkle_torch(self, img_gray: torch.Tensor) -> torch.Tensor:
        # 1. Laplacian
        laplacian = kf.laplacian(img_gray, kernel_size=3)
        laplacian = torch.abs(laplacian)

        # 2. Gabor
        gabor_feat = F.conv2d(img_gray, self.gabor_kernels, padding=15)
        gabor_response = torch.mean(torch.abs(gabor_feat), dim=1, keepdim=True)

        # 3. Fusion
        wrinkle_map = 0.6 * laplacian + 0.4 * gabor_response
        return wrinkle_map

    def get_pore_torch(self, img_gray: torch.Tensor) -> torch.Tensor:
        g1 = kf.gaussian_blur2d(img_gray, (5, 5), (1.0, 1.0))
        g2 = kf.gaussian_blur2d(img_gray, (9, 9), (2.5, 2.5))
        dog = torch.abs(g1 - g2)
        return dog

    def get_redness_torch(self, img_rgb: torch.Tensor) -> torch.Tensor:
        # Kornia RGB to LAB (L:0~100, a:-128~127, b:-128~127)
        img_lab = kc.rgb_to_lab(img_rgb)
        _, a, _ = torch.chunk(img_lab, 3, dim=1)

        redness = F.relu(a)  # 양의 a만 사용
        redness = kf.gaussian_blur2d(redness, (15, 15), (3.0, 3.0))
        return redness

    def forward(
        self,
        img_gen: torch.Tensor,        # (B,3,H,W), [-1,1]
        target_maps: torch.Tensor,    # (B,3,H,W), [0,1]
        mask: torch.Tensor,           # (B,1,H,W), 0/1
    ) -> torch.Tensor:
        """
        AMP/autocast 환경에서도 안정성을 높이기 위해
        이 Loss 내부는 float32로 강제해서 계산.
        """
        # [-1,1] -> [0,1]
        img_gen_norm = (img_gen + 1.0) / 2.0

        # 모두 float32로 캐스팅 (gabor, kornia 연산 안전성↑)
        img_gen_norm = img_gen_norm.float()
        target_maps = target_maps.float()
        mask = mask.float()

        # Grayscale for wrinkle/pore
        img_gray = kc.rgb_to_grayscale(img_gen_norm)

        # Raw feature extraction
        raw_wrinkle = self.get_wrinkle_torch(img_gray)
        raw_pore = self.get_pore_torch(img_gray)
        raw_redness = self.get_redness_torch(img_gen_norm)

        # 예측값을 0~1로 정규화해서 target range에 맞춤
        pred_wrinkle = self.normalize_minmax(raw_wrinkle)
        pred_pore = self.normalize_minmax(raw_pore)
        pred_redness = self.normalize_minmax(raw_redness)

        # 채널 순서: [Redness, Wrinkle, Pore]
        pred_stack = torch.cat([pred_redness, pred_wrinkle, pred_pore], dim=1)

        # L1 Loss with Mask
        loss = F.l1_loss(pred_stack * mask, target_maps[:, :3] * mask)
        return loss


# --- [3] VGG Perceptual Loss ---
class VGGLoss(nn.Module):
    """
    VGG19의 앞부분 feature를 사용한 perceptual loss.
    입력 이미지는 [-1,1] 범위를 가정하고, 내부에서 ImageNet 정규화까지 수행한다.
    """

    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
            self.register_buffer(
                "mean", torch.tensor(weights.meta["mean"]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor(weights.meta["std"]).view(1, 3, 1, 1)
            )
        except Exception:
            # 구버전 torchvision 호환
            vgg = models.vgg19(pretrained=True).features
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])

        # VGG는 고정
        for param in self.parameters():
            param.requires_grad = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        [-1,1] 범위의 이미지를 [0,1]로 변환 후 ImageNet mean/std 정규화.
        """
        x = (x + 1.0) / 2.0
        # AMP 상황에서도 VGG는 float32로 고정하는 편이 안전
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

        loss = F.l1_loss(h1_x, h1_y) + F.l1_loss(h2_x, h2_y)
        return loss
