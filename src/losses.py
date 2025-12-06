### 손실 함수 구현 (losses.py)
### 논문 3.5 손실 함수의 핵심인 Cycle Consistency Loss와 Identity Loss를 구현
### 특히 Cycle Loss를 위해서는 Phase 2에서 OpenCV로 짠 로직을 PyTorch/Kornia(미분 가능)로 재구현해야 함

import torch
import torch.nn as nn
import torchvision.models as models
import kornia.filters as kf
import kornia.color as kc
from torch.nn import functional as F

# --- [1] Identity Loss (ArcFace) ---
class IdentityLoss(nn.Module):
    def __init__(self, pretrained_path, device='cuda'):
        super().__init__()
        # ResNet-50 IR Backbone 로드 (외부 파일 필요)
        # 실제 사용시에는 'backbone.py' 같은 모델 정의 파일이 필요하나,
        # 여기서는 torchvision resnet50을 수정해서 로드한다고 가정하거나
        # 또는 InsightFace 라이브러리의 torch backend를 사용
        print(f"Loading ArcFace from {pretrained_path}")
        self.net = self._load_arcface(pretrained_path).to(device).eval()
        self.loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _load_arcface(self, path):
        # (구현 편의를 위해 ResNet50 구조만 가져옴, 실제론 model_ir_se50.py 필요)
        # 여기서는 단순히 ResNet18을 Placeholder로 사용하지만,
        # 실제론 다운받은 model_ir_se50 가중치를 로드해야 함.
        net = models.resnet50(pretrained=False)
        net.fc = nn.Linear(2048, 512) # ArcFace Output Dim
        # try: net.load_state_dict(torch.load(path))
        # except: print("Warning: ID Loss weights not loaded correctly.")
        return net

    def forward(self, img_real, img_fake):
        # ArcFace 입력 규격 (112x112)으로 리사이즈
        real_resize = F.interpolate(img_real, size=(112, 112), mode='bilinear')
        fake_resize = F.interpolate(img_fake, size=(112, 112), mode='bilinear')

        with torch.no_grad():
            emb_real = self.net(real_resize)
        emb_fake = self.net(fake_resize)

        # 1 - Cosine Similarity
        loss = 1.0 - self.loss_fn(emb_real, emb_fake)
        return loss.mean()

# --- [2] Cycle Consistency Loss (Differentiable Extraction) ---
class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 미분 가능한 특징 추출기 초기화 (Kornia 활용)

    def get_wrinkle_torch(self, img_gray):
        # Laplacian
        laplacian = kf.laplacian(img_gray, kernel_size=3)

        # Gabor (Kornia doesn't have a direct bank, simulate with conv2d)
        # 간소화를 위해 Sobel로 엣지 강도 근사 (실제론 Gabor Kernel 생성 필요)
        sobel = kf.sobel(img_gray)

        wrinkle_map = 0.6 * torch.abs(laplacian) + 0.4 * sobel
        return wrinkle_map

    def get_pore_torch(self, img_gray):
        # DoG implementation
        g1 = kf.gaussian_blur2d(img_gray, (3, 3), (0.9, 0.9))
        g2 = kf.gaussian_blur2d(img_gray, (5, 5), (2.2, 2.2))
        dog = torch.abs(g1 - g2)
        return dog

    def get_redness_torch(self, img_rgb):
        # RGB to LAB
        img_lab = kc.rgb_to_lab(img_rgb)
        l, a, b = torch.chunk(img_lab, 3, dim=1)

        # Redness (a channel > 0)
        redness = F.relu(a) # Positive bias
        # Guided Filter 대용으로 Median Blur 사용 (미분 가능 근사)
        redness = kf.median_blur(redness, (3, 3))
        return redness

    def forward(self, img_gen, target_maps, mask):
        # img_gen: (B, 3, H, W) [-1, 1] -> [0, 1]로 변환 필요할 수 있음
        img_gen_norm = (img_gen + 1) / 2.0
        img_gray = kf.rgb_to_grayscale(img_gen_norm)

        # Extract features from Generated Image
        pred_wrinkle = self.get_wrinkle_torch(img_gray)
        pred_pore = self.get_pore_torch(img_gray)
        pred_redness = self.get_redness_torch(img_gen_norm)

        pred_stack = torch.cat([pred_redness, pred_wrinkle, pred_pore], dim=1)

        # Masking
        # target_maps[:, :3]은 (Redness, Wrinkle, Pore)
        loss = F.l1_loss(pred_stack * mask, target_maps[:, :3] * mask)
        return loss

# --- [3] VGG Loss (Perceptual) ---
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # VGG의 중간 레이어 추출
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False # 고정

    def forward(self, x, y):
        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)

        loss = F.l1_loss(h1_x, h1_y) + F.l1_loss(h2_x, h2_y)
        return loss
