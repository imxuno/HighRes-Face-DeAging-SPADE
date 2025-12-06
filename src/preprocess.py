##### 핵심 전처리 코드ㅡ
## 이미지 로드 → 얼굴 정렬 → 인종 필터링 → 파싱 → 3채널 조건지도(주름/모공/홍조) 생성

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis
from PIL import Image

# --- [설정] ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW_DATA_PATH = "./data/raw"
SAVE_PATH = "./data/processed"
WEIGHTS_PATH = "./weights"

# --- [1. 알고리즘: 미세 요소 추출 (논문 3.3)] ---

def get_wrinkle_map(img_gray, mask):
    """
    논문 3.3.1 주름 검출: Laplacian + Gabor Filter Bank
    Formula: Norm(0.6 * |Laplacian| + 0.4 * Gabor)
    """
    # 1. Laplacian (2차 미분)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)

    # 2. Gabor Filter Bank (0, 45, 90, 135도)
    gabor_response = np.zeros_like(img_gray, dtype=np.float64)
    ksize = 31 # 커널 크기
    for theta in [0, 45, 90, 135]:
        theta_rad = np.deg2rad(theta)
        # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi)
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_64F)
        filtered = cv2.filter2D(img_gray, cv2.CV_64F, kernel)
        gabor_response += np.abs(filtered)

    gabor_response /= 4.0 # 평균

    # 3. Fusion & Normalize
    # 가중치: 논문 실험값 (alpha=0.6)
    wrinkle_map = 0.6 * laplacian + 0.4 * gabor_response

    # ROI Masking & Normalization
    wrinkle_map = wrinkle_map * mask

    # Robust Normalization (Percentile 5~95%)
    vmin, vmax = np.percentile(wrinkle_map[mask > 0], [5, 95])
    wrinkle_map = np.clip((wrinkle_map - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return wrinkle_map.astype(np.float32)

def get_pore_map(img_gray, mask):
    """
    논문 3.3.2 모공 검출: DoG (Difference of Gaussians)
    Formula: Norm(G_sigma1 - G_sigma2)
    """
    #
    sigma_small = 0.9
    sigma_large = 2.2

    g1 = cv2.GaussianBlur(img_gray, (0, 0), sigma_small)
    g2 = cv2.GaussianBlur(img_gray, (0, 0), sigma_large)

    dog = g1 - g2

    # 모공은 어두운 점이므로 음수 값을 반전 (함몰 부위 탐지)
    # 하지만 DoG 특성상 엣지가 강조되므로 절대값 사용 혹은 양수 클리핑
    # 여기서는 텍스처 강도를 위해 절대값 사용
    pore_map = np.abs(dog) * mask

    # Normalization (Percentile 3~97%)
    if np.sum(mask) > 0:
        vmin, vmax = np.percentile(pore_map[mask > 0], [3, 97])
        pore_map = np.clip((pore_map - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return pore_map.astype(np.float32)

def get_redness_map(img_bgr, mask):
    """
    논문 3.3.3 홍조 검출: LAB Color Space + Guided Filter
    Target: a* channel (Green-Red axis)
    """
    # 1. BGR to LAB 변환
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # 2. Extract a* channel (Redness)
    # OpenCV LAB range: L(0-255), a(0-255), b(0-255).
    # a channel: 128 is neutral. >128 is Red, <128 is Green.
    a_float = a_channel.astype(np.float32)

    # 양의 편향(Redness)만 고려: 128 이상인 값만 추출
    redness_raw = np.maximum(0, a_float - 128.0)

    # 3. Guided Filter
    # 가이드 이미지로 L 채널(밝기) 사용 -> 엣지 보존 스무딩
    radius = 15
    eps = 0.001
    # cv2.ximgproc가 없다면 일반 GaussianBlur로 대체 가능하나, Guided Filter 권장
    try:
        from cv2.ximgproc import guidedFilter
        redness_map = guidedFilter(guide=l_channel, src=redness_raw, radius=radius, eps=eps)
    except ImportError:
        print("Warning: opencv-contrib-python not found. Using GaussianBlur instead.")
        redness_map = cv2.GaussianBlur(redness_raw, (0, 0), 3)

    redness_map = redness_map * mask

    # Normalization
    if np.sum(mask) > 0:
        vmin, vmax = np.percentile(redness_map[mask > 0], [5, 95])
        redness_map = np.clip((redness_map - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return redness_map.astype(np.float32)

# --- [2. 유틸리티: 모델 로드 및 파이프라인] ---

class FacePreprocessor:
    def __init__(self):
        # 1. InsightFace (Detection & Alignment)
        # provider: RTX 5080 사용 시 'CUDAExecutionProvider'
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. BiSeNet (Parsing)
        # 모델 정의가 복잡하므로 여기서는 간소화된 로딩 로직 사용 (실제론 utils/bisenet.py 필요)
        # 사용자는 사전 학습된 79999_iter.pth가 필요함
        self.parsing_net = self._load_bisenet()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _load_bisenet(self):
        # BiSeNet 모델 로드 (Placeholder)
        # 실제 구현시에는 github의 model.py를 import 해야 함.
        # 여기서는 로드되었다고 가정.
        print(f"Loading BiSeNet from {os.path.join(WEIGHTS_PATH, '79999_iter.pth')}...")
        # net = BiSeNet(n_classes=19)
        # net.load_state_dict(torch.load(path))
        # return net.to(DEVICE).eval()
        return None # (실제 코드 실행을 위해선 이 부분을 구현해야 함)

    def process_single_image(self, img_path):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: return

        # 1. Face Detection & Alignment
        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"No face detected: {filename}")
            return

        # 가장 큰 얼굴 선택
        face = sorted(faces, key=lambda x: x.bbox[2]*x.bbox[3])[-1]

        # [논문 3.2] 인종 분류 (FairFace Logic Placeholder)
        # 실제 FairFace 모델 추론 코드가 필요함. 여기서는 InsightFace gender로 예시 대체
        # if not self.is_east_asian(face_img): return

        # InsightFace Alignment (1024x1024로 리사이즈 필요할 수 있음)
        # 여기서는 crop 후 resize
        bbox = face.bbox.astype(int)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        size = int(max(w, h) * 1.5) # 여유 있게 크롭

        face_img = cv2.getRectSubPix(img, (size, size), center)
        face_img = cv2.resize(face_img, (1024, 1024)) # 논문 해상도

        # 2. Face Parsing (Get Skin Mask)
        # (실제 모델 추론 코드로 대체 필요)
        # mask = self.run_parsing(face_img)
        # 여기서는 테스트를 위해 임시로 중앙을 피부로 가정
        mask = np.zeros((1024, 1024), dtype=np.float32)
        cv2.circle(mask, (512, 512), 400, 1, -1)

        # 3. Feature Extraction [논문 3.3]
        img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        wrinkle = get_wrinkle_map(img_gray, mask)
        pore = get_pore_map(img_gray, mask)
        redness = get_redness_map(face_img, mask)

        # 4. Save Results
        base_name = os.path.splitext(filename)[0]
        os.makedirs(os.path.join(SAVE_PATH, 'images'), exist_ok=True)
        os.makedirs(os.path.join(SAVE_PATH, 'maps'), exist_ok=True)

        cv2.imwrite(os.path.join(SAVE_PATH, 'images', filename), face_img)

        # 조건지도는 NPY로 저장 (학습용) 및 시각화용 PNG 저장
        maps_stack = np.stack([redness, wrinkle, pore], axis=0) # [3, H, W]
        np.save(os.path.join(SAVE_PATH, 'maps', f"{base_name}_cond.npy"), maps_stack)

        # 시각화 저장 (R, G, B 채널에 매핑)
        vis_img = cv2.merge([
            (pore * 255).astype(np.uint8),    # B: Pore
            (wrinkle * 255).astype(np.uint8), # G: Wrinkle
            (redness * 255).astype(np.uint8)  # R: Redness
        ])
        cv2.imwrite(os.path.join(SAVE_PATH, 'maps', f"{base_name}_vis.png"), vis_img)
        print(f"Processed: {filename}")

if __name__ == "__main__":
    processor = FacePreprocessor()

    # data/raw 폴더의 모든 이미지 처리
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        print(f"Please put test images in {RAW_DATA_PATH}")
    else:
        files = os.listdir(RAW_DATA_PATH)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                processor.process_single_image(os.path.join(RAW_DATA_PATH, f))
