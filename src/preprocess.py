import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis
from PIL import Image

# Utils 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'utils'))

# Feature Extract 함수 임포트
try:
    from feature_extract import get_wrinkle_map, get_pore_map, get_redness_map
except ImportError as e:
    print(f"Error: Could not import feature_extract.py. {e}")
    exit(1)

from utils.bisenet import BiSeNet

# --- [설정] ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(current_dir)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights")
IMG_SIZE = 1024 
BISENET_INPUT_SIZE = 512

class FacePreprocessor:
    def __init__(self):
        print(f"Initializing FacePreprocessor on {DEVICE}...")
        
        # 1. InsightFace
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. BiSeNet 로드
        self.parsing_net = self._load_bisenet()
        
        # [수정됨] BiSeNet용 정규화 (ImageNet 표준 사용 - 이전 코드와 동일하게 맞춤)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _load_bisenet(self):
        weight_path = os.path.join(WEIGHTS_PATH, '79999_iter.pth')
        if not os.path.exists(weight_path):
            print(f"Error: BiSeNet weights not found at {weight_path}")
            exit(1)
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        net.to(DEVICE).eval()
        return net

    def run_parsing(self, img):
        # 1. Resize
        img_resized = cv2.resize(img, (BISENET_INPUT_SIZE, BISENET_INPUT_SIZE))
        
        # 2. To Tensor (ImageNet Normalize 적용됨)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = self.to_tensor(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        
        # 3. Inference
        with torch.no_grad():
            out = self.parsing_net(img_tensor)[0]
        
        # 4. Argmax
        parsing_map = out.squeeze(0).cpu().numpy().argmax(0)
        
        # 5. Mask Definition (Skin=1, Nose=10)
        # 79999_iter.pth 기준: 
        # 0:bg, 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye, 6:eye_g, 7:l_ear, 8:r_ear, 
        # 9:ear_r, 10:nose, 11:mouth, 12:u_lip, 13:l_lip, 14:neck, 15:neck_l, 16:cloth, 17:hair, 18:hat
        
        # 확실한 피부 영역(1)과 코(10)만 선택. (입술, 눈 등 배제)
        skin_mask_small = np.where((parsing_map == 1) | (parsing_map == 10), 1, 0).astype(np.float32)
        
        # 6. Resize back to 1024
        skin_mask = cv2.resize(skin_mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # [디버깅용] 파싱 결과 컬러맵 저장 (이게 이상하면 정규화 문제임)
        # debug_viz = (parsing_map * (255/19)).astype(np.uint8)
        # debug_viz = cv2.applyColorMap(debug_viz, cv2.COLORMAP_JET)
        # return skin_mask, debug_viz # 디버깅 필요시 주석 해제
        
        return skin_mask

    def process_single_image(self, img_path):
        filename = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path)
            if img is None: return
        except: return

        # 1. Face Detection
        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"Skipping {filename}: No face detected.")
            return

        face = sorted(faces, key=lambda x: x.bbox[2]*x.bbox[3])[-1]

        # Crop & Resize
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = int(max(w, h) * 1.5)
        
        face_img = cv2.getRectSubPix(img, (size, size), (center_x, center_y))
        if face_img is None or face_img.size == 0: return
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

        # 2. Face Parsing
        mask = self.run_parsing(face_img)
        
        if np.sum(mask) < (IMG_SIZE * IMG_SIZE * 0.05):
            print(f"Skipping {filename}: Invalid mask size.")
            return

        # 3. Feature Extraction
        img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        wrinkle = get_wrinkle_map(img_gray, mask)
        pore = get_pore_map(img_gray, mask)
        redness = get_redness_map(face_img, mask)

        # 4. Save
        base_name = os.path.splitext(filename)[0]
        os.makedirs(os.path.join(SAVE_PATH, 'images'), exist_ok=True)
        os.makedirs(os.path.join(SAVE_PATH, 'maps'), exist_ok=True)

        cv2.imwrite(os.path.join(SAVE_PATH, 'images', filename), face_img)

        # Save NPY (4 channels)
        maps_stack = np.stack([redness, wrinkle, pore, mask], axis=0)
        np.save(os.path.join(SAVE_PATH, 'maps', f"{base_name}_cond.npy"), maps_stack)

        # [수정됨] 디버깅을 위한 마스크 시각화 강화
        # 마스크 영역(흰색) + 원본 이미지 오버레이
        mask_viz = (mask * 255).astype(np.uint8)
        mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
        
        # 텍스처 맵 시각화
        vis_img = cv2.merge([
            (pore * 255).astype(np.uint8),
            (wrinkle * 255).astype(np.uint8),
            (redness * 255).astype(np.uint8)
        ])
        
        # 마스크와 맵을 나란히 저장 (왼쪽: 마스크 확인용, 오른쪽: 텍스처 확인용)
        debug_concat = np.hstack([mask_viz, vis_img])
        
        cv2.imwrite(os.path.join(SAVE_PATH, 'maps', f"{base_name}_vis.png"), debug_concat)
        
        print(f"Processed: {filename}")

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        print(f"Created {RAW_DATA_PATH}. Please put raw images here.")
        exit()
        
    processor = FacePreprocessor()
    files = [f for f in os.listdir(RAW_DATA_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images.")
    
    for f in files:
        processor.process_single_image(os.path.join(RAW_DATA_PATH, f))