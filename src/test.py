##### 시각화 및 제어 결과 생성 (test.py)
### 학습된 체크포인트(.pth)를 로드하고, 테스트 이미지를 입력받아 원본 복원(Reconstruction)과 속성별 제어(Editing) 결과를 생성
### 기능:
### mode='reconstruction': 입력 그대로 복원 (화질 확인용).
### mode='edit': 주름/모공/홍조 채널 값을 0으로 만들어 De-aging 결과 생성.

import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from networks import SPADEGenerator

def load_model(checkpoint_path, device):
    # label_nc=4 (3 Maps + 1 Mask) -> 학습 때 설정과 맞춰야 함
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint)
    netG.eval()
    return netG

def process_image(img_path, map_path, img_size=512):
    # 이미지 로드 및 전처리
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # 정규화 (-1 ~ 1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0) # (1, 3, H, W)

    # 조건지도 로드
    if os.path.exists(map_path):
        cond_map = np.load(map_path) # (3, H, W)
        # 리사이즈
        cond_map = cond_map.transpose(1, 2, 0)
        cond_map = cv2.resize(cond_map, (img_size, img_size))
        cond_map = cond_map.transpose(2, 0, 1)
    else:
        # 더미 (테스트용)
        cond_map = np.zeros((3, img_size, img_size), dtype=np.float32)

    return img_tensor, cond_map

def run_inference(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.output_dir, exist_ok=True)

    # 1. 모델 로드
    netG = load_model(opt.checkpoint, device)
    print(f"Model loaded from {opt.checkpoint}")

    # 2. 파일 리스트
    files = [f for f in os.listdir(opt.img_dir) if f.endswith(('.png', '.jpg'))]

    for filename in files:
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(opt.img_dir, filename)
        map_path = os.path.join(opt.map_dir, f"{base_name}_cond.npy")

        # 데이터 준비
        real_img, cond_numpy = process_image(img_path, map_path, opt.img_size)
        real_img = real_img.to(device)

        # Skin Mask (Placeholder)
        mask = np.ones((1, opt.img_size, opt.img_size), dtype=np.float32)

        # --- [시나리오 1: Reconstruction (원본 복원)] ---
        cond_recon = np.concatenate([cond_numpy, mask], axis=0)
        cond_tensor = torch.from_numpy(cond_recon).float().unsqueeze(0).to(device)

        with torch.no_grad():
            fake_recon, _ = netG(real_img, cond_tensor)

        save_result(fake_recon, os.path.join(opt.output_dir, f"{base_name}_recon.png"))

        # --- [시나리오 2: Editing (각 요소 제거)] ---
        # 0: Redness, 1: Wrinkle, 2: Pore
        edit_modes = {'no_redness': 0, 'no_wrinkle': 1, 'no_pore': 2}

        for mode_name, ch_idx in edit_modes.items():
            cond_edit = cond_numpy.copy()
            cond_edit[ch_idx, :, :] = 0.0 # 해당 채널 제거 (De-aging)

            # SPADE Input 구성
            spade_in = np.concatenate([cond_edit, mask], axis=0)
            spade_tensor = torch.from_numpy(spade_in).float().unsqueeze(0).to(device)

            with torch.no_grad():
                fake_edit, _ = netG(real_img, spade_tensor)

            save_result(fake_edit, os.path.join(opt.output_dir, f"{base_name}_{mode_name}.png"))

        print(f"Generated results for {filename}")

def save_result(tensor, path):
    # Tensor (-1~1) -> Numpy (0~255) -> Save
    img = tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--img_dir', type=str, default='./data/processed/images', help='Test images')
    parser.add_argument('--map_dir', type=str, default='./data/processed/maps', help='Test maps')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output folder')
    parser.add_argument('--img_size', type=int, default=512)

    opt = parser.parse_args()
    run_inference(opt)
