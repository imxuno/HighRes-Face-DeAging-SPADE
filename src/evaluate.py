### 정량적 평가 (Attr-MAE) 계산 (evaluate.py)
### 논문의 Table 2를 채우기 위한 코드
### 생성된 이미지에서 다시 특징(주름/모공/홍조)을 추출하고, 우리가 입력으로 넣었던 목표값(Target)과의 차이(L1 Loss)를 계산하여 평균을 냄
### 핵심 로직: Phase 3의 CycleConsistencyLoss 클래스에 있는 Feature Extractor를 재사용


import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FaceDataset
from networks import SPADEGenerator
from losses import CycleConsistencyLoss # 특징 추출기 재사용

def evaluate_mae(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    netG.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    netG.eval()

    # 2. 특징 추출기 (CycleLoss 모듈 활용)
    extractor = CycleConsistencyLoss().to(device)

    # 3. 데이터셋
    dataset = FaceDataset(opt.data_root, is_train=False, image_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    total_mae = torch.zeros(3).to(device) # [Redness, Wrinkle, Pore]
    count = 0

    print("Starting Evaluation (Attr-MAE)...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            real_img = data['image'].to(device)
            # 원본 조건지도 (이것을 Target으로 가정하고 복원 실험)
            target_maps = data['target_maps'].to(device)
            mask = data['mask'].to(device)
            spade_input = data['spade_input'].to(device)

            # 생성
            fake_img, _ = netG(real_img, spade_input)

            # 생성된 이미지에서 맵 재추출 ([-1,1] -> [0,1] normalization inside extractor logic check needed)
            # losses.py의 forward 로직과 유사하게 구현
            img_gen_norm = (fake_img + 1) / 2.0 # 0~1 range

            # Kornia 기반 추출 (losses.py에 정의된 함수 사용)
            # 주의: losses.py의 get_..._torch 함수들을 public으로 쓰거나 여기서 호출
            # 편의상 extractor.forward 내부 로직을 활용하기 위해 아래와 같이 계산
            # 하지만 정확한 MAE를 위해선 개별 채널 차이를 봐야 함.

            import kornia.filters as kf
            import kornia.color as kc

            img_gray = kf.rgb_to_grayscale(img_gen_norm)

            # 예측값 추출
            pred_wrinkle = extractor.get_wrinkle_torch(img_gray)
            pred_pore = extractor.get_pore_torch(img_gray)
            pred_redness = extractor.get_redness_torch(img_gen_norm)

            pred_stack = torch.cat([pred_redness, pred_wrinkle, pred_pore], dim=1)

            # MAE 계산 (Pixel-wise L1 distance masked by skin region)
            # 채널별 평균 계산
            diff = torch.abs(pred_stack - target_maps[:, :3]) * mask

            # 마스크 영역 내의 평균 오차 계산
            valid_pixels = mask.sum() + 1e-6
            mae_per_channel = diff.sum(dim=(2, 3)) / valid_pixels # (B, 3)

            total_mae += mae_per_channel.sum(dim=0)
            count += real_img.size(0)

    avg_mae = total_mae / count
    print("="*30)
    print(f"Evaluation Result (Table 2)")
    print(f"Attr-MAE (Redness): {avg_mae[0]:.4f}")
    print(f"Attr-MAE (Wrinkle): {avg_mae[1]:.4f}")
    print(f"Attr-MAE (Pore)   : {avg_mae[2]:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data/processed')
    parser.add_argument('--img_size', type=int, default=512)
    opt = parser.parse_args()

    evaluate_mae(opt)
