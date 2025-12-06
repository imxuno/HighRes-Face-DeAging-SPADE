##### dataset.py
### 데이터 로더
### preprocess.py를 통해 저장된 images/와 maps/ 폴더에서 데이터를 짝지어 불러옴
### 핵심: 모델 입력인 input_sem은 [3채널 텍스처(R/G/B) + N채널 마스크]=가 합쳐진 형태여야 함
### 전처리된 데이터(이미지, 조건지도, 마스크)를 불러와 모델에 맞는 텐서로 변환

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=512):
        # 1024x1024는 메모리 부담이 크므로 학습 시 512 Resize 옵션 제공
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.map_dir = os.path.join(root_dir, 'maps')
        self.image_size = image_size

        # 파일 리스트 로드
        self.image_names = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg'))]
        self.is_train = is_train

        # Transform (이미지용: -1 ~ 1 정규화)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        base_name = os.path.splitext(img_name)[0]

        # 1. Load Image
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Load Maps (Condition + Mask)
        # preprocess.py에서 저장한 .npy 파일 로드
        # 저장 포맷: [3, H, W] (Redness, Wrinkle, Pore)
        map_path = os.path.join(self.map_dir, f"{base_name}_cond.npy")

        if not os.path.exists(map_path):
            # 파일이 없으면 더미 데이터 생성 (디버깅용)
            cond_map = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            cond_map = np.load(map_path)

        # 3. Resize (if needed)
        if self.image_size != img.shape[0]:
            img = cv2.resize(img, (self.image_size, self.image_size))
            # 맵 리사이즈 (CH, H, W -> H, W, CH로 바꿔서 resize 후 다시 복구)
            cond_map = cond_map.transpose(1, 2, 0)
            cond_map = cv2.resize(cond_map, (self.image_size, self.image_size))
            cond_map = cond_map.transpose(2, 0, 1)

        # 4. Generate Semantic Mask (Placeholder)
        # 실제론 BiSeNet 결과가 있어야 하지만, 현재는 preprocess.py에서
        # 마스크를 따로 저장하지 않았으므로, 여기서 임의로 Skin Mask 생성하거나
        # preprocess.py를 수정해서 마스크도 .npy에 포함해야 함.
        # 여기서는 편의상 전체를 Skin(1)으로 가정하는 더미 마스크 생성
        mask = np.ones((1, self.image_size, self.image_size), dtype=np.float32)

        # 5. Concatenate (Condition Maps + Masks) -> SPADE Input
        # Input: [Redness, Wrinkle, Pore, Skin_Mask] = 4 Channel
        spade_input = np.concatenate([cond_map, mask], axis=0)

        # 6. To Tensor
        img_tensor = self.transform(img)
        spade_tensor = torch.from_numpy(spade_input).float()

        # Cycle Loss용 Target Maps (R,G,B)
        target_maps = torch.from_numpy(cond_map).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {
            'image': img_tensor,        # Real Image (-1~1)
            'spade_input': spade_tensor, # SPADE Condition (Map + Mask)
            'target_maps': target_maps,  # GT Maps for Cycle Loss
            'mask': mask_tensor,         # ROI Mask
            'name': base_name
        }
