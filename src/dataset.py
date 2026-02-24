import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# [🔥필수] Windows 환경에서 DataLoader 멈춤(Deadlock) 방지 설정
# OpenCV가 내부적으로 멀티스레딩을 사용하지 않도록 제한하여 PyTorch와 충돌을 막습니다.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class FaceDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=512):
        """
        root_dir/
          ├─ images/
          │    └─ xxx.png / xxx.jpg ...
          └─ maps/
               └─ xxx_cond.npy  # (4, H, W) = [Redness, Wrinkle, Pore, Mask]
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.map_dir = os.path.join(root_dir, "maps")
        self.image_size = image_size
        self.is_train = is_train

        # --- 디렉터리 존재 여부 확인 ---
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"[FaceDataset] Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.map_dir):
            print(f"[FaceDataset Warning] Map dir not found: {self.map_dir}")
            print("  → 모든 샘플에 대해 dummy mask(=1.0)가 사용됩니다.")

        # --- 이미지 파일 목록 로드 (정렬해서 재현성 확보) ---
        self.image_names = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if len(self.image_names) == 0:
            raise RuntimeError(f"[FaceDataset] No images found in: {self.img_dir}")

        # 정규화: [-1, 1] 범위로 변환
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        base_name = os.path.splitext(img_name)[0]

        # ------------------------------------------------------------------
        # 1. 이미지 로드 (Load Image)
        # ------------------------------------------------------------------
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # 에러 발생 시 None 반환 -> train.py에서 safe_collate가 건너뜀
            print(f"[Image Error] {img_name}: {e}")
            return None

        # ------------------------------------------------------------------
        # 2. 맵 로드 (Load Condition Maps) - Cond 3ch + Mask 1ch = 4ch
        # ------------------------------------------------------------------
        map_path = os.path.join(self.map_dir, f"{base_name}_cond.npy")

        if not os.path.exists(map_path):
            # 파일이 없으면 더미 데이터 생성 (디버깅용, 학습엔 권장하지 않음)
            full_map = np.zeros(
                (4, img.shape[0], img.shape[1]), dtype=np.float32
            )
            full_map[3, :, :] = 1.0  # 마스크 채널은 1로 설정
        else:
            try:
                full_map = np.load(map_path, allow_pickle=False)
                full_map = np.asarray(full_map, dtype=np.float32)

                # 기대 형태: (C, H, W)
                if full_map.ndim != 3:
                    print(
                        f"[Map Warning] {base_name}: expected 3D map (C,H,W), "
                        f"got shape {full_map.shape}. Skipping this sample."
                    )
                    return None

            except Exception as e:
                print(f"[Map Load Error] {base_name}: {e}")
                return None

        # ------------------------------------------------------------------
        # 3. 리사이즈 (Resize)
        # ------------------------------------------------------------------
        try:
            # 이미지 리사이즈
            if (
                self.image_size != img.shape[0]
                or self.image_size != img.shape[1]
            ):
                img = cv2.resize(
                    img,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            # 맵 리사이즈: 채널별로 직접 리사이즈해서 transpose 문제 회피
            full_map = full_map.astype(np.float32, copy=False)
            C, Hm, Wm = full_map.shape  # C >= 4 기대

            if Hm != self.image_size or Wm != self.image_size:
                resized_map = np.zeros(
                    (C, self.image_size, self.image_size),
                    dtype=np.float32,
                )
                for c in range(C):
                    resized_map[c] = cv2.resize(
                        full_map[c],
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                full_map = resized_map

        except Exception as e:
            print(f"[Resize Error] {base_name}: {e}")
            return None

        # ------------------------------------------------------------------
        # 4. 채널 분리 (Split Condition Maps and Mask)
        # ------------------------------------------------------------------
        # 전처리 단계에서 [Redness, Wrinkle, Pore, Mask] 순서로 저장됨이라고 가정
        if full_map.shape[0] < 4:
            print(
                f"[Map Channel Warning] {base_name}: "
                f"expected >=4 channels, got {full_map.shape[0]}. Skipping."
            )
            return None
        elif full_map.shape[0] > 4:
            # 필요 이상으로 채널이 많을 경우 앞의 4개만 사용
            print(
                f"[Map Channel Info] {base_name}: "
                f"{full_map.shape[0]} channels found, using first 4."
            )
            full_map = full_map[:4, :, :]

        cond_map = full_map[:3, :, :]  # (3, H, W) -> SPADE 입력용 조건 지도
        mask = full_map[3:4, :, :]     # (1, H, W) -> Loss 마스킹용 (채널 차원 유지)

        # ------------------------------------------------------------------
        # 5. SPADE 입력 데이터 생성 (Condition + Mask)
        # ------------------------------------------------------------------
        spade_input = np.concatenate([cond_map, mask], axis=0)  # (4, H, W)

        # ------------------------------------------------------------------
        # 6. 텐서 변환 (To Tensor)
        # ------------------------------------------------------------------
        img_tensor = self.transform(img)                      # (3, H, W), [-1,1]
        spade_tensor = torch.from_numpy(spade_input).float()
        target_maps_tensor = torch.from_numpy(cond_map).float()
        mask_tensor = torch.from_numpy(mask).float()

        return {
            "image": img_tensor,           # Real Image ([-1, 1])
            "spade_input": spade_tensor,   # Generator 입력 (Red, Wrinkle, Pore, Mask)
            "target_maps": target_maps_tensor,  # Cycle Loss 정답지 (Red, Wrinkle, Pore)
            "mask": mask_tensor,           # Loss 마스크 (Mask)
            "name": base_name,
        }
