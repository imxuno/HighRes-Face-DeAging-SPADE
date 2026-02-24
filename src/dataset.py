import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Windows + OpenCV 멀티스레딩 충돌 방지
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class FaceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        image_size: int = 256,
        allow_dummy_maps: bool = False,
        flip_prob: float = 0.5,
    ):
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

        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.allow_dummy_maps = bool(allow_dummy_maps)
        self.flip_prob = float(flip_prob)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"[FaceDataset] Image dir not found: {self.img_dir}")

        if not os.path.isdir(self.map_dir):
            if self.allow_dummy_maps:
                print(f"[FaceDataset Warning] Map dir not found: {self.map_dir}")
                print("  -> allow_dummy_maps=True 이므로 dummy cond/mask를 사용합니다 (학습엔 비권장).")
            else:
                raise FileNotFoundError(
                    f"[FaceDataset] Map dir not found: {self.map_dir}\n"
                    f"  maps/가 없으면 학습 품질이 심각하게 망가집니다. (allow_dummy_maps로 강제 가능)"
                )

        self.image_names = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if len(self.image_names) == 0:
            raise RuntimeError(f"[FaceDataset] No images found in: {self.img_dir}")

        # [-1,1] normalize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        base_name = os.path.splitext(img_name)[0]

        # -------------------------
        # 1) Load image
        # -------------------------
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Image Error] {img_name}: {e}")
            return None

        # -------------------------
        # 2) Load map (4,H,W)
        # -------------------------
        map_path = os.path.join(self.map_dir, f"{base_name}_cond.npy")

        if (not os.path.exists(map_path)) and self.allow_dummy_maps:
            full_map = np.zeros((4, img.shape[0], img.shape[1]), dtype=np.float32)
            full_map[3, :, :] = 1.0
        else:
            if not os.path.exists(map_path):
                print(f"[Map Missing] {base_name}: {map_path}")
                return None
            try:
                full_map = np.load(map_path, allow_pickle=False)
                full_map = np.asarray(full_map, dtype=np.float32)
                if full_map.ndim != 3:
                    print(f"[Map Warning] {base_name}: expected (C,H,W), got {full_map.shape}")
                    return None
            except Exception as e:
                print(f"[Map Load Error] {base_name}: {e}")
                return None

        # 채널 개수 정리
        if full_map.shape[0] < 4:
            print(f"[Map Channel Warning] {base_name}: expected >=4, got {full_map.shape[0]}")
            return None
        if full_map.shape[0] > 4:
            # 앞 4개만 사용
            full_map = full_map[:4, :, :]

        # -------------------------
        # 3) Resize
        # -------------------------
        try:
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

            C, Hm, Wm = full_map.shape
            if Hm != self.image_size or Wm != self.image_size:
                resized_map = np.zeros((C, self.image_size, self.image_size), dtype=np.float32)
                for c in range(C):
                    # ✅ cond(0~2): linear / mask(3): nearest
                    interp = cv2.INTER_NEAREST if c == 3 else cv2.INTER_LINEAR
                    resized_map[c] = cv2.resize(full_map[c], (self.image_size, self.image_size), interpolation=interp)
                full_map = resized_map
        except Exception as e:
            print(f"[Resize Error] {base_name}: {e}")
            return None

        # -------------------------
        # 4) Split + sanitize
        # -------------------------
        cond_map = np.clip(full_map[:3], 0.0, 1.0).astype(np.float32)    # (3,H,W)
        mask = np.clip(full_map[3:4], 0.0, 1.0).astype(np.float32)       # (1,H,W)
        mask = (mask > 0.5).astype(np.float32)                           # ✅ binary 강제

        # -------------------------
        # 5) Train aug (flip) - 이미지+맵 동시 반전
        # -------------------------
        if self.is_train and (np.random.rand() < self.flip_prob):
            img = np.ascontiguousarray(img[:, ::-1, :])
            cond_map = np.ascontiguousarray(cond_map[:, :, ::-1])
            mask = np.ascontiguousarray(mask[:, :, ::-1])

        # -------------------------
        # 6) Make tensors
        # -------------------------
        spade_input = np.concatenate([cond_map, mask], axis=0)  # (4,H,W)

        img_tensor = self.transform(img)                        # (3,H,W) [-1,1]
        spade_tensor = torch.from_numpy(spade_input).float()    # (4,H,W) [0,1] + mask
        target_maps_tensor = torch.from_numpy(cond_map).float() # (3,H,W)
        mask_tensor = torch.from_numpy(mask).float()            # (1,H,W)

        return {
            "image": img_tensor,
            "spade_input": spade_tensor,
            "target_maps": target_maps_tensor,
            "mask": mask_tensor,
            "name": base_name,
        }
