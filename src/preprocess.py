import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Path / Import Safety
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                      # project root
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)  # src 경로를 sys.path에 추가 (utils 패키지 임포트 안정화)

try:
    # utils/feature_extract.py
    from feature_extract import get_wrinkle_map, get_pore_map, get_redness_map
except Exception as e:
    print(f"[FATAL] Could not import utils.feature_extract: {e}")
    raise

try:
    # utils/bisenet.py
    from utils.bisenet import BiSeNet
except Exception as e:
    print(f"[FATAL] Could not import utils.bisenet: {e}")
    raise


# -----------------------------
# Global Settings
# -----------------------------
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True if DEVICE == "cuda" else False


class FacePreprocessor:
    def __init__(
        self,
        weights_dir: str,
        img_size: int = 1024,
        bisenet_input_size: int = 512,
        padding: float = 1.5,
        min_mask_ratio: float = 0.05,
        det_size: int = 640,
    ):
        self.weights_dir = weights_dir
        self.img_size = int(img_size)
        self.bisenet_input_size = int(bisenet_input_size)
        self.padding = float(padding)
        self.min_mask_ratio = float(min_mask_ratio)
        self.det_size = int(det_size)

        print(f"[Init] DEVICE={DEVICE}, IMG_SIZE={self.img_size}, BISENET_IN={self.bisenet_input_size}")

        # 1) InsightFace
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if DEVICE == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if DEVICE == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))

        # 2) BiSeNet
        self.parsing_net = self._load_bisenet()

        # BiSeNet 입력 정규화 (ImageNet)
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _load_bisenet(self):
        weight_path = os.path.join(self.weights_dir, "79999_iter.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"[FATAL] BiSeNet weights not found: {weight_path}")

        net = BiSeNet(n_classes=19)

        state = torch.load(weight_path, map_location="cpu")
        # DataParallel 학습 가중치의 'module.' prefix 제거 대응
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state

        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warn] BiSeNet missing keys: {len(missing)}")
        if unexpected:
            print(f"[Warn] BiSeNet unexpected keys: {len(unexpected)}")

        net.to(DEVICE).eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    @torch.inference_mode()
    def run_parsing(self, bgr_img_1024: np.ndarray) -> np.ndarray:
        """
        입력: BGR 이미지(IMG_SIZE x IMG_SIZE)
        출력: skin_mask (H,W) float32, 0/1
        """
        # 1) Resize for parsing
        img_resized = cv2.resize(
            bgr_img_1024,
            (self.bisenet_input_size, self.bisenet_input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2) To tensor (RGB + ImageNet normalize)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = self.to_tensor(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

        # 3) Inference
        out = self.parsing_net(img_tensor)
        # 일부 구현은 (out, aux) 형태일 수 있어 첫 출력만 사용
        if isinstance(out, (list, tuple)):
            out = out[0]
        # out shape: (B,19,H,W)
        parsing_map = out.squeeze(0).argmax(0).detach().cpu().numpy().astype(np.uint8)

        # 4) mask definition: skin=1, nose=10 (눈/입술/머리카락 등 제외)
        skin_mask_small = np.where(
            (parsing_map == 1) | (parsing_map == 10), 1.0, 0.0
        ).astype(np.float32)

        # 5) Resize back to IMG_SIZE
        skin_mask = cv2.resize(
            skin_mask_small,
            (bgr_img_1024.shape[1], bgr_img_1024.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        return skin_mask

    def _select_largest_face(self, faces):
        # bbox = [x1, y1, x2, y2]
        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(np.float32)
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

        faces = sorted(faces, key=area)
        return faces[-1]

    def _safe_crop_square(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """
        bbox 기반으로 padding을 준 정사각 crop 후 IMG_SIZE로 resize.
        bbox: (4,) int/float
        """
        x1, y1, x2, y2 = bbox.astype(np.float32)
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(img.shape[1] - 1), x2)
        y2 = min(float(img.shape[0] - 1), y2)

        w = x2 - x1
        h = y2 - y1
        if w <= 2 or h <= 2:
            return None

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        size = int(max(w, h) * self.padding)
        size = max(size, 32)

        # getRectSubPix는 경계 밖을 0으로 패딩해줌
        face_img = cv2.getRectSubPix(img, (size, size), (cx, cy))
        if face_img is None or face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return face_img

    def process_single_image(
        self,
        img_path: str,
        out_images_dir: str,
        out_maps_dir: str,
        save_debug: bool = True,
        overwrite: bool = False,
    ) -> bool:
        """
        성공 시 True, 스킵/실패 시 False
        """
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        out_img_path = os.path.join(out_images_dir, filename)
        out_npy_path = os.path.join(out_maps_dir, f"{base_name}_cond.npy")
        out_vis_path = os.path.join(out_maps_dir, f"{base_name}_vis.png")

        if (not overwrite) and os.path.exists(out_npy_path) and os.path.exists(out_img_path):
            return False  # skip existing

        img = cv2.imread(img_path)
        if img is None:
            print(f"[Skip] Cannot read: {filename}")
            return False

        # 1) Face Detection
        faces = self.app.get(img)
        if not faces:
            print(f"[Skip] No face detected: {filename}")
            return False

        face = self._select_largest_face(faces)

        # 2) Crop & Resize
        face_img = self._safe_crop_square(img, face.bbox)
        if face_img is None:
            print(f"[Skip] Invalid crop: {filename}")
            return False

        # 3) Face Parsing -> mask
        mask = self.run_parsing(face_img)
        if float(np.sum(mask)) < (self.img_size * self.img_size * self.min_mask_ratio):
            print(f"[Skip] Mask too small ({np.sum(mask)}): {filename}")
            return False

        # 4) Feature Extraction
        img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        try:
            wrinkle = get_wrinkle_map(img_gray, mask)
            pore = get_pore_map(img_gray, mask)
            redness = get_redness_map(face_img, mask)
        except Exception as e:
            print(f"[Skip] Feature extract failed: {filename} -> {e}")
            return False

        # 5) Safety: dtype/shape/range
        def _fix_map(m):
            m = np.asarray(m, dtype=np.float32)
            if m.ndim != 2:
                raise ValueError(f"map must be (H,W), got {m.shape}")
            if m.shape[0] != self.img_size or m.shape[1] != self.img_size:
                m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            m = np.clip(m, 0.0, 1.0)
            return m

        try:
            wrinkle = _fix_map(wrinkle)
            pore = _fix_map(pore)
            redness = _fix_map(redness)
            mask2 = _fix_map(mask)  # float32 0~1
            # mask는 0/1로 강제(시각화/학습 안정)
            mask2 = (mask2 > 0.5).astype(np.float32)
        except Exception as e:
            print(f"[Skip] Map sanitize failed: {filename} -> {e}")
            return False

        # 6) Save
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_maps_dir, exist_ok=True)

        cv2.imwrite(out_img_path, face_img)

        # NPY: (4, H, W)  [redness, wrinkle, pore, mask]
        maps_stack = np.stack([redness, wrinkle, pore, mask2], axis=0).astype(np.float32)
        np.save(out_npy_path, maps_stack)

        if save_debug:
            # Debug vis: [mask | pore/wrinkle/redness]
            mask_viz = (mask2 * 255).astype(np.uint8)
            mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)

            vis_img = cv2.merge(
                [
                    (pore * 255).astype(np.uint8),
                    (wrinkle * 255).astype(np.uint8),
                    (redness * 255).astype(np.uint8),
                ]
            )
            debug_concat = np.hstack([mask_viz, vis_img])
            cv2.imwrite(out_vis_path, debug_concat)

        return True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "raw"),
        help="Raw images directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Output directory (will create images/ and maps/ inside)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "weights"),
        help="Weights directory (contains 79999_iter.pth)",
    )

    parser.add_argument("--img_size", type=int, default=1024, help="Final cropped face size")
    parser.add_argument("--bisenet_input_size", type=int, default=512, help="BiSeNet input size")
    parser.add_argument("--padding", type=float, default=1.5, help="Crop padding ratio")
    parser.add_argument("--min_mask_ratio", type=float, default=0.05, help="Min mask area ratio")
    parser.add_argument("--det_size", type=int, default=640, help="InsightFace det_size")

    parser.add_argument("--save_debug", action="store_true", help="Save debug vis images")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if outputs already exist")

    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    weights_dir = args.weights_dir

    out_images_dir = os.path.join(out_dir, "images")
    out_maps_dir = os.path.join(out_dir, "maps")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[Init] Created raw_dir: {raw_dir}")
        print("[Action] Put raw images there and run again.")
        return

    files = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()

    print(f"[Info] Found {len(files)} images in: {raw_dir}")
    print(f"[Info] Output to: {out_dir}")
    print(f"[Info] save_debug={args.save_debug}, skip_existing={args.skip_existing}")

    processor = FacePreprocessor(
        weights_dir=weights_dir,
        img_size=args.img_size,
        bisenet_input_size=args.bisenet_input_size,
        padding=args.padding,
        min_mask_ratio=args.min_mask_ratio,
        det_size=args.det_size,
    )

    ok = 0
    skipped = 0
    for f in tqdm(files, desc="Processing"):
        img_path = os.path.join(raw_dir, f)
        done = processor.process_single_image(
            img_path,
            out_images_dir=out_images_dir,
            out_maps_dir=out_maps_dir,
            save_debug=args.save_debug,
            overwrite=not args.skip_existing,
        )
        if done:
            ok += 1
        else:
            skipped += 1

    print(f"[Done] processed={ok}, skipped/failed={skipped}")
    print(f"[Out] images: {out_images_dir}")
    print(f"[Out] maps:   {out_maps_dir}")


if __name__ == "__main__":
    main()
