import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from networks import SPADEGenerator
from dataset import FaceDataset


def resolve_image_name(data_root: str, image_name: str) -> str:
    images_dir = os.path.join(data_root, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ not found in data_root: {images_dir}")

    p = os.path.join(images_dir, image_name)
    if os.path.exists(p):
        return image_name

    base = os.path.splitext(image_name)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        cand = base + ext
        if os.path.exists(os.path.join(images_dir, cand)):
            return cand

    for fn in os.listdir(images_dir):
        if os.path.splitext(fn)[0] == base:
            return fn

    raise FileNotFoundError(f"Could not find image '{image_name}' in {images_dir}")


def find_index_by_name(dataset: FaceDataset, image_name: str) -> int:
    base = os.path.splitext(image_name)[0]
    for i, fn in enumerate(dataset.image_names):
        if os.path.splitext(fn)[0] == base:
            return i
    raise ValueError(f"Image '{image_name}' not found in dataset list.")


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def denorm_to_01(t: torch.Tensor) -> torch.Tensor:
    return clamp01((t + 1.0) / 2.0)


def save_img01(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(t, path)


def make_alpha(mask01: torch.Tensor, dilate: int = 0, feather: int = 0) -> torch.Tensor:
    """
    mask01: (1,1,H,W) 0/1
    dilate: dilation radius (0이면 off)
    feather: avg blur kernel size (0이면 off)
    return alpha: (1,1,H,W) in [0,1]
    """
    alpha = mask01

    if dilate and dilate > 0:
        k = 2 * dilate + 1
        alpha = F.max_pool2d(alpha, kernel_size=k, stride=1, padding=dilate)

    if feather and feather > 0:
        # 홀수 커널 보장
        k = feather if (feather % 2 == 1) else (feather + 1)
        pad = k // 2
        alpha = F.avg_pool2d(alpha, kernel_size=k, stride=1, padding=pad)

    return clamp01(alpha)


def edit_cond_scale_to_zero(cond: torch.Tensor, mask: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """
    cond: (1,3,H,W) [0,1]
    mask: (1,1,H,W) 0/1
    factors: (1,3,1,1)
    """
    out = clamp01(cond * factors)
    # mask 밖은 원본 유지(대개 0이지만 안전)
    out = out * mask + cond * (1 - mask)
    return out


def edit_cond_contrast(cond: torch.Tensor, mask: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """
    평균(μ)은 유지하면서 대비만 줄이는 방식 (OOD 덜함)
    cond_edit = μ + factor*(cond-μ)
    """
    eps = 1e-6
    m = mask
    m_sum = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)  # (1,1,1,1)

    # 채널별 평균(μ): (1,3,1,1)
    mu = (cond * m).sum(dim=(2, 3), keepdim=True) / m_sum

    out = mu + factors * (cond - mu)
    out = clamp01(out)
    out = out * m + cond * (1 - m)
    return out


@torch.inference_mode()
def run_once(netG, real_img, spade_input, out_path: str,
             blend_skin: bool, alpha_3ch: torch.Tensor):
    fake_img, _ = netG(real_img, spade_input)

    fake01 = denorm_to_01(fake_img)
    real01 = denorm_to_01(real_img)

    if blend_skin:
        out01 = fake01 * alpha_3ch + real01 * (1 - alpha_3ch)
    else:
        out01 = fake01

    save_img01(out01, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to netG checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root (with images/ and maps/)")
    parser.add_argument("--image_name", type=str, required=True, help="Image file name inside images/")
    parser.add_argument("--img_size", type=int, default=256, help="Must match training (e.g., 256)")
    parser.add_argument("--out_dir", type=str, default="../results/deaging_demo", help="Output dir")

    parser.add_argument("--wrinkle_factor", type=float, default=1.0)
    parser.add_argument("--pore_factor", type=float, default=1.0)
    parser.add_argument("--redness_factor", type=float, default=1.0)

    parser.add_argument("--save_debug_maps", action="store_true")

    # ✅ new: 편집/합성 옵션
    parser.add_argument("--edit_method", type=str, default="contrast", choices=["contrast", "scale"],
                        help="contrast: 평균 유지하며 약화(추천), scale: 0쪽으로 스케일(기존)")
    parser.add_argument("--blend_skin", action="store_true",
                        help="If set, blend output only on skin mask (recommended)")
    parser.add_argument("--mask_dilate", type=int, default=6, help="Mask dilation radius (pixels)")
    parser.add_argument("--mask_feather", type=int, default=21, help="Mask feather kernel size")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_name = resolve_image_name(args.data_root, args.image_name)
    base = os.path.splitext(image_name)[0]

    ds = FaceDataset(args.data_root, image_size=args.img_size)
    idx = find_index_by_name(ds, image_name)
    sample = ds[idx]
    if sample is None:
        raise RuntimeError(f"Sample load failed for {image_name}")

    real_img = sample["image"].unsqueeze(0).to(device)       # (1,3,H,W) [-1,1]
    spade_in = sample["spade_input"].unsqueeze(0).to(device) # (1,4,H,W)

    cond = spade_in[:, :3].float()                           # (1,3,H,W) [0,1]
    mask = (spade_in[:, 3:4].float() > 0.5).float()          # (1,1,H,W) 0/1

    # alpha(합성용) 만들기
    alpha = make_alpha(mask, dilate=args.mask_dilate, feather=args.mask_feather)  # (1,1,H,W)
    alpha_3 = alpha.repeat(1, 3, 1, 1)  # (1,3,H,W)

    # factor 텐서
    factors = torch.tensor(
        [args.redness_factor, args.wrinkle_factor, args.pore_factor],
        device=device,
        dtype=cond.dtype,
    ).view(1, 3, 1, 1)

    # ✅ cond 편집
    if args.edit_method == "contrast":
        cond_edit = edit_cond_contrast(cond, mask, factors)
    else:
        cond_edit = edit_cond_scale_to_zero(cond, mask, factors)

    spade_edit = torch.cat([cond_edit, mask], dim=1)  # (1,4,H,W)

    # 모델 로드
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    netG.load_state_dict(state, strict=True)
    netG.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # 저장
    save_img01(denorm_to_01(real_img), os.path.join(args.out_dir, f"{base}_input.png"))

    # combo
    run_once(netG, real_img, spade_edit,
             os.path.join(args.out_dir, f"{base}_fake_combo.png"),
             blend_skin=args.blend_skin, alpha_3ch=alpha_3)

    # single edits
    # wrinkle only
    f_w = torch.tensor([1.0, args.wrinkle_factor, 1.0], device=device).view(1, 3, 1, 1)
    cond_w = edit_cond_contrast(cond, mask, f_w) if args.edit_method == "contrast" else edit_cond_scale_to_zero(cond, mask, f_w)
    run_once(netG, real_img, torch.cat([cond_w, mask], dim=1),
             os.path.join(args.out_dir, f"{base}_fake_wrinkle.png"),
             blend_skin=args.blend_skin, alpha_3ch=alpha_3)

    # pore only
    f_p = torch.tensor([1.0, 1.0, args.pore_factor], device=device).view(1, 3, 1, 1)
    cond_p = edit_cond_contrast(cond, mask, f_p) if args.edit_method == "contrast" else edit_cond_scale_to_zero(cond, mask, f_p)
    run_once(netG, real_img, torch.cat([cond_p, mask], dim=1),
             os.path.join(args.out_dir, f"{base}_fake_pore.png"),
             blend_skin=args.blend_skin, alpha_3ch=alpha_3)

    # redness only
    f_r = torch.tensor([args.redness_factor, 1.0, 1.0], device=device).view(1, 3, 1, 1)
    cond_r = edit_cond_contrast(cond, mask, f_r) if args.edit_method == "contrast" else edit_cond_scale_to_zero(cond, mask, f_r)
    run_once(netG, real_img, torch.cat([cond_r, mask], dim=1),
             os.path.join(args.out_dir, f"{base}_fake_redness.png"),
             blend_skin=args.blend_skin, alpha_3ch=alpha_3)

    if args.save_debug_maps:
        save_img01(cond.cpu(), os.path.join(args.out_dir, f"{base}_cond_orig.png"))
        save_img01(cond_edit.cpu(), os.path.join(args.out_dir, f"{base}_cond_edit.png"))
        save_img01(mask.repeat(1, 3, 1, 1).cpu(), os.path.join(args.out_dir, f"{base}_mask.png"))
        save_img01(alpha_3.cpu(), os.path.join(args.out_dir, f"{base}_alpha.png"))

    print("[OK] Saved to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
