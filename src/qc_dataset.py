import os
import argparse
import shutil
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def load_cond_npy(path: str):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32)

    # 허용 형태: (4,H,W) 또는 (H,W,4)
    if arr.ndim != 3:
        raise ValueError(f"npy must be 3D, got {arr.shape}")

    if arr.shape[0] == 4:
        # (4,H,W)
        return arr
    if arr.shape[-1] == 4:
        # (H,W,4) -> (4,H,W)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)

    raise ValueError(f"unexpected channel shape {arr.shape}")


def binarize_mask(mask: np.ndarray, thr: float = 0.5):
    mask = np.asarray(mask, dtype=np.float32)
    mask = (mask > thr).astype(np.float32)
    return mask


def mask_boundary(mask01: np.ndarray):
    m = (mask01 * 255).astype(np.uint8)
    edges = cv2.Canny(m, 50, 150)
    return edges


def overlay_edges(bgr: np.ndarray, edges: np.ndarray):
    out = bgr.copy()
    # edges==255인 위치를 빨간색으로 표시(BGR)
    out[edges > 0] = (0, 0, 255)
    return out


def energy_outside_mask(map01: np.ndarray, mask01: np.ndarray, eps: float = 1e-8):
    # map01: [0,1], mask01: 0/1
    total = float(np.sum(map01) + eps)
    outside = float(np.sum(map01 * (1.0 - mask01)))
    return outside / total


def var_inside_mask(map01: np.ndarray, mask01: np.ndarray):
    inside = map01[mask01 > 0.5]
    if inside.size < 64:
        return 0.0
    return float(np.var(inside))


def clip01(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return x


def quality_check_one(
    img_path: str,
    npy_path: str,
    min_mask_ratio: float,
    max_mask_ratio: float,
    max_outside_ratio: float,
    min_var_inside: float,
):
    """
    return: (ok:bool, metrics:dict, reason:str)
    """
    metrics = {}
    reason = ""

    img = safe_imread(img_path)
    if img is None:
        return False, metrics, "img_read_fail"

    try:
        cond = load_cond_npy(npy_path)
    except Exception as e:
        return False, metrics, f"npy_load_fail:{e}"

    if not np.isfinite(cond).all():
        return False, metrics, "npy_has_nan_inf"

    # (4,H,W)
    if cond.shape[0] < 4:
        return False, metrics, f"npy_channels_lt4:{cond.shape}"

    H, W = img.shape[:2]
    if cond.shape[1] != H or cond.shape[2] != W:
        return False, metrics, f"shape_mismatch img({H},{W}) npy({cond.shape[1]},{cond.shape[2]})"

    redness = clip01(cond[0])
    wrinkle = clip01(cond[1])
    pore = clip01(cond[2])
    mask = binarize_mask(cond[3])

    mask_ratio = float(np.mean(mask))
    metrics["mask_ratio"] = mask_ratio

    if mask_ratio < min_mask_ratio:
        return False, metrics, "mask_too_small"
    if mask_ratio > max_mask_ratio:
        return False, metrics, "mask_too_large"

    # map 에너지가 마스크 밖으로 많이 새면 (정렬 불량/마스크 불량 가능)
    outside_red = energy_outside_mask(redness, mask)
    outside_wri = energy_outside_mask(wrinkle, mask)
    outside_por = energy_outside_mask(pore, mask)

    metrics["outside_red_ratio"] = outside_red
    metrics["outside_wrinkle_ratio"] = outside_wri
    metrics["outside_pore_ratio"] = outside_por

    if max(outside_red, outside_wri, outside_por) > max_outside_ratio:
        return False, metrics, "too_much_outside_mask"

    # 맵이 거의 상수(정보 없음)면 제외
    var_red = var_inside_mask(redness, mask)
    var_wri = var_inside_mask(wrinkle, mask)
    var_por = var_inside_mask(pore, mask)

    metrics["var_red_inside"] = var_red
    metrics["var_wrinkle_inside"] = var_wri
    metrics["var_pore_inside"] = var_por

    if max(var_red, var_wri, var_por) < min_var_inside:
        return False, metrics, "maps_too_flat"

    # 이미지가 지나치게 어둡거나(거의 검정) 깨졌을 가능성
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    metrics["img_mean"] = float(np.mean(gray))
    metrics["img_std"] = float(np.std(gray))
    if metrics["img_std"] < 2.0:
        return False, metrics, "img_low_contrast"

    return True, metrics, "ok"


def move_or_copy(src: str, dst: str, action: str):
    ensure_dir(os.path.dirname(dst))
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(src, dst)
    else:
        raise ValueError("action must be copy or move")


def make_viz_sample(out_path: str, samples: list, title: str, max_cols: int = 4):
    """
    samples: list of dict {img_path, npy_path}
    """
    if not samples:
        return

    tiles = []
    for s in samples:
        img = safe_imread(s["img_path"])
        if img is None:
            continue
        try:
            cond = load_cond_npy(s["npy_path"])
        except:
            continue
        mask = binarize_mask(cond[3])
        edges = mask_boundary(mask)
        ov = overlay_edges(img, edges)

        # 왼쪽: overlay / 오른쪽: (pore,wrinkle,red) 합성
        pore = (clip01(cond[2]) * 255).astype(np.uint8)
        wrinkle = (clip01(cond[1]) * 255).astype(np.uint8)
        red = (clip01(cond[0]) * 255).astype(np.uint8)
        vis = cv2.merge([pore, wrinkle, red])  # B,G,R처럼 보이지만 그냥 비교용

        pair = np.hstack([ov, vis])
        tiles.append(pair)

    if not tiles:
        return

    # resize tiles to same height
    h = min(t.shape[0] for t in tiles)
    resized = []
    for t in tiles:
        if t.shape[0] != h:
            scale = h / t.shape[0]
            w = int(t.shape[1] * scale)
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(t)

    cols = min(max_cols, len(resized))
    rows = int(np.ceil(len(resized) / cols))

    # pad last row
    blank = np.zeros_like(resized[0])
    while len(resized) < rows * cols:
        resized.append(blank)

    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(resized[r * cols : (r + 1) * cols]))
    grid = np.vstack(row_imgs)

    # title bar
    bar_h = 40
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([bar, grid])

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True, help="processed dataset root (contains images/ maps/)")
    parser.add_argument("--img_dir", type=str, default=None, help="override images dir")
    parser.add_argument("--map_dir", type=str, default=None, help="override maps dir")

    parser.add_argument("--report_csv", type=str, default="./qc_report.csv", help="csv path")
    parser.add_argument("--bad_dir", type=str, default="./qc_bad", help="where to put bad samples (images/maps)")
    parser.add_argument("--action", type=str, default="copy", choices=["copy", "move"], help="copy/move bad samples")

    parser.add_argument("--min_mask_ratio", type=float, default=0.05)
    parser.add_argument("--max_mask_ratio", type=float, default=0.60)
    parser.add_argument("--max_outside_ratio", type=float, default=0.25)
    parser.add_argument("--min_var_inside", type=float, default=1e-4)

    parser.add_argument("--viz_n", type=int, default=16, help="number of good/bad samples to visualize")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    img_dir = args.img_dir or os.path.join(args.data_root, "images")
    map_dir = args.map_dir or os.path.join(args.data_root, "maps")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"images dir not found: {img_dir}")
    if not os.path.isdir(map_dir):
        raise FileNotFoundError(f"maps dir not found: {map_dir}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_files.sort()

    rows = []
    good = []
    bad = []

    bad_img_out = os.path.join(args.bad_dir, "images")
    bad_map_out = os.path.join(args.bad_dir, "maps")

    print(f"[QC] images={img_dir}")
    print(f"[QC] maps  ={map_dir}")
    print(f"[QC] total images={len(img_files)}")

    for f in tqdm(img_files, desc="QC"):
        base = os.path.splitext(f)[0]
        img_path = os.path.join(img_dir, f)
        npy_path = os.path.join(map_dir, f"{base}_cond.npy")
        vis_path = os.path.join(map_dir, f"{base}_vis.png")

        if not os.path.exists(npy_path):
            rows.append({"name": base, "ok": False, "reason": "missing_npy"})
            bad.append({"img_path": img_path, "npy_path": npy_path})
            continue

        ok, metrics, reason = quality_check_one(
            img_path=img_path,
            npy_path=npy_path,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            max_outside_ratio=args.max_outside_ratio,
            min_var_inside=args.min_var_inside,
        )

        row = {"name": base, "ok": ok, "reason": reason}
        row.update(metrics)
        rows.append(row)

        if ok:
            good.append({"img_path": img_path, "npy_path": npy_path})
        else:
            bad.append({"img_path": img_path, "npy_path": npy_path})

            # bad 샘플 격리 (image/npy + vis 있으면 같이)
            try:
                move_or_copy(img_path, os.path.join(bad_img_out, f), args.action)
            except Exception:
                pass

            try:
                move_or_copy(npy_path, os.path.join(bad_map_out, os.path.basename(npy_path)), args.action)
            except Exception:
                pass

            if os.path.exists(vis_path):
                try:
                    move_or_copy(vis_path, os.path.join(bad_map_out, os.path.basename(vis_path)), args.action)
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(os.path.abspath(args.report_csv)))
    df.to_csv(args.report_csv, index=False, encoding="utf-8-sig")

    print(f"[QC Done] good={len(good)}, bad={len(bad)}")
    print(f"[Report] {os.path.abspath(args.report_csv)}")
    print(f"[Bad Dir] {os.path.abspath(args.bad_dir)} (action={args.action})")

    # sample viz
    n = max(0, int(args.viz_n))
    if n > 0:
        good_s = random.sample(good, k=min(n, len(good))) if good else []
        bad_s = random.sample(bad, k=min(n, len(bad))) if bad else []

        make_viz_sample(
            out_path=os.path.join(os.path.dirname(args.report_csv), "qc_good_samples.png"),
            samples=good_s,
            title=f"QC GOOD samples (n={len(good_s)}) | left=mask overlay, right=pore/wrinkle/red",
        )
        make_viz_sample(
            out_path=os.path.join(os.path.dirname(args.report_csv), "qc_bad_samples.png"),
            samples=bad_s,
            title=f"QC BAD samples (n={len(bad_s)}) | left=mask overlay, right=pore/wrinkle/red",
        )
        print("[Viz] qc_good_samples.png / qc_bad_samples.png saved next to report_csv")


if __name__ == "__main__":
    main()
