import cv2
import numpy as np


# -------------------------
# Common Utilities
# -------------------------
def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _prepare_mask(mask: np.ndarray | None, shape_hw: tuple[int, int], erode_ksize: int = 0) -> np.ndarray | None:
    """
    mask를 (H,W) float32 0/1로 정리.
    - mask가 None이면 None 반환
    - shape이 다르면 resize
    - 값 범위가 0~255 / 0~1 / 기타여도 자동 threshold
    - erode_ksize > 0이면 경계 잡음 줄이기 위한 erosion 적용
    """
    if mask is None:
        return None

    m = np.asarray(mask)
    if m.ndim != 2:
        # (H,W,1) 같은 경우 대비
        m = np.squeeze(m)
        if m.ndim != 2:
            raise ValueError(f"mask must be 2D (H,W), got {m.shape}")

    H, W = shape_hw
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    m = _to_float32(m)

    # 값 범위에 따라 threshold 자동화
    # - 0~255면 127 기준
    # - 0~1이면 0.5 기준
    mx = float(np.max(m)) if m.size else 0.0
    thr = 127.0 if mx > 1.5 else 0.5
    m = (m > thr).astype(np.float32)

    if erode_ksize and erode_ksize > 0:
        k = int(erode_ksize)
        k = k if (k % 2 == 1) else (k + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(m, kernel, iterations=1).astype(np.float32)

    return m


def normalize_by_percentile(src: np.ndarray, mask: np.ndarray | None = None, p: tuple[int, int] = (2, 98)) -> np.ndarray:
    """
    src를 percentile 기반으로 0~1 normalize.
    - mask가 있으면 mask 내부 값만으로 percentile 계산
    - src/mask shape 불일치 시 src shape 기준으로 mask resize(권장: 호출 전에 맞춰주기)
    - 값이 너무 적거나 percentile이 망가지면 0맵 반환
    """
    x = np.asarray(src)
    if x.ndim != 2:
        x = np.squeeze(x)
        if x.ndim != 2:
            raise ValueError(f"src must be 2D (H,W), got {x.shape}")

    x = _to_float32(x)

    if mask is not None:
        m = _prepare_mask(mask, (x.shape[0], x.shape[1]), erode_ksize=0)
        vals = x[m > 0.5].ravel()
    else:
        vals = x.ravel()

    # finite 값만 사용
    vals = vals[np.isfinite(vals)]
    if vals.size < 32:
        out = np.zeros_like(x, dtype=np.float32)
        if mask is not None:
            out[m <= 0.5] = 0.0
        return out

    lo, hi = np.percentile(vals, p)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        out = np.zeros_like(x, dtype=np.float32)
        if mask is not None:
            out[m <= 0.5] = 0.0
        return out

    denom = max(1e-6, float(hi - lo))
    out = (x - float(lo)) / denom
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if mask is not None:
        out[m <= 0.5] = 0.0
    return out


# -------------------------
# 1) Wrinkle Map (Laplacian + Gabor)
# -------------------------
def get_wrinkle_map(img_gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    입력:
      - img_gray: (H,W) uint8/float
      - mask: (H,W) 0/1 혹은 0~255 혹은 0~1
    출력:
      - wrinkle_map: (H,W) float32, 0~1, mask 밖 0
    """
    g = np.asarray(img_gray)
    if g.ndim != 2:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = _to_float32(g)

    H, W = g.shape[:2]
    m = _prepare_mask(mask, (H, W), erode_ksize=3) if mask is not None else None

    # Laplacian (float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)

    # Gabor bank (losses.py와 동일 파라미터)
    ksize = 31
    gabor_sum = np.zeros((H, W), dtype=np.float32)

    for theta_deg in (0, 45, 90, 135):
        theta = np.deg2rad(theta_deg)
        for lam in (4.0, 8.0):
            for sigma in (2.0, 3.0):
                kern = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    theta,
                    lam,
                    0.5,
                    0,
                    ktype=cv2.CV_32F,
                )
                fimg = cv2.filter2D(g, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT101)
                gabor_sum += np.abs(fimg)

    # Normalize & Fusion
    lap_n = normalize_by_percentile(lap, m, p=(2, 98))
    gab_n = normalize_by_percentile(gabor_sum, m, p=(2, 98))

    wr = 0.6 * lap_n + 0.4 * gab_n
    wr = np.clip(wr, 0.0, 1.0).astype(np.float32)

    if m is not None:
        wr[m <= 0.5] = 0.0
    return wr


# -------------------------
# 2) Pore Map (DoG)
# -------------------------
def get_pore_map(img_gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    DoG 기반 모공 강조 맵.
    """
    g = np.asarray(img_gray)
    if g.ndim != 2:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = _to_float32(g)

    H, W = g.shape[:2]
    m = _prepare_mask(mask, (H, W), erode_ksize=3) if mask is not None else None

    g1 = cv2.GaussianBlur(g, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101)
    g2 = cv2.GaussianBlur(g, (0, 0), 2.5, borderType=cv2.BORDER_REFLECT101)
    dog = np.abs(g1 - g2).astype(np.float32)

    pore = normalize_by_percentile(dog, m, p=(5, 95))
    pore = np.clip(pore, 0.0, 1.0).astype(np.float32)

    if m is not None:
        pore[m <= 0.5] = 0.0
    return pore


# -------------------------
# 3) Redness Map (LAB a-channel + smoothing)
# -------------------------
def get_redness_map(img_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    홍조: LAB의 a 채널(128 기준)에서 양수만 사용.
    guidedFilter가 있으면 guidedFilter, 없으면 bilateral -> gaussian fallback.
    """
    bgr = np.asarray(img_bgr)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"img_bgr must be (H,W,3), got {bgr.shape}")

    H, W = bgr.shape[:2]
    m = _prepare_mask(mask, (H, W), erode_ksize=3) if mask is not None else None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, _ = cv2.split(lab)

    # OpenCV LAB: a가 0~255, 128이 중립 -> (a-128) 양수만
    pos_a = np.maximum(a - 128.0, 0.0).astype(np.float32)

    # guide는 L을 사용(에지 보존)
    # guidedFilter는 cv2.ximgproc (opencv-contrib) 필요
    sm = None
    try:
        from cv2.ximgproc import guidedFilter  # type: ignore
        # guidedFilter는 float32도 OK
        sm = guidedFilter(guide=L, src=pos_a, radius=15, eps=0.01)
    except Exception:
        # 1) bilateral: 피부 텍스처를 너무 죽이지 않으면서 완만하게
        try:
            sm = cv2.bilateralFilter(pos_a, d=0, sigmaColor=25, sigmaSpace=7)
        except Exception:
            sm = None

    if sm is None:
        # 최후 fallback
        sm = cv2.GaussianBlur(pos_a, (0, 0), 5, borderType=cv2.BORDER_REFLECT101)

    red = normalize_by_percentile(sm, m, p=(2, 98))
    red = np.clip(red, 0.0, 1.0).astype(np.float32)

    if m is not None:
        red[m <= 0.5] = 0.0
    return red
