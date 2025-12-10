import cv2
import numpy as np

# 1. 공통 유틸리티
def normalize_by_percentile(src, mask=None, p=(2, 98)):
    src = src.astype(np.float32)
    if mask is not None:
        vals = src[mask > 0].ravel()
    else:
        vals = src.ravel()
    if vals.size < 10: return np.zeros_like(src, dtype=np.float32)
    lo, hi = np.percentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(src, dtype=np.float32)
    denom = max(1e-6, hi - lo)
    dst = np.clip((src - lo) / denom, 0.0, 1.0)
    if mask is not None: dst[mask == 0] = 0.0
    return dst

# 2. 주름 (Wrinkle) - Gabor + Laplacian (복구됨)
def get_wrinkle_map(img_gray, mask):
    # Laplacian
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    laplacian = np.abs(laplacian)
    
    # Gabor Bank
    gabor_response = np.zeros_like(img_gray, dtype=np.float64)
    ksize = 31
    # losses.py와 동일한 파라미터 사용
    filters = []
    for theta in [0, 45, 90, 135]:
        for lam in [4.0, 8.0]:
            for sigma in [2.0, 3.0]:
                kern = cv2.getGaborKernel((ksize, ksize), sigma, np.deg2rad(theta), lam, 0.5, 0, ktype=cv2.CV_64F)
                fimg = cv2.filter2D(img_gray, cv2.CV_64F, kern)
                gabor_response += np.abs(fimg)
    
    # Normalize & Fusion
    lap_norm = normalize_by_percentile(laplacian, mask)
    gabor_norm = normalize_by_percentile(gabor_response, mask)
    wrinkle_map = 0.6 * lap_norm + 0.4 * gabor_norm
    return np.clip(wrinkle_map, 0, 1).astype(np.float32)

# 3. 모공 (Pore) - DoG (유지)
def get_pore_map(img_gray, mask):
    g1 = cv2.GaussianBlur(img_gray, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101)
    g2 = cv2.GaussianBlur(img_gray, (0, 0), 2.5, borderType=cv2.BORDER_REFLECT101)
    dog = np.abs(g1.astype(np.float32) - g2.astype(np.float32))
    return normalize_by_percentile(dog, mask, p=(5, 95)).astype(np.float32)

# 4. 홍조 (Redness) - Guided Filter (유지)
def get_redness_map(img_bgr, mask):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(img_lab)
    pos_a = np.maximum(a - 128.0, 0.0)
    try:
        from cv2.ximgproc import guidedFilter
        redness_map = guidedFilter(guide=L, src=pos_a, radius=15, eps=0.01)
    except:
        redness_map = cv2.GaussianBlur(pos_a, (0, 0), 5)
    return normalize_by_percentile(redness_map, mask, p=(2, 98)).astype(np.float32)