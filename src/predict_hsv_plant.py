import cv2
import numpy as np
import glob
import os

# ==========================
# CONFIG
# ==========================

# Folder containing plant-only crops (sawi only).
# Each file should be a small image where almost all pixels are sawi.
CROPS_DIR = "sawi_crops"      # change if you use a different folder
PATTERN   = "*.jpg"           # or "*.png" depending on your files

# Percentiles used for robust min/max (ignore extreme outliers)
LOW_PCT  = 5.0
HIGH_PCT = 95.0


def main():
    paths = glob.glob(os.path.join(CROPS_DIR, PATTERN))
    if not paths:
        print(f"No images found in '{CROPS_DIR}' with pattern '{PATTERN}'.")
        print("Fill this folder with sawi-only crops first.")
        return

    all_hsv = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("Warning: failed to read", p)
            continue

        # Convert BGR -> HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Flatten to (N_pixels, 3) and append
        all_hsv.append(hsv.reshape(-1, 3))

    if not all_hsv:
        print("No valid images loaded.")
        return

    all_hsv = np.vstack(all_hsv)  # shape: (N_pixels_total, 3)
    H = all_hsv[:, 0].astype(np.float32)
    S = all_hsv[:, 1].astype(np.float32)
    V = all_hsv[:, 2].astype(np.float32)

    print(f"Total sawi pixels used: {len(all_hsv)}")

    # Raw min/max (for reference)
    print("Raw H min/max:", H.min(), H.max())
    print("Raw S min/max:", S.min(), S.max())
    print("Raw V min/max:", V.min(), V.max())

    # Robust percentiles
    h_low  = np.percentile(H, LOW_PCT)
    h_high = np.percentile(H, HIGH_PCT)
    s_low  = np.percentile(S, LOW_PCT)
    s_high = np.percentile(S, HIGH_PCT)
    v_low  = np.percentile(V, LOW_PCT)
    v_high = np.percentile(V, HIGH_PCT)

    print(f"H {LOW_PCT:.0f}–{HIGH_PCT:.0f} percentiles:", h_low, h_high)
    print(f"S {LOW_PCT:.0f}–{HIGH_PCT:.0f} percentiles:", s_low, s_high)
    print(f"V {LOW_PCT:.0f}–{HIGH_PCT:.0f} percentiles:", v_low, v_high)

    # Build suggested HSV_LOWER / HSV_UPPER
    hsv_lower = np.array([
        max(0, int(h_low  - 2)),   # small margin
        max(0, int(s_low  - 10)),
        max(0, int(v_low  - 10))
    ], dtype=np.uint8)

    hsv_upper = np.array([
        min(179, int(h_high + 2)), # H max is 179 in OpenCV
        min(255, int(s_high + 10)),
        min(255, int(v_high + 10))
    ], dtype=np.uint8)

    print("\nSuggested HSV thresholds for sawi:")
    print("HSV_LOWER =", hsv_lower.tolist())
    print("HSV_UPPER =", hsv_upper.tolist())

    print("\nCopy these into your main script, e.g.:")
    print(f"HSV_LOWER = np.array({hsv_lower.tolist()}, dtype=np.uint8)")
    print(f"HSV_UPPER = np.array({hsv_upper.tolist()}, dtype=np.uint8)")


if __name__ == "__main__":
    main()
