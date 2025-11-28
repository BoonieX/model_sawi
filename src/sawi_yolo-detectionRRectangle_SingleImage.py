#!/usr/bin/env python3
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# -------------------- CONFIGURATION --------------------
CONFIG = {
    # I/O
    "INPUT_IMAGE": "test_image/sawi_1.jpg",
    "OUT_DIR": "test_image/output_yolo_row",

    # YOLO
    "YOLO_WEIGHTS": "./runs/detect/train2/weights/best.pt",
    "YOLO_CONF": 0.59,
    "PLANT_CLASS_ID": 0,        # <-- change if your plant class ID is different
    "USE_CLASS_FILTER": True,   # set False if your model has only 1 class

    # ArUco calibration
    "ARUCO_DICT": cv2.aruco.DICT_4X4_50,
    "MARKER_SIZE_CM": 6.8,      # physical size of marker edge

    # Green mask (for measuring plant spine INSIDE each box)
    "H_MIN": 30, "S_MIN": 40,  "V_MIN": 40,
    "H_MAX": 90, "S_MAX": 255, "V_MAX": 255,

    # Row logic
    "NUM_ROWS": 3,              # low / medium / high
    "ROW_PADDING_PX": 10,       # padding around row wrapper when drawing

    # Optional: limit plants horizontally around ArUco (set None to disable)
    # Example: 300 means only plants whose centers are within 300px in X from the ArUco center
    "X_TOLERANCE_FROM_ARUCO": None
}

# -------------------- UTILITIES --------------------


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def get_curve_length(img_crop, px_per_cm, vis_img=None, offset=(0, 0)):
    """
    Measures the 'spine' of the plant inside the YOLO box using Rotated Rectangle.
    Uses green HSV mask only INSIDE the crop (does not affect row detection).
    """
    if img_crop is None or img_crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([CONFIG["H_MIN"], CONFIG["S_MIN"], CONFIG["V_MIN"]])
    upper = np.array([CONFIG["H_MAX"], CONFIG["S_MAX"], CONFIG["V_MAX"]])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (center, (w, h), angle) = rect
    length_px = max(w, h)

    if vis_img is not None:
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box[:, 0] += offset[0]
        box[:, 1] += offset[1]
        cv2.drawContours(vis_img, [box], 0, (255, 100, 100), 2)

    if px_per_cm <= 0:
        return 0.0

    return float(length_px) / px_per_cm


# -------------------- ROW CLUSTERING (1D K-MEANS) --------------------

def kmeans_1d(values, k, max_iters=50):
    """
    Very simple 1D k-means on an array of shape (N,).
    Returns centers (k,) and labels (N,).
    """
    values = np.asarray(values, dtype=np.float32)
    N = len(values)
    if N == 0 or k <= 0:
        return np.array([]), np.array([])

    # If there are fewer points than clusters, just return each as its own cluster
    if N <= k:
        centers = np.array(sorted(list(set(values))))
        labels = np.zeros_like(values, dtype=int)
        return centers, labels

    # Initialize centers linearly between min and max
    vmin, vmax = values.min(), values.max()
    centers = np.linspace(vmin, vmax, k, dtype=np.float32)

    for _ in range(max_iters):
        # Assign step
        distances = np.abs(values[:, None] - centers[None, :])  # (N, k)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for ci in range(k):
            cluster_vals = values[labels == ci]
            if len(cluster_vals) > 0:
                new_centers[ci] = cluster_vals.mean()

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


# -------------------- CORE LOGIC --------------------

def detect_aruco_and_calibrate(frame):
    """
    Detects the largest ArUco marker, returns:
    - px_per_cm
    - aruco_box (x1, y1, x2, y2)
    - aruco_center (cx, cy)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(CONFIG['ARUCO_DICT'])
    detector = cv2.aruco.ArucoDetector(aruco_dict,
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(corners) == 0:
        return None, None, None

    # pick the largest marker
    idx = np.argmax([cv2.contourArea(c[0]) for c in corners])
    mk = corners[idx][0]  # (4, 2)

    # pixels per cm (edge length)
    px_per_cm = np.linalg.norm(mk[0] - mk[1]) / CONFIG['MARKER_SIZE_CM']

    ax1, ay1 = int(np.min(mk[:, 0])), int(np.min(mk[:, 1]))
    ax2, ay2 = int(np.max(mk[:, 0])), int(np.max(mk[:, 1]))
    aruco_box = (ax1, ay1, ax2, ay2)
    aruco_center = ((ax1 + ax2) // 2, (ay1 + ay2) // 2)

    return px_per_cm, aruco_box, aruco_center


def detect_plants_with_yolo(frame):
    """
    Runs YOLO, returns a list of dicts:
    [
      {
        "box": (x1, y1, x2, y2),
        "cx": center_x,
        "cy": center_y,
        "conf": confidence,
        "cls": class_id
      },
      ...
    ]
    """
    model = YOLO(CONFIG['YOLO_WEIGHTS'])
    results = model.predict(frame, conf=CONFIG['YOLO_CONF'], verbose=False)

    plants = []
    if not results or not results[0].boxes:
        return plants

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0]) if box.cls is not None else -1

        if CONFIG["USE_CLASS_FILTER"] and cls_id != CONFIG["PLANT_CLASS_ID"]:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        plants.append({
            "box": (x1, y1, x2, y2),
            "cx": cx,
            "cy": cy,
            "conf": conf,
            "cls": cls_id
        })

    return plants


def assign_rows_by_clustering(plants):
    """
    Uses 1D k-means on plant center Y positions to assign row_id to each plant.
    Modifies 'plants' in-place, returns row_centers (array of size NUM_ROWS).
    """
    if not plants:
        return np.array([])

    cys = np.array([p["cy"] for p in plants], dtype=np.float32)
    centers, labels = kmeans_1d(cys, CONFIG["NUM_ROWS"])

    if centers.size == 0:
        return np.array([])

    # Attach row_id to each plant
    for p, row_id in zip(plants, labels):
        p["row_id"] = int(row_id)

    return centers


def select_active_row(plants, row_centers, aruco_center):
    """
    Chooses which row is 'active' based on which row center is closest
    to the ArUco center Y.
    Returns:
      active_row_id (int) or None if fails
    """
    if row_centers.size == 0 or aruco_center is None:
        return None

    _, aruco_cy = aruco_center

    # choose row whose center is closest to ArUco Y
    diffs = np.abs(row_centers - aruco_cy)
    active_row_id = int(np.argmin(diffs))
    return active_row_id


def filter_plants_in_active_row(plants, active_row_id, aruco_center):
    """
    Returns only plants that belong to the chosen row.
    Optionally applies X-distance filter around ArUco.
    """
    if active_row_id is None:
        return []

    row_plants = [p for p in plants if p.get("row_id") == active_row_id]

    xtol = CONFIG["X_TOLERANCE_FROM_ARUCO"]
    if xtol is not None and aruco_center is not None:
        aruco_cx, _ = aruco_center
        row_plants = [
            p for p in row_plants
            if abs(p["cx"] - aruco_cx) <= xtol
        ]

    return row_plants


def compute_row_wrapper(row_plants, img_shape):
    """
    Based on all plants in active row, compute a big bounding box (wrapper).
    """
    if not row_plants:
        return None

    h, w = img_shape[:2]

    xs1 = [p["box"][0] for p in row_plants]
    ys1 = [p["box"][1] for p in row_plants]
    xs2 = [p["box"][2] for p in row_plants]
    ys2 = [p["box"][3] for p in row_plants]

    wx1, wy1 = min(xs1), min(ys1)
    wx2, wy2 = max(xs2), max(ys2)

    pad = CONFIG["ROW_PADDING_PX"]
    wx1 = max(0, wx1 - pad)
    wy1 = max(0, wy1 - pad)
    wx2 = min(w - 1, wx2 + pad)
    wy2 = min(h - 1, wy2 + pad)

    return (wx1, wy1, wx2, wy2)


# -------------------- MAIN --------------------

def main():
    ensure_dir(CONFIG['OUT_DIR'])

    print(f"Processing {CONFIG['INPUT_IMAGE']}...")
    frame = cv2.imread(CONFIG['INPUT_IMAGE'])
    if frame is None:
        print("Error: Image not found.")
        return

    vis = frame.copy()
    h, w = frame.shape[:2]

    # 1. Detect ArUco & calibration
    px_per_cm, aruco_box, aruco_center = detect_aruco_and_calibrate(frame)
    if aruco_box is None:
        print("No ArUco Found.")
        return

    ax1, ay1, ax2, ay2 = aruco_box
    mk_poly = np.array([
        [ax1, ay1],
        [ax2, ay1],
        [ax2, ay2],
        [ax1, ay2]
    ], dtype=int)

    # draw ArUco anchor area (approx box)
    cv2.polylines(vis, [mk_poly], True, (0, 255, 255), 2)
    draw_text(vis, "ANCHOR", (ax1, ay1 - 10), (0, 255, 255))

    # 2. Detect all plants via YOLO
    plants = detect_plants_with_yolo(frame)
    if not plants:
        print("No plants detected by YOLO.")
        return

    # 3. Assign row for each plant by clustering on center Y
    row_centers = assign_rows_by_clustering(plants)
    if row_centers.size == 0:
        print("Row clustering failed.")
        return

    # (Optional) visualize row centers for debugging
    # for ci, center_y in enumerate(row_centers):
    #     cv2.line(vis, (0, int(center_y)), (w, int(center_y)), (255, 0, 0), 1)
    #     draw_text(vis, f"row_center_{ci}", (5, int(center_y) - 5), (255, 0, 0))

    # 4. Decide which row is active (closest to ArUco center Y)
    active_row_id = select_active_row(plants, row_centers, aruco_center)
    if active_row_id is None:
        print("Could not determine active row.")
        return

    print(f"Active row id: {active_row_id}")

    # 5. Filter plants that belong to active row (+ optional X tolerance)
    row_plants = filter_plants_in_active_row(plants, active_row_id, aruco_center)
    if not row_plants:
        print("No plants in the active row.")
        return

    # 6. Compute row wrapper
    wrapper_box = compute_row_wrapper(row_plants, frame.shape)
    if wrapper_box is not None:
        wx1, wy1, wx2, wy2 = wrapper_box
        cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
        draw_text(vis, "ACTIVE ROW", (wx1, wy1 - 10), (0, 0, 255))

    # 7. Measure each plant in active row
    csv_data = []

    for i, p in enumerate(row_plants, start=1):
        x1, y1, x2, y2 = p["box"]

        # Draw the plant box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        crop = frame[max(0, y1):min(h, y2),
                     max(0, x1):min(w, x2)]

        height_cm = get_curve_length(crop, px_per_cm, vis, (x1, y1))

        draw_text(vis, f"#{i} {height_cm:.1f}cm", (x1, y1 + 20), (0, 255, 0))
        csv_data.append([i, height_cm])

    # 8. Save visualization
    save_path = os.path.join(
        CONFIG['OUT_DIR'],
        f"yolo_row_{os.path.basename(CONFIG['INPUT_IMAGE'])}"
    )
    cv2.imwrite(save_path, vis)
    print(f"Saved output image: {save_path}")

    # 9. Append CSV measurements
    if csv_data:
        csv_path = os.path.join(CONFIG['OUT_DIR'], "measurements.csv")
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for pid, ht in csv_data:
                writer.writerow([ts, pid, ht])
                print(f"  Plant #{pid}: {ht:.1f} cm")
    else:
        print("No measurements written to CSV (no plants in active row).")


if __name__ == "__main__":
    main()
