#!/usr/bin/env python3
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import math

# ================================================================
# CONFIGURATION
# ================================================================
CONFIG = {
    # ------------------------------------------------------------
    # INPUTS / OUTPUTS
    # ------------------------------------------------------------
    "INPUT_IMAGE": "test_image/sawi_3.jpeg",
    "OUT_DIR": "debug/output_full_system",

    # ------------------------------------------------------------
    # YOLO MODEL (YOUR TRAINED BEST.PT)
    # ------------------------------------------------------------
    "YOLO_WEIGHTS": r"./runs/detect/train2/weights/best.pt",
    "YOLO_CONF": 0.59,
    "USE_CLASS_FILTER": False,    # True only if YOLO has multiple classes
    "PLANT_CLASS_ID": 0,

    # ------------------------------------------------------------
    # ARUCO SETTINGS
    # ------------------------------------------------------------
    "ARUCO_DICT": cv2.aruco.DICT_4X4_50,
    "MARKER_SIZE_CM": 6.8,        # Edge size in cm

    # ------------------------------------------------------------
    # PIPE DETECTION SETTINGS (HoughLinesP)
    # ------------------------------------------------------------
    "PIPE_MIN_LENGTH": 250,       # minimum length of detected line (px)
    "PIPE_MAX_GAP": 20,           # max gap for HoughLinesP
    "PIPE_THRESHOLD": 50,         # threshold for Hough
    "PIPE_GROUP_DISTANCE": 50,    # Y-distance to group lines into one pipe

    # ------------------------------------------------------------
    # ROW SETTINGS
    # ------------------------------------------------------------
    "ROW_PADDING_PX": 12,
    "MIN_PLANTS_PER_ROW": 1,
    "ALLOW_YOLO_ROW_FALLBACK": True,

    # ------------------------------------------------------------
    # GREEN MASK FOR HEIGHT MEASUREMENT
    # ------------------------------------------------------------
    "H_MIN": 30, "S_MIN": 40, "V_MIN": 40,
    "H_MAX": 90, "S_MAX": 255, "V_MAX": 255,
}

# ================================================================
# BASIC UTILITIES
# ================================================================
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def draw_text(img, text, pos, color=(255,255,255), size=0.6, thick=1):
    x, y = pos
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thick, cv2.LINE_AA)

def draw_line(img, p1, p2, color, w=2):
    cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color, w)

def draw_circle(img, pos, r, color, w=2):
    cv2.circle(img, (int(pos[0]),int(pos[1])), r, color, w)

def midpoint(a,b):
    return int((a[0]+b[0])/2), int((a[1]+b[1])/2)
# ================================================================
# ARUCO DETECTION + CALIBRATION
# ================================================================
def detect_aruco_markers(frame):
    """
    Detect ALL ArUco markers in the image.
    Return:
        arucos = list of dicts:
                 { "corners": 4x2 array, "center": (cx,cy), "box": (x1,y1,x2,y2), "size_px": float }
        px_per_cm = pixel_per_cm based on the LARGEST marker
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(CONFIG["ARUCO_DICT"])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return [], None

    arucos = []
    sizes = []

    for mk in corners:
        pts = mk[0]                          # shape (4,2)
        cx = int(np.mean(pts[:,0]))
        cy = int(np.mean(pts[:,1]))
        x1 = int(np.min(pts[:,0]))
        y1 = int(np.min(pts[:,1]))
        x2 = int(np.max(pts[:,0]))
        y2 = int(np.max(pts[:,1]))
        size_px = float(np.linalg.norm(pts[0] - pts[1]))

        arucos.append({
            "corners": pts,
            "center": (cx,cy),
            "box": (x1,y1,x2,y2),
            "size_px": size_px
        })

        sizes.append(size_px)

    # calibration: largest marker size
    px_per_cm = max(sizes)/CONFIG["MARKER_SIZE_CM"] if sizes else None

    return arucos, px_per_cm


# ================================================================
# PIPE DETECTION (HoughLinesP)
# ================================================================
def detect_pipes(frame):
    """
    Detect horizontal pipes using HoughLinesP.
    Returns:
        pipe_rows : list of pipe structures
        edges     : Canny edge image (for debug)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 1)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=CONFIG["PIPE_THRESHOLD"],
        minLineLength=CONFIG["PIPE_MIN_LENGTH"],
        maxLineGap=CONFIG["PIPE_MAX_GAP"]
    )

    if lines is None:
        return [], edges

    horizontals = []
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        if abs(y2 - y1) < 10:
            horizontals.append(((x1,y1),(x2,y2)))

    if not horizontals:
        return [], edges

    # Group horizontal segments by Y
    groups = []
    used = [False] * len(horizontals)

    for i, ln in enumerate(horizontals):
        if used[i]: continue
        x1,y1 = ln[0]
        x2,y2 = ln[1]
        cy = (y1+y2)/2

        group = [ln]
        used[i] = True

        for j, ln2 in enumerate(horizontals):
            if used[j]: continue
            cy2 = (ln2[0][1] + ln2[1][1]) / 2
            if abs(cy2 - cy) < CONFIG["PIPE_GROUP_DISTANCE"]:
                group.append(ln2)
                used[j] = True

        groups.append(group)

    pipe_rows = []
    for grp in groups:
        cys = [(ln[0][1] + ln[1][1]) / 2 for ln in grp]
        pipe_rows.append({
            "center_y": float(np.mean(cys)),
            "lines": grp
        })

    pipe_rows.sort(key=lambda r: r["center_y"])

    return pipe_rows, edges


def filter_invalid_pipes(pipe_rows, plants, frame_h):
    """
    Reject Hough pipes that are actually floor edges or noise.
    """
    if not pipe_rows:
        return []

    if not plants:
        return []

    valid_rows = []

    # Collect ALL plant bottoms (y2)
    plant_bottoms = [p["box"][3] for p in plants]
    min_y = min(plant_bottoms)
    max_y = max(plant_bottoms)

    for pr in pipe_rows:
        cy = pr["center_y"]
        line_span = max([abs(ln[0][0] - ln[1][0]) for ln in pr["lines"]])

        # ❌ Reject if line is WAY below the plants (floor edge)
        if cy > max_y + 70:  
            continue

        # ❌ Reject if line is too far above plants
        if cy < min_y - 150:
            continue

        # ❌ Reject if line is extremely long compared to pipe width
        if line_span > frame_h * 0.9:  
            continue

        # ❌ Reject if lines are too short (noise)
        if line_span < 80:
            continue

        # Passed all checks → valid pipe
        valid_rows.append(pr)

    return valid_rows


# ================================================================
# DEBUG: SAVE CANNY & HOUGH VISUALIZATIONS
# ================================================================
def debug_save_pipe_detection_images(frame, edges, pipe_rows, out_dir):
    """
    Save debug images:
        1) canny_debug.jpg   : shows Canny edge map
        2) hough_debug.jpg   : shows detected pipe lines overlayed on edges
    """
    ensure_dir(out_dir)

    # 1. Save the Canny edges
    canny_path = os.path.join(out_dir, "canny_debug.jpg")
    cv2.imwrite(canny_path, edges)
    
    # 2. Create a visualization image for Hough lines
    hough_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for i, pr in enumerate(pipe_rows):
        color = (0, 255, 255)  # yellow
        for (p1, p2) in pr["lines"]:
            cv2.line(hough_vis, p1, p2, color, 2)

        # draw pipe row center
        cy = int(pr["center_y"])
        cv2.line(hough_vis, (0, cy), (frame.shape[1]-1, cy), (0, 0, 255), 1)

    hough_path = os.path.join(out_dir, "hough_debug.jpg")
    cv2.imwrite(hough_path, hough_vis)

    print(f"Saved Canny debug:  {canny_path}")
    print(f"Saved Hough debug:  {hough_path}")


# ================================================================
# YOLO PLANT DETECTION
# ================================================================
def yolo_detect_plants(frame):
    model = YOLO(CONFIG["YOLO_WEIGHTS"])

    # Run YOLO with verbose for debugging
    results = model.predict(frame, conf=CONFIG["YOLO_CONF"], verbose=True)

    if not results:
        print("YOLO: no results object returned.")
        return []

    r = results[0]
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        print("YOLO: 0 boxes detected at all.")
        return []

    # Print model class names
    print("YOLO model classes (id:name):", model.names)

    # Print raw detections
    cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
    conf_list = boxes.conf.cpu().numpy().tolist()
    xyxy_list = boxes.xyxy.cpu().numpy().tolist()

    print(f"YOLO: {len(cls_list)} raw boxes detected:")
    for i, (cls_id, conf, bb) in enumerate(zip(cls_list, conf_list, xyxy_list)):
        print(f"  box {i}: cls={cls_id}, name={model.names.get(cls_id, '?')}, conf={conf:.3f}, xyxy={bb}")

    plants = []
    for (cls_id, conf, bb) in zip(cls_list, conf_list, xyxy_list):
        x1, y1, x2, y2 = map(int, bb)

        # Optional class filter
        if CONFIG["USE_CLASS_FILTER"]:
            if cls_id != CONFIG["PLANT_CLASS_ID"]:
                continue

        plants.append({
            "box": (x1, y1, x2, y2),
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
            "cls": cls_id
        })

    print(f"YOLO: {len(plants)} boxes left after class filter.")
    return plants


# ================================================================
# ROW ASSIGNMENT (PIPE-BASED USING y2 BOTTOM ANCHOR)
# ================================================================
def assign_rows_pipe_based(pipe_rows, plants):
    """
    Assign plants to nearest pipe (row) using the BOTTOM of the plant box (y2).
    This is the correct physical anchor because the net pot sits in the pipe.
    """
    if not pipe_rows:
        return []

    rows = [[] for _ in pipe_rows]

    for p in plants:
        x1, y1, x2, y2 = p["box"]

        # CRITICAL FIX:
        # Use y2 (bottom of the plant bounding box)
        # The pipe touches the pot, and pot is always at bottom of the box.
        plant_anchor_y = y2

        # Distance from plant-bottom to each pipe center
        dists = [abs(plant_anchor_y - pr["center_y"]) for pr in pipe_rows]
        nearest_row = int(np.argmin(dists))

        # Optional: sanity check (noise filter)
        # If plant bottom is extremely far from pipe row, skip it.
        if dists[nearest_row] < 200:  # Adjust based on your image resolution
            p["row_id"] = nearest_row
            rows[nearest_row].append(p)

    return rows



# ================================================================
# YOLO GAP SPLITTING (FALLBACK ROW DETECTION)
# ================================================================
def yolo_gap_split(plants, num_rows):
    """
    If no pipes are detected, split rows by largest vertical gaps in plant 'cy'.
    """
    if len(plants) == 0:
        return []

    plants_sorted = sorted(plants, key=lambda p: p["cy"])
    cys = [p["cy"] for p in plants_sorted]

    if len(plants_sorted) <= num_rows:
        rows = []
        for i,p in enumerate(plants_sorted):
            p["row_id"] = i
            rows.append([p])
        return rows

    gaps = [(cys[i+1]-cys[i], i) for i in range(len(cys)-1)]
    gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)

    boundary_count = min(num_rows-1, len(gaps_sorted))
    boundary_indices = sorted([gaps_sorted[i][1] for i in range(boundary_count)])

    rows = []
    start = 0
    for idx in boundary_indices:
        chunk = plants_sorted[start:idx+1]
        for p in chunk:
            p["row_id"] = len(rows)
        rows.append(chunk)
        start = idx+1

    last_chunk = plants_sorted[start:]
    for p in last_chunk:
        p["row_id"] = len(rows)
    rows.append(last_chunk)

    return rows


# ================================================================
# ARUCO > PIPE MAPPING
# ================================================================
def map_arucos_to_pipes(arucos, pipe_rows, vis):
    """
    Map each ArUco marker to nearest pipe (row).
    Draw debug arrows.
    Returns:
        aruco_row_map = dict: aruco_index -> row_index
        placement_side = "left" or "right"
    """
    h, w = vis.shape[:2]

    if not arucos:
        return {}, None

    # determine placement side: average ArUco cx
    mean_cx = np.mean([a["center"][0] for a in arucos])
    placement_side = "left" if mean_cx < w/2 else "right"

    aruco_row_map = {}

    if pipe_rows:
        for idx, a in enumerate(arucos):
            acx, acy = a["center"]
            # nearest pipe by Y
            dists = [abs(acy - pr["center_y"]) for pr in pipe_rows]
            nearest_pipe = int(np.argmin(dists))
            aruco_row_map[idx] = nearest_pipe

            # debug line
            pr = pipe_rows[nearest_pipe]
            pipe_y = pr["center_y"]
            draw_line(vis, (acx,acy), (acx,pipe_y), (255,100,0), 2)
            draw_circle(vis, (acx,pipe_y), 6, (255,100,0), -1)
    else:
        # no pipes: fallback (rows by YOLO)
        for idx, a in enumerate(arucos):
            aruco_row_map[idx] = None

    return aruco_row_map, placement_side


# ================================================================
# COMPUTE ROW WRAPPER BOUNDING BOX
# ================================================================
def compute_row_wrapper(plants, frame_shape):
    if not plants:
        return None

    h, w = frame_shape[:2]
    xs1 = [p["box"][0] for p in plants]
    ys1 = [p["box"][1] for p in plants]
    xs2 = [p["box"][2] for p in plants]
    ys2 = [p["box"][3] for p in plants]

    pad = CONFIG["ROW_PADDING_PX"]

    return (
        max(0, min(xs1)-pad),
        max(0, min(ys1)-pad),
        min(w-1, max(xs2)+pad),
        min(h-1, max(ys2)+pad)
    )


# ================================================================
# LEFT / RIGHT SORTING
# ================================================================
def sort_row_plants(plants, placement_side):
    """
    Sort plants within a row based on ArUco placement side.
    """
    if placement_side == "left":
        # normal left → right
        plants.sort(key=lambda p: p["box"][0])
    else:
        # right → left
        plants.sort(key=lambda p: p["box"][0], reverse=True)

    return plants


# ================================================================
# HEIGHT MEASUREMENT (HSV + Rotated Rectangle)
# ================================================================
def measure_plant_height(frame, vis, plant, px_cm):
    x1, y1, x2, y2 = plant["box"]

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([CONFIG["H_MIN"], CONFIG["S_MIN"], CONFIG["V_MIN"]])
    upper = np.array([CONFIG["H_MAX"], CONFIG["S_MAX"], CONFIG["V_MAX"]])
    mask = cv2.inRange(hsv, lower, upper)

    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (center, (w,h), angle) = rect
    length_px = max(w,h)

    # draw measurement box
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    box[:,0] += x1
    box[:,1] += y1
    cv2.drawContours(vis, [box], 0, (255,100,100), 2)

    height_cm = float(length_px)/px_cm if px_cm else 0.0
    return height_cm
# ================================================================
# MAIN PIPELINE + VISUALIZATION + CSV SAVING
# ================================================================
def main():
    ensure_dir(CONFIG["OUT_DIR"])

    frame = cv2.imread(CONFIG["INPUT_IMAGE"])
    if frame is None:
        print(f"Error: image not found: {CONFIG['INPUT_IMAGE']}")
        return

    vis = frame.copy()
    h, w = frame.shape[:2]

    # ------------------------------------------------------------
    # 1) Detect ArUco markers (possibly multiple) + calibration
    # ------------------------------------------------------------
    arucos, px_per_cm = detect_aruco_markers(frame)
    if not arucos:
        print("Warning: No ArUco markers found. Heights will be in pixels, not cm.")
        px_per_cm = None
    else:
        print(f"Detected {len(arucos)} ArUco marker(s). px/cm: {px_per_cm:.3f}" if px_per_cm else
              f"Detected {len(arucos)} ArUco marker(s), but calibration failed.")

    # Draw ArUco debug
    for i, a in enumerate(arucos):
        pts = a["corners"].astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
        cx, cy = a["center"]
        draw_circle(vis, (cx, cy), 4, (0, 255, 255), -1)
        draw_text(vis, f"ARUCO {i}", (cx + 5, cy - 5), (0, 255, 255))

    # ------------------------------------------------------------
    # 2) Detect pipes (rows) via HoughLinesP
    # ------------------------------------------------------------
    pipe_rows, edges = detect_pipes(frame)

    # DEBUG save raw hough before filtering
    debug_save_pipe_detection_images(frame, edges, pipe_rows, CONFIG["OUT_DIR"])

    # ------------------------------------------------------------
    # 3) YOLO plant detection (must happen BEFORE pipe validation)
    # ------------------------------------------------------------
    plants = yolo_detect_plants(frame)
    print(f"Detected {len(plants)} plant(s) with YOLO.")

    if not plants:
        outp = os.path.join(CONFIG["OUT_DIR"], "no_plants.jpg")
        cv2.imwrite(outp, vis)
        print(f"No plants found. Saved: {outp}")
        return

    # ------------------------------------------------------------
    # 4) Validate pipe rows using plant bottoms
    # ------------------------------------------------------------
    pipe_rows = filter_invalid_pipes(pipe_rows, plants, frame.shape[0])
    print(f"Valid pipe rows after filtering: {len(pipe_rows)}")

    # Save Canny + Hough debug images
    debug_save_pipe_detection_images(
        frame=frame,
        edges=edges,
        pipe_rows=pipe_rows,
        out_dir=CONFIG["OUT_DIR"]
    )

    # color palette for rows
    row_colors = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0)   # cyan
    ]

    # Draw pipe lines & row centers
    for i, pr in enumerate(pipe_rows):
        color = row_colors[i % len(row_colors)]
        for ln in pr["lines"]:
            draw_line(vis, ln[0], ln[1], color, 2)

        cy = int(pr["center_y"])
        draw_line(vis, (0, cy), (w - 1, cy), color, 1)
        draw_text(vis, f"PIPE_ROW {i}", (5, cy - 5), color)

    # ------------------------------------------------------------
    # 3) YOLO plant detection
    # ------------------------------------------------------------
    plants = yolo_detect_plants(frame)
    print(f"Detected {len(plants)} plant(s) with YOLO.")
    if not plants:
        output_path = os.path.join(CONFIG["OUT_DIR"], "no_plants_detected.jpg")
        cv2.imwrite(output_path, vis)
        print(f"No plants found. Saved debug image to {output_path}")
        return

    # Draw initial plant boxes faintly (for debug)
    for p in plants:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # ------------------------------------------------------------
    # 4) Map ArUco -> pipes, get placement side (left/right)
    # ------------------------------------------------------------
    aruco_row_map, placement_side = map_arucos_to_pipes(arucos, pipe_rows, vis)
    if placement_side is None:
        placement_side = "left"  # default if no ArUco at all
    print(f"ArUco placement side: {placement_side}")
    if aruco_row_map:
        print("ArUco → row mapping:", aruco_row_map)

    # ------------------------------------------------------------
    # 5) Build row structure (pipe-based primary, YOLO fallback)
    # ------------------------------------------------------------
    if pipe_rows:
        # use pipes as primary row separators
        rows = assign_rows_pipe_based(pipe_rows, plants)
        print("Row assignment: using pipe-based grouping.")
    else:
        # no pipes detected
        if CONFIG["ALLOW_YOLO_ROW_FALLBACK"]:
            # guess number of rows: if we have ArUcos, use that count; else 1
            num_rows_guess = len(arucos) if arucos else 1
            print(f"No pipes detected. Falling back to YOLO gap-split with {num_rows_guess} row(s).")
            rows = yolo_gap_split(plants, num_rows_guess)
        else:
            print("No pipes detected and YOLO fallback disabled. Using single row for all plants.")
            rows = [plants]

    # Filter rows that have too few plants (optional)
    row_infos = []  # list of (row_index, [plants])
    for i, r in enumerate(rows):
        if len(r) >= CONFIG["MIN_PLANTS_PER_ROW"]:
            row_infos.append((i, r))
        else:
            print(f"Skipping row {i} because it has only {len(r)} plant(s).")

    if not row_infos:
        print("No rows with enough plants after filtering.")
        output_path = os.path.join(CONFIG["OUT_DIR"], "no_valid_rows.jpg")
        cv2.imwrite(output_path, vis)
        print(f"Saved debug image to {output_path}")
        return

    # ------------------------------------------------------------
    # 6) For each row: sort plants, draw wrapper, measure heights
    # ------------------------------------------------------------
    csv_rows = []
    for display_idx, (row_id, row_plants) in enumerate(row_infos):
        color = row_colors[display_idx % len(row_colors)]

        # Sort plants within the row based on ArUco placement side
        sort_row_plants(row_plants, placement_side)

        # Compute row wrapper
        wrapper = compute_row_wrapper(row_plants, frame.shape)
        if wrapper is not None:
            wx1, wy1, wx2, wy2 = wrapper
            cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), color, 2)
            draw_text(vis, f"ROW {row_id}", (wx1 + 5, wy1 - 5), color)

        # Process plants in this row
        for plant_idx, p in enumerate(row_plants, start=1):
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Height measurement
            if px_per_cm:
                height_cm = measure_plant_height(frame, vis, p, px_per_cm)
                label = f"R{row_id} P{plant_idx} {height_cm:.1f}cm"
                csv_rows.append((row_id, plant_idx, height_cm))
            else:
                # no calibration: report in px
                height_px = measure_plant_height(frame, vis, p, 1.0)
                label = f"R{row_id} P{plant_idx} {height_px:.0f}px"
                csv_rows.append((row_id, plant_idx, height_px))

            draw_text(vis, label, (x1, max(0, y1 - 8)), color)

    # ------------------------------------------------------------
    # 7) Save visualization
    # ------------------------------------------------------------
    out_image_path = os.path.join(CONFIG["OUT_DIR"], "full_result.jpg")
    cv2.imwrite(out_image_path, vis)
    print(f"Saved visualization to: {out_image_path}")

    # ------------------------------------------------------------
    # 8) Save CSV
    # ------------------------------------------------------------
    if csv_rows:
        csv_path = os.path.join(CONFIG["OUT_DIR"], "measurements.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # header
                if px_per_cm:
                    writer.writerow(["timestamp", "row_id", "plant_id", "height_cm"])
                else:
                    writer.writerow(["timestamp", "row_id", "plant_id", "height_px"])

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row_id, plant_id, height in csv_rows:
                writer.writerow([ts, row_id, plant_id, height])

        print(f"Saved {len(csv_rows)} measurement(s) to: {csv_path}")
    else:
        print("No measurements to save.")


if __name__ == "__main__":
    main()
