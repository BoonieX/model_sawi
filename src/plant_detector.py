#!/usr/bin/env python3
"""
Plant Height Monitor
- Supports multiple measurement algorithms.
- Supports multiple row detection methods.
- Continuous Auto-Calibration.
"""

import os
import cv2
import time
import queue
import threading
import csv
import numpy as np
import argparse
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import requests
from sort.sort import Sort

# -------------------- Configuration --------------------
CONFIG = {
    "URL": "http://10.31.238.252:8080/video", 
    "MODE": "stream",
    
    # AI / Detection
    "YOLO_WEIGHTS": "./runs/detect/train2/weights/best.pt", 
    "YOLO_CONF": 0.55,
    "IMG_SIZE": 640,
    
    # Calibration
    "ARUCO_DICT": "DICT_4X4_50",
    "MARKER_SIZE_CM": 6.8,

    # Row Detection
    "ROW_DETECTION_METHOD": "pipe", # "pipe" or "aruco"
    "PIPE_MIN_LENGTH": 250,
    "PIPE_MAX_GAP": 20,
    "PIPE_THRESHOLD": 50,
    "PIPE_GROUP_DISTANCE": 50,

    # Measurement
    "ALGORITHM": "spine", # "spine" or "rotated_rect"
    
    # HSV Color range for plant masking
    "H_MIN": 30, "S_MIN": 40,  "V_MIN": 40,
    "H_MAX": 90, "S_MAX": 255, "V_MAX": 255,
    
    # Tracking
    "TRACK_BUFFER": 60,
    
    # System
    "SAVE_INTERVAL": 10,
    "OUT_DIR": "output_frames",
    "CSV_FILE": "height_log.csv"
}

# -------------------- Utilities --------------------
def ensure_dir(d):
    """Ensures a directory exists."""
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    """Draws text with a black outline."""
    cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# -------------------- Pipe Detection --------------------
def detect_pipes(frame):
    """
    Detects horizontal pipes using HoughLinesP.
    Returns a list of pipe structures, where each pipe is a dictionary
    containing the center Y-coordinate and the line segments that form it.
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
        return []

    # Filter for near-horizontal lines
    horizontals = [ln[0] for ln in lines if abs(ln[0][3] - ln[0][1]) < 10]
    if not horizontals:
        return []

    # Group horizontal lines by their Y-coordinate
    groups = []
    used = [False] * len(horizontals)
    for i, (x1, y1, x2, y2) in enumerate(horizontals):
        if used[i]: continue
        
        current_group = [(x1, y1, x2, y2)]
        used[i] = True
        mean_y = (y1 + y2) / 2
        
        for j, (nx1, ny1, nx2, ny2) in enumerate(horizontals):
            if not used[j] and abs(((ny1 + ny2) / 2) - mean_y) < CONFIG["PIPE_GROUP_DISTANCE"]:
                current_group.append((nx1, ny1, nx2, ny2))
                used[j] = True
        groups.append(current_group)

    # Create pipe structures from groups
    pipe_rows = []
    for group in groups:
        ys = [(line[1] + line[3]) / 2 for line in group]
        pipe_rows.append({
            "center_y": float(np.mean(ys)),
            "lines": group
        })

    # Sort pipes from top to bottom
    pipe_rows.sort(key=lambda r: r["center_y"])
    return pipe_rows

def assign_rows_pipe_based(pipe_rows, detections):
    """
    Assigns each detection to the nearest pipe row based on the detection's
    bottom edge (y2). Returns a list of detections with an added 'row_id'.
    """
    if not pipe_rows:
        return detections # Return as is if no pipes

    for det in detections:
        # Use the bottom of the bounding box as the anchor point
        plant_anchor_y = det['box'][3] 
        
        # Find the closest pipe
        dists = [abs(plant_anchor_y - pr["center_y"]) for pr in pipe_rows]
        if dists:
            nearest_row_idx = np.argmin(dists)
            # Only assign if the plant is reasonably close to the pipe
            if dists[nearest_row_idx] < 200: # Heuristic distance threshold
                 det['row_id'] = nearest_row_idx
    return detections

# -------------------- Measurement Algorithms --------------------

def get_length_spine_slicing(img_crop, pixels_per_cm, debug_img=None, box_offset=(0,0)):
    """
    Calculates actual length of a bent plant using the "Spine Slicing" method.
    """
    if img_crop is None or img_crop.size == 0 or not pixels_per_cm:
        return 0.0

    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([CONFIG["H_MIN"], CONFIG["S_MIN"], CONFIG["V_MIN"]])
    upper = np.array([CONFIG["H_MAX"], CONFIG["S_MAX"], CONFIG["V_MAX"]])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    h, w = mask.shape
    num_slices = 15 
    slice_h = h // num_slices
    points = []
    
    for i in range(num_slices):
        y_start, y_end = h - (i + 1) * slice_h, h - i * slice_h
        slice_mask = mask[y_start:y_end, :]
        coords = cv2.findNonZero(slice_mask)
        if coords is not None:
            points.append((int(np.mean(coords[:, 0, 0])), int(y_start + slice_h / 2)))

    total_px_length = 0.0
    if len(points) > 1:
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            total_px_length += np.linalg.norm(np.array(p1) - np.array(p2))
            if debug_img is not None:
                g_p1 = (p1[0] + box_offset[0], p1[1] + box_offset[1])
                g_p2 = (p2[0] + box_offset[0], p2[1] + box_offset[1])
                cv2.line(debug_img, g_p1, g_p2, (0, 0, 255), 2)
                cv2.circle(debug_img, g_p2, 2, (0, 0, 255), -1)

    return (total_px_length or h) / pixels_per_cm

def get_length_rotated_rect(img_crop, pixels_per_cm, debug_img=None, box_offset=(0,0)):
    """
    Calculates plant length using a rotated bounding box (cv2.minAreaRect).
    """
    if img_crop is None or img_crop.size == 0 or not pixels_per_cm:
        return 0.0

    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([CONFIG["H_MIN"], CONFIG["S_MIN"], CONFIG["V_MIN"]])
    upper = np.array([CONFIG["H_MAX"], CONFIG["S_MAX"], CONFIG["V_MAX"]])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    length_px = max(rect[1])
    
    if debug_img is not None:
        box = np.int0(cv2.boxPoints(rect))
        box[:, 0] += box_offset[0]
        box[:, 1] += box_offset[1]
        cv2.drawContours(debug_img, [box], 0, (255, 0, 0), 2)

    return length_px / pixels_per_cm

ALGORITHMS = {"spine": get_length_spine_slicing, "rotated_rect": get_length_rotated_rect}

# -------------------- Video I/O Classes --------------------

class CaptureManager:
    def __init__(self, url, mode="stream"):
        self.url, self.mode = url, mode
        self.lock, self.frame, self.running = threading.Lock(), None, True
        self.cap, self.thread = None, threading.Thread(target=self._loop, daemon=True)
        self._open()
        self.thread.start()

    def _open(self):
        if self.cap: self.cap.release()
        print(f"Connecting to {self.url}...")
        try:
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"Error opening video source: {e}")
            self.cap = None

    def _loop(self):
        while self.running:
            if self.mode == "snapshot":
                try:
                    resp = requests.get(self.url, timeout=2.0)
                    resp.raise_for_status()
                    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                    with self.lock: self.frame = img
                except Exception as e:
                    print(f"Snapshot Error: {e}")
                    time.sleep(1.0)
            else:
                if not (self.cap and self.cap.isOpened()):
                    time.sleep(1.0); self._open(); continue
                ret, img = self.cap.read()
                with self.lock: self.frame = img if ret else None
                if not ret: print("Stream read failed. Reconnecting..."); time.sleep(0.5); self._open()
    
    def get(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join()
        if self.cap: self.cap.release()

class InferenceWorker:
    def __init__(self, weights, conf_thr):
        self.model = YOLO(weights)
        self.conf = conf_thr
        self.input_q = queue.Queue(maxsize=1)
        self.output_q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                frame = self.input_q.get(timeout=0.2)
                if frame is None: continue
                results = self.model.predict(frame, conf=self.conf, verbose=False, imgsz=CONFIG["IMG_SIZE"])
                if not self.output_q.empty():
                    try: self.output_q.get_nowait()
                    except queue.Empty: pass
                self.output_q.put(results)
            except queue.Empty: continue

    def predict_async(self, frame):
        if not self.input_q.full(): self.input_q.put(frame)

    def get_latest(self):
        try: return self.output_q.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join()

# -------------------- Main Application Logic --------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ensure_dir(os.path.join(script_dir, '..', CONFIG['OUT_DIR']))
    weights_path = os.path.join(script_dir, '..', CONFIG['YOLO_WEIGHTS'])
    csv_path = os.path.join(script_dir, '..', CONFIG['CSV_FILE'])
    
    cam = CaptureManager(CONFIG['URL'], CONFIG['MODE'])
    worker = InferenceWorker(weights_path, CONFIG['YOLO_CONF'])
    tracker = Sort(max_age=CONFIG['TRACK_BUFFER'], min_hits=3, iou_threshold=0.3)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CONFIG['ARUCO_DICT'].upper()))
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    measurement_func = ALGORITHMS[CONFIG['ALGORITHM']]

    pxcm_history, pixels_per_cm, last_aruco_x = deque(maxlen=30), None, None
    last_save_time = time.time()

    win_name = f"Plant Monitor (Algo: {CONFIG['ALGORITHM']}, Row: {CONFIG['ROW_DETECTION_METHOD']})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"Running. Algo: '{CONFIG['ALGORITHM']}', Row Detection: '{CONFIG['ROW_DETECTION_METHOD']}'. Press 'q' to quit.")

    while True:
        frame = cam.get()
        if frame is None: time.sleep(0.01); continue
        vis_img, h, w = frame.copy(), frame.shape[0], frame.shape[1]

        worker.predict_async(frame)
        yolo_result = worker.get_latest()
        
        corners, ids, _ = aruco_detector.detectMarkers(frame)
        if ids is not None:
            areas = [cv2.contourArea(c[0]) for c in corners]
            marker_corners = corners[np.argmax(areas)][0]
            width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
            if width_px > 0:
                pxcm_history.append(width_px / CONFIG['MARKER_SIZE_CM'])
                pixels_per_cm = np.mean(pxcm_history)
            last_aruco_x = np.mean(marker_corners[:, 0])
            cv2.polylines(vis_img, [marker_corners.astype(int)], True, (0, 255, 255), 2)
        elif last_aruco_x:
            draw_text(vis_img, "(Using Last Marker Pos)", (10, 80), (0, 255, 255))
        
        raw_detections = []
        if yolo_result and yolo_result[0].boxes:
            for box in yolo_result[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_detections.append({'box': (x1, y1, x2, y2), 'conf': box.conf[0], 'row_id': -1})

        # --- Row Assignment ---
        if CONFIG['ROW_DETECTION_METHOD'] == 'pipe':
            pipe_rows = detect_pipes(frame)
            detections = assign_rows_pipe_based(pipe_rows, raw_detections)
            row_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
            for i, pr in enumerate(pipe_rows):
                color = row_colors[i % len(row_colors)]
                cv2.line(vis_img, (0, int(pr['center_y'])), (w, int(pr['center_y'])), color, 1)
        else: # 'aruco' method
            detections = []
            if last_aruco_x and raw_detections:
                plant_centers_x = [(d['box'][0] + d['box'][2]) / 2 for d in raw_detections]
                closest_plant_idx = np.argmin(np.abs(np.array(plant_centers_x) - last_aruco_x))
                is_right_side = plant_centers_x[closest_plant_idx] > last_aruco_x
                
                for i, d in enumerate(raw_detections):
                    is_on_side = ((d['box'][0] + d['box'][2]) / 2 > last_aruco_x) if is_right_side else ((d['box'][0] + d['box'][2]) / 2 < last_aruco_x)
                    if is_on_side:
                        detections.append(d)
                cv2.line(vis_img, (int(last_aruco_x), 0), (int(last_aruco_x), h), (255, 255, 0), 1)

        # Convert to format required by SORT: [x1, y1, x2, y2, score]
        sort_detections = np.array([[*d['box'], d['conf']] for d in detections])
        tracked_objs = tracker.update(sort_detections) if len(sort_detections) > 0 else np.empty((0, 5))
        
        log_data = []
        for obj in tracked_objs:
            x1, y1, x2, y2, track_id = map(int, obj)
            
            # Match tracked object back to its detection to get row_id
            matched_det = min(detections, key=lambda d: np.linalg.norm(np.array([(d['box'][0]+d['box'][2])/2, (d['box'][1]+d['box'][3])/2]) - np.array([(x1+x2)/2, (y1+y2)/2])))
            row_id = matched_det.get('row_id', -1)

            length_cm = 0.0
            if pixels_per_cm and (y2 > y1 and x2 > x1):
                length_cm = measurement_func(frame[y1:y2, x1:x2], pixels_per_cm, vis_img, (x1, y1))
            
            color = row_colors[row_id % len(row_colors)] if row_id != -1 else (0, 255, 0)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label = f"#{track_id} L:{length_cm:.1f}cm"
            if row_id != -1: label = f"R{row_id} {label}"
            draw_text(vis_img, label, (x1, y1 - 10), color)
            log_data.append({"id": track_id, "row": row_id, "length": length_cm})

        if pixels_per_cm:
            draw_text(vis_img, f"Scale: {pixels_per_cm:.2f} px/cm", (10, 30), (0, 255, 0))
        else:
            draw_text(vis_img, "SEARCHING FOR ARUCO MARKER...", (10, 30), (0, 0, 255))

        if time.time() - last_save_time > CONFIG['SAVE_INTERVAL'] and log_data:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for data in log_data:
                    writer.writerow([ts, data['id'], data['row'], f"{data['length']:.2f}"])
            
            img_path = os.path.join(script_dir, '..', CONFIG['OUT_DIR'], f"log_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(img_path, vis_img)
            draw_text(vis_img, "LOG SAVED", (w - 150, 30), (0, 0, 255))
            last_save_time = time.time()

        cv2.imshow(win_name, vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.stop(); worker.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Height Monitor")
    parser.add_argument("--url", type=str, help=f"Video stream URL (default: {CONFIG['URL']})")
    parser.add_argument("--weights", type=str, help=f"Path to YOLO weights (default: {CONFIG['YOLO_WEIGHTS']})")
    parser.add_argument("--conf", type=float, help=f"YOLO confidence threshold (default: {CONFIG['YOLO_CONF']})")
    parser.add_argument("--algo", type=str, choices=ALGORITHMS.keys(), help=f"Measurement algorithm (default: {CONFIG['ALGORITHM']})")
    parser.add_argument("--row-detection", type=str, choices=['pipe', 'aruco'], help=f"Row detection method (default: {CONFIG['ROW_DETECTION_METHOD']})")
    parser.add_argument("--marker-size", type=float, help=f"ArUco marker size in CM (default: {CONFIG['MARKER_SIZE_CM']})")

    args = parser.parse_args()
    if args.url: CONFIG['URL'] = args.url
    if args.weights: CONFIG['YOLO_WEIGHTS'] = args.weights
    if args.conf: CONFIG['YOLO_CONF'] = args.conf
    if args.algo: CONFIG['ALGORITHM'] = args.algo
    if args.row_detection: CONFIG['ROW_DETECTION_METHOD'] = args.row_detection
    if args.marker_size: CONFIG['MARKER_SIZE_CM'] = args.marker_size
    
    main()
