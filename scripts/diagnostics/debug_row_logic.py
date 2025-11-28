#!/usr/bin/env python3
import cv2
import numpy as np
import os
from ultralytics import YOLO

# -------------------- CONFIGURATION --------------------
CONFIG = {
    # CHANGE THIS TO YOUR IMAGE PATH
    "INPUT_IMAGE": "test_image/sawi_2.jpg", 
    "OUT_DIR": "debug_output",
    
    # USE YOUR WEIGHTS
    "YOLO_WEIGHTS": "./runs/detect/train2/weights/best.pt", 
    
    # 1. LOWER CONFIDENCE to ensure YOLO actually sees the plant
    "YOLO_CONF": 0.59,
    
    # 2. INCREASE TOLERANCE (The likely fix)
    # This defines how far UP or DOWN the plant center can be from the ArUco
    "ROW_ALIGNMENT_TOLERANCE": 150, 
    
    "ARUCO_DICT": cv2.aruco.DICT_4X4_50,
}

def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

def main():
    ensure_dir(CONFIG['OUT_DIR'])
    
    print(f"Diagnosing {CONFIG['INPUT_IMAGE']}...")
    frame = cv2.imread(CONFIG['INPUT_IMAGE'])
    if frame is None: return print("Image not found.")
    vis = frame.copy()
    h, w = frame.shape[:2]

    # 1. Detect ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(CONFIG['ARUCO_DICT'])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        print("CRITICAL FAIL: No ArUco detected.")
        return

    # Get ArUco Data
    idx = np.argmax([cv2.contourArea(c[0]) for c in corners])
    mk = corners[idx][0]
    ax1, ay1, ax2, ay2 = int(np.min(mk[:, 0])), int(np.min(mk[:, 1])), int(np.max(mk[:, 0])), int(np.max(mk[:, 1]))
    
    # The Reference Horizon (Center Y of ArUco)
    aruco_cy = (ay1 + ay2) // 2
    
    # Draw Reference Lines
    cv2.polylines(vis, [mk.astype(int)], True, (0, 255, 255), 3) # Yellow Marker
    # Draw the "Allowed Zone" lines (Cyan)
    y_top_limit = aruco_cy - CONFIG['ROW_ALIGNMENT_TOLERANCE']
    y_bottom_limit = aruco_cy + CONFIG['ROW_ALIGNMENT_TOLERANCE']
    cv2.line(vis, (0, y_top_limit), (w, y_top_limit), (255, 255, 0), 2)
    cv2.line(vis, (0, y_bottom_limit), (w, y_bottom_limit), (255, 255, 0), 2)
    cv2.putText(vis, "Allowed Zone Top", (10, y_top_limit - 10), 0, 0.7, (255, 255, 0), 2)

    # 2. Run YOLO
    model = YOLO(CONFIG['YOLO_WEIGHTS'])
    results = model.predict(frame, conf=CONFIG['YOLO_CONF'], verbose=False)
    
    detected_count = 0
    
    if results[0].boxes:
        for box in results[0].boxes:
            detected_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Plant Center
            plant_cy = (y1 + y2) // 2
            
            # DIAGNOSTIC LOGIC
            vertical_dist = abs(plant_cy - aruco_cy)
            is_aligned = vertical_dist < CONFIG['ROW_ALIGNMENT_TOLERANCE']
            
            if is_aligned:
                # SUCCESS CASE: GREEN BOX
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"OK! (Dist:{vertical_dist})"
                cv2.putText(vis, label, (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
            else:
                # FAILURE CASE: BLUE BOX (Raw Detection Rejected)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                reason = "TOO HIGH" if plant_cy < aruco_cy else "TOO LOW"
                label = f"{reason} (Dist:{vertical_dist} > {CONFIG['ROW_ALIGNMENT_TOLERANCE']})"
                
                # Draw text to explain why
                cv2.putText(vis, label, (x1, y1-10), 0, 0.5, (255, 0, 0), 2)
                # Draw line connecting plant center to ArUco center to visualize distance
                cv2.line(vis, ((x1+x2)//2, plant_cy), ((x1+x2)//2, aruco_cy), (0,0,255), 1)

    if detected_count == 0:
        cv2.putText(vis, "YOLO DETECTED NOTHING (Check Weights/Conf)", (50, 50), 0, 1, (0,0,255), 3)

    # Save
    out_path = os.path.join(CONFIG['OUT_DIR'], "diagnostic_result.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Saved diagnostic image to: {out_path}")
    print("Check the image to see BLUE boxes (Rejected) vs GREEN boxes (Accepted).")

if __name__ == "__main__":
    main()