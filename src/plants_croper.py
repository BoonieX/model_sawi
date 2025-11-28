import cv2, os, glob
from ultralytics import YOLO


model = YOLO("runs/detect/train2/weights/best.pt")
os.makedirs("sawi_crops", exist_ok=True)

img_foler = "datasets/plants/images/train"
paths = glob.glob(os.path.join(img_foler, "*.jpg"))

for img_path in paths:
    img = cv2.imread(img_path)
    res = model(img, conf=0.6, verbose=False)[0]
    for j, box in enumerate(res.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"sawi_crops/{os.path.basename(img_path)}_{j}.jpg", crop)

