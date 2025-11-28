from ultralytics import YOLO
import cv2

model = YOLO(r"./runs/detect/train2/weights/best.pt")
print("model.names:", model.names)

# Test on rack image
img = cv2.imread(r"test_image/sawi_2.jpg")  # adjust path if needed
results = model.predict(img, conf=0.3, show=True, verbose=True)

r = results[0]
print("boxes:", len(r.boxes))
if len(r.boxes):
    print("cls:", r.boxes.cls.cpu().numpy(), "conf:", r.boxes.conf.cpu().numpy())
