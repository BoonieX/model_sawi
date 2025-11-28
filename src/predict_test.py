from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
results = model.predict(
    source="../tambahdataset/unlabeled/IMG-20251113-WA0002.jpg",
    conf=0.59,
    iou=0.5,
    save=True,          # save annotated images
    save_txt=True,      # save YOLO txt labels
    save_conf=True      # include confidences in txt
)
