import json, os, random, shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import math

# ====== KONFIGURASI ======
LABELME_DIR   = r"C:\Users\user\Documents\label2\caisim dataset"         # folder sumber JSON+gambar
OUT_ROOT      = r"C:\Users\user\Documents\label2\to_yolo"      # folder keluaran YOLO
VAL_RATIO     = 0.2                             # 20% untuk val
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# =========================

def load_classes(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        # fallback: deteksi kelas dari JSON (urut abjad)
        names = set()
        for j in Path(LABELME_DIR).glob("*.json"):
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sh in data.get("shapes", []):
                names.add(sh.get("label", "plant"))
        names = sorted(list(names))
        print("[INFO] classes.txt tidak ditemukan. Kelas terdeteksi:", names)
        return names
    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print("[INFO] classes:", names)
    return names

def polygon_to_bbox(points: List[List[float]]) -> Tuple[float,float,float,float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min

def rect_to_bbox(points: List[List[float]]) -> Tuple[float,float,float,float]:
    # Labelme rectangle: [x1,y1], [x2,y2]
    x1,y1 = points[0]
    x2,y2 = points[1]
    x_min, y_min = min(x1,x2), min(y1,y2)
    w, h = abs(x2-x1), abs(y2-y1)
    return x_min, y_min, w, h

def to_yolo_line(x, y, w, h, img_w, img_h, class_id) -> str:
    # convert to normalized (cx, cy, w, h)
    cx = x + w/2.0
    cy = y + h/2.0
    cx /= img_w; cy /= img_h
    w  /= img_w; h  /= img_h
    # clamp
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w  = min(max(w,  1e-6), 1.0)
    h  = min(max(h,  1e-6), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def ensure_dirs(root: Path):
    for p in [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)

def main():
    src = Path(LABELME_DIR)
    out = Path(OUT_ROOT)
    ensure_dirs(out)

    class_map = load_classes(src / "classes.txt")
    class_to_id = {name: i for i, name in enumerate(class_map)}

    # Kumpulkan pasangan (image, json)
    pairs = []
    for jpath in src.glob("*.json"):
        stem = jpath.stem
        # cari image dengan nama sama
        img_path = None
        for ext in IMAGE_EXTS:
            cand = src / f"{stem}{ext}"
            if cand.exists():
                img_path = cand; break
        if img_path is None:
            print(f"[SKIP] Gambar untuk {jpath.name} tidak ditemukan")
            continue
        pairs.append((img_path, jpath))

    # Shuffle dan split
    random.shuffle(pairs)
    n_val = math.floor(len(pairs) * VAL_RATIO)
    val_set = set(pairs[:n_val])

    for img_path, jpath in pairs:
        split = "val" if (img_path, jpath) in val_set else "train"
        # load image size
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # parse json
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        yolo_lines = []
        for sh in data.get("shapes", []):
            label = sh.get("label", "").strip()
            if label not in class_to_id:
                # kalau kelas tidak ada di classes.txt → otomatis tambahkan
                class_to_id[label] = len(class_to_id)
                class_map.append(label)
                print(f"[WARN] Kelas baru terdeteksi: {label}")

            cid = class_to_id[label]
            pts = sh.get("points", [])
            st  = sh.get("shape_type", "polygon")

            if st == "rectangle" and len(pts) >= 2:
                x, y, w, h = rect_to_bbox(pts[:2])
            else:
                # polygon/line/circle → turunkan ke bbox
                if len(pts) < 3:
                    continue
                x, y, w, h = polygon_to_bbox(pts)

            if w <= 1 or h <= 1:
                continue

            yolo_lines.append(to_yolo_line(x, y, w, h, img_w, img_h, cid))

        # salin image & tulis label
        dst_img = out / "images" / split / img_path.name
        dst_lbl = out / "labels" / split / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    # tulis ulang classes.txt (jaga konsistensi urutan id)
    with open(out / "classes.txt", "w", encoding="utf-8") as f:
        for n in class_map:
            f.write(n + "\n")

    # buat data.yaml
    yaml = f"""path: {out.as_posix()}
train: images/train
val: images/val
names:
"""
    for i, name in enumerate(class_map):
        yaml += f"  {i}: {name}\n"

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml)

    print("\n[SUKSES] Konversi selesai.")
    print(f"Dataset YOLO ada di: {out}")
    print(f"Train images : {(out/'images'/'train').glob('*')}")
    print(f"Val images   : {(out/'images'/'val').glob('*')}")
    print(f"Lihat data.yaml: {out/'data.yaml'}")

if __name__ == "__main__":
    main()
