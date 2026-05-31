import os
import cv2

INPUT_BASE_DIR = os.path.join("data", "processed")
OUTPUT_BASE_DIR = os.path.join("data", "patches")
TARGET_SIZE = (224, 224)

print("Memulai ekstraksi Patch (Micro-RoI) ke 224x224...")

for split in ["train", "val", "test"]:
    input_split = os.path.join(INPUT_BASE_DIR, split)
    output_split = os.path.join(OUTPUT_BASE_DIR, split)

    if not os.path.exists(input_split):
        continue

    for class_name in os.listdir(input_split):
        class_dir = os.path.join(input_split, class_name)
        if not os.path.isdir(class_dir):
            continue

        out_class_dir = os.path.join(output_split, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape

            # Ekstraksi fisik Micro-RoI (85% bawah, 30% kiri)
            y_start = int(h * 0.85)
            x_end = int(w * 0.30)
            roi = img[y_start:h, 0:x_end]

            # Zoom-In dengan interpolasi kualitas tinggi
            patch_resized = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

            out_path = os.path.join(out_class_dir, img_name)
            cv2.imwrite(out_path, patch_resized)

print(f"Selesai! Patch tersimpan di: {OUTPUT_BASE_DIR}")
