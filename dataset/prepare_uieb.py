"""
UIEB Dataset Preprocessor & Auto-Annotator for YOLO
Takes the raw 890 underwater photos from larjeck/uieb-dataset-raw,
detects underwater features/objects, formats into YOLO annotations,
and structures into train/val sets for Raspberry Pi training/testing.
"""
import sys
import random
import shutil
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def auto_annotate_and_split(
    raw_dir: Path = config.UIEB_RAW_DIR,
    output_dir: Path = config.UIEB_DATA_DIR,
    max_images: int = 200,  # Process a solid batch (or all 890)
    train_ratio: float = 0.8
):
    print(f"\n=======================================================")
    print(f" Processing Real-World UIEB Raw Dataset")
    print(f" Source: {raw_dir}")
    print(f" Output Target: {output_dir}")
    print(f"=======================================================\n")

    if not raw_dir.exists():
        print(f"[-] UIEB raw directory not found: {raw_dir}")
        return False

    image_paths = sorted(list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg")))
    print(f"[+] Found {len(image_paths)} real underwater photos in UIEB raw dataset.")

    if len(image_paths) == 0:
        print("[-] No images found.")
        return False

    # Shuffle for uniform split
    random.seed(42)
    random.shuffle(image_paths)

    selected_images = image_paths[:max_images] if max_images else image_paths
    print(f"[+] Auto-annotating and formatting {len(selected_images)} images into YOLO structure...")

    # Load ONNX detector if available for auto-labeling
    use_onnx = False
    session = None
    if config.MODEL_ONNX.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(config.MODEL_ONNX), providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            use_onnx = True
            print(f"[+] Loaded ONNX model for high-precision auto-annotation.")
        except Exception as e:
            print(f"[!] ONNX loader notice: {e}")

    # Create destination directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = int(len(selected_images) * train_ratio)

    for idx, img_path in enumerate(selected_images):
        split = "train" if idx < n_train else "val"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape
        labels = []

        if use_onnx and session is not None:
            # Run ONNX inference for auto-annotation
            img_sz = 320
            resized = cv2.resize(img, (img_sz, img_sz))
            blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
            outputs = session.run(None, {input_name: blob})
            raw_out = outputs[0][0]
            if raw_out.shape[0] < raw_out.shape[1]:
                raw_out = raw_out.T

            for row in raw_out:
                cx, cy, bw, bh = row[:4]
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                score = float(scores[cls_id])
                if score >= 0.25:
                    # Normalize coords
                    norm_cx = cx / img_sz
                    norm_cy = cy / img_sz
                    norm_bw = bw / img_sz
                    norm_bh = bh / img_sz
                    # Map to class range [0..5]
                    target_cls = min(cls_id, 5)
                    labels.append(f"{target_cls} {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")

        # If model produced no detections or without ONNX, use adaptive salient region detection
        if len(labels) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 25, 4)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if (h * w * 0.005) < area < (h * w * 0.4):
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    norm_cx = (bx + bw / 2) / w
                    norm_cy = (by + bh / 2) / h
                    norm_bw = bw / w
                    norm_bh = bh / h
                    cls_id = 0 if (bw / float(bh)) > 1.2 else 2  # 0: fish, 2: crab/obstacle
                    labels.append(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                    if len(labels) >= 4:
                        break

        # Save image and label
        dest_img_path = output_dir / "images" / split / img_path.name
        dest_lbl_path = output_dir / "labels" / split / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dest_img_path)
        dest_lbl_path.write_text("\n".join(labels))

    print(f"\n[+] Successfully formatted UIEB Raw Dataset at: {output_dir}")
    print(f"    Train images: {n_train}")
    print(f"    Validation images: {len(selected_images) - n_train}")
    return True


if __name__ == "__main__":
    auto_annotate_and_split(max_images=200)
