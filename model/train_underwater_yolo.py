"""
Training Pipeline for Underwater YOLOv8n / YOLO11n on UIEB Dataset
Optimized for Raspberry Pi edge deployment.
"""
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def train_model(
    model_name: str = config.MODEL_BASE,
    data_yaml: Path = config.BASE_DIR / "dataset" / "uieb.yaml",
    epochs: int = 50,
    img_size: int = config.INFERENCE_IMG_SIZE,
    batch_size: int = 16,
    device: str = "cpu"
):
    """
    Trains a lightweight YOLO model on the UIEB dataset.
    """
    print(f"\n=======================================================")
    print(f" Starting Underwater YOLO Training")
    print(f" Base Model: {model_name}")
    print(f" Dataset YAML: {data_yaml}")
    print(f" Input Resolution: {img_size}x{img_size}")
    print(f" Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    print(f"=======================================================\n")

    # Load pre-trained nano YOLO model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=2,
        project=str(config.MODELS_DIR.parent),
        name="uieb_run",
        exist_ok=True,
        # Underwater data augmentations:
        hsv_h=0.02,   # Slight hue shift (water color variations)
        hsv_s=0.5,    # Saturation variation
        hsv_v=0.4,    # Value/lightness variation
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation for small objects (shrimp, small fish)
    )

    # Save best model to configured location
    best_weights = Path(model.trainer.best)
    if best_weights.exists():
        import shutil
        target_path = config.MODEL_TRAINED_PT
        shutil.copy2(best_weights, target_path)
        print(f"\n[+] Successfully saved best model weights to: {target_path}")

        # Automatically export to ONNX for Raspberry Pi
        print("[+] Exporting model to ONNX for Raspberry Pi...")
        model.export(format="onnx", imgsz=img_size, dynamic=False, simplify=True)
        print(f"[+] ONNX export ready at {config.MODEL_ONNX}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Underwater YOLO on UIEB Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMG_SIZE, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt or yolo11n.pt)")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )
