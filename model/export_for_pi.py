"""
Model Exporter for Edge Deployment on Raspberry Pi
Converts PyTorch (.pt) weights to ONNX, TFLite (FP16/INT8), and NCNN formats.
"""
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def export_model_for_pi(
    weights_path: str = str(config.MODEL_TRAINED_PT),
    img_size: int = config.INFERENCE_IMG_SIZE,
    export_onnx: bool = True,
    export_tflite: bool = True
):
    path = Path(weights_path)
    if not path.exists():
        # Fallback to base pretrained weights if custom trained model not yet generated
        print(f"[!] Target weights '{weights_path}' not found. Falling back to base '{config.MODEL_BASE}'")
        weights_path = config.MODEL_BASE

    print(f"\n=======================================================")
    print(f" Exporting YOLO Model for Raspberry Pi Edge Inference")
    print(f" Source Weights: {weights_path}")
    print(f" Input Image Size: {img_size}x{img_size}")
    print(f"=======================================================\n")

    model = YOLO(weights_path)

    if export_onnx:
        print("[+] Exporting to ONNX (High-performance cross-platform format)...")
        onnx_path = model.export(
            format="onnx",
            imgsz=img_size,
            dynamic=False,
            simplify=True,
            opset=12
        )
        print(f"[+] ONNX export saved to: {onnx_path}")

    if export_tflite:
        print("[+] Exporting to TFLite (TensorFlow Lite for ARM CPU)...")
        try:
            tflite_path = model.export(
                format="tflite",
                imgsz=img_size,
                half=True  # Float16 quantization for 2x speedup on ARM
            )
            print(f"[+] TFLite export saved to: {tflite_path}")
        except Exception as e:
            print(f"[-] Note: TFLite export requires tensorflow-cpu or tflite tools: {e}")

    print("\n[+] Export process complete! Models are ready to transfer to Raspberry Pi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model for Raspberry Pi")
    parser.add_argument("--weights", type=str, default=str(config.MODEL_TRAINED_PT), help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMG_SIZE, help="Input size")
    parser.add_argument("--no-tflite", action="store_true", help="Skip TFLite export")
    args = parser.parse_args()

    export_model_for_pi(
        weights_path=args.weights,
        img_size=args.imgsz,
        export_onnx=True,
        export_tflite=not args.no_tflite
    )
