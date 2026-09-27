"""
UIEB Dataset Live Stream & Evaluation Runner
Demonstrates the complete sensor fusion pipeline on actual real-world photos
from the downloaded larjeck/uieb-dataset-raw dataset.
Generates side-by-side comparisons of:
  [Raw Murky Underwater Photo] vs [Enhanced Telemetry + Sonar + Detections]
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from deploy.underwater_pi_fusion import SensorFusionPipeline
from utils.image_enhancement import enhance_underwater_frame


def run_uieb_demonstration(num_frames: int = 5, save_comparison: bool = True):
    print("\n=======================================================")
    print(" Running Sensor Fusion on Downloaded UIEB Raw Dataset")
    print(f" Source: {config.UIEB_RAW_DIR}")
    print("=======================================================\n")

    raw_images = sorted(list(config.UIEB_RAW_DIR.glob("*.png")) + list(config.UIEB_RAW_DIR.glob("*.jpg")))
    if not raw_images:
        print(f"[-] No UIEB raw images found in {config.UIEB_RAW_DIR}")
        return False

    print(f"[+] Loaded {len(raw_images)} real underwater photos from UIEB dataset.")

    # Initialize sensor fusion pipeline with source="uieb"
    pipeline = SensorFusionPipeline(camera_source="uieb", headless=True)

    out_dir = config.BASE_DIR / "deploy" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process first few real underwater images from UIEB
    saved_snapshots = []
    for i in range(min(num_frames, len(raw_images))):
        img_path = raw_images[i * 12]  # Sample diverse images
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        h, w = 480, 640
        raw_resized = cv2.resize(raw_bgr, (w, h))

        # 1. Enhance murky underwater visibility
        enhanced = enhance_underwater_frame(raw_resized)

        # 2. Measure ultrasonic sonar distance (simulated or hardware)
        dist = pipeline.sonar.get_distance()

        # 3. Detect objects / obstacles
        detections = pipeline.detector.predict(enhanced)

        # 4. Render AR HUD
        telemetry_frame = pipeline.hud.draw(enhanced, detections, dist)

        # 5. Create side-by-side comparison: [Raw Image] | [Enhanced + Sonar Telemetry]
        divider = np.full((h, 4, 3), (255, 255, 255), dtype=np.uint8)
        
        # Label on raw side
        raw_labeled = raw_resized.copy()
        cv2.putText(raw_labeled, f"RAW UIEB: {img_path.name}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        comparison = np.hstack([raw_labeled, divider, telemetry_frame])
        
        frame_path = out_dir / f"uieb_frame_{i:02d}.jpg"
        cv2.imwrite(str(frame_path), telemetry_frame)

        if i == 0:
            comp_path = out_dir / "uieb_side_by_side_comparison.jpg"
            cv2.imwrite(str(comp_path), comparison)
            print(f"[+] Saved side-by-side comparison to: {comp_path}")

        print(f"[{i+1}/{num_frames}] Processed {img_path.name} | Sonar: {dist:5.1f} cm | Detections: {len(detections)}")
        saved_snapshots.append(frame_path)

    pipeline.stop()
    print(f"\n[+] Successfully tested pipeline on real UIEB raw photos! Output saved in: {out_dir}")
    return True


if __name__ == "__main__":
    run_uieb_demonstration(num_frames=4, save_comparison=True)
