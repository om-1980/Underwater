"""
Desktop Simulation Launcher
Runs the complete Underwater Telemetry, Camera CLAHE, and Ultrasonic Sonar Fusion pipeline
with simulated sensors and camera feed, saving output snapshots for verification.
"""
import sys
import time
from pathlib import Path
import cv2

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from deploy.underwater_pi_fusion import SensorFusionPipeline


def run_desktop_test(duration_sec: int = 5, save_screenshot: bool = True):
    print("\n=======================================================")
    print(" Running Underwater Fusion System Simulation")
    print(f" Duration: {duration_sec} seconds")
    print("=======================================================\n")

    # Run in headless mode so it works cleanly in background/automation
    pipeline = SensorFusionPipeline(headless=True)
    
    # Start the pipeline in background thread
    import threading
    pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipeline_thread.start()

    start_time = time.time()
    last_frame = None

    while time.time() - start_time < duration_sec:
        frame = pipeline.camera.read()
        if frame is not None:
            dist = pipeline.current_distance_cm
            detections = pipeline.detector.predict(frame)
            annotated = pipeline.hud.draw(frame, detections, dist)
            last_frame = annotated
        time.sleep(0.1)

    # Save snapshot
    if save_screenshot and last_frame is not None:
        out_dir = config.BASE_DIR / "deploy" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = out_dir / "simulation_snapshot.jpg"
        cv2.imwrite(str(snapshot_path), last_frame)
        print(f"\n[+] Snapshot saved successfully to: {snapshot_path}")

    pipeline.stop()
    print("[+] Desktop simulation completed successfully.")


if __name__ == "__main__":
    run_desktop_test(duration_sec=3, save_screenshot=True)
