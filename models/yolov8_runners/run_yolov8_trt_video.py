from ultralytics import YOLO
from pathlib import Path

# ========= CONFIG =========

# Path to your TensorRT engine
ENGINE_PATH = Path(r"C:\models\yolov8x_fp16.engine")

# Folder containing your stitched MP4 videos (360 or fixed)
VIDEO_FOLDER = Path(r"C:\max 2 jerms")  # <-- change if your videos are somewhere else

# Where to save TensorRT outputs (annotated videos, etc.)
OUT_ROOT = VIDEO_FOLDER / "trt_runs"

# ==========================


def main():
    if not ENGINE_PATH.exists():
        print(f"Engine file not found: {ENGINE_PATH}")
        return

    if not VIDEO_FOLDER.exists():
        print(f"Video folder not found: {VIDEO_FOLDER}")
        return

    # Find all MP4 files in the folder
    videos = sorted([p for p in VIDEO_FOLDER.iterdir() if p.suffix.lower() == ".mp4"])
    if not videos:
        print(f"No .mp4 files found in folder: {VIDEO_FOLDER}")
        return

    print("Found the following videos:")
    for v in videos:
        print("  -", v.name)

    OUT_ROOT.mkdir(exist_ok=True)

    print("\nLoading YOLOv8 TensorRT engine on GPU...")
    model = YOLO(str(ENGINE_PATH))  # this uses TensorRT backend

    # Force-disable PyTorch warmup and use GPU 0
    model.overrides['device'] = 'cuda:0'
    model.overrides['warmup'] = False
    model.overrides['batch'] = 1

    for v in videos:
        run_name = v.stem  # one subfolder per video
        print(f"\n=== Running TensorRT on {v.name} ===")

        # Ultralytics handles reading video, running TensorRT, and saving annotated video
        model.predict(
            source=str(v),
            device=0,               # GPU 0 (your 5090)
            save=True,              # save annotated video
            project=str(OUT_ROOT),  # root output folder
            name=run_name,          # subfolder for this video
            exist_ok=True,
            vid_stride=1,           # use every frame
            stream=False,
            warmup=False,           # 🔥 stop PyTorch GPU warmup
            verbose=False           # optional, keeps logs cleaner
        )

    print("\nAll videos processed with TensorRT.")
    print(f"Outputs saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()

   