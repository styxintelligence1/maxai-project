from pathlib import Path
from ultralytics import YOLO


# ------------- SETTINGS ------------- #

# Folder with your short test videos
VIDEO_FOLDER = Path(r"C:\test_videos")

# Where to save annotated videos
OUT_ROOT = Path(r"C:\max 2 jerms\annotated_gpu")

# Path to your YOLOv8 model (.pt)
MODEL_PATH = Path(r"C:\models\yolov8x.pt")

# How many frames to skip (1 = every frame, 2 = every 2nd frame, etc.)
VID_STRIDE = 1  # you can change to 2 later for more speed


# ------------- MAIN LOGIC ------------- #

def main():
    # Find videos
    videos = sorted([p for p in VIDEO_FOLDER.glob("*.mp4") if p.is_file()])
    if not videos:
        print(f"No .mp4 files found in folder: {VIDEO_FOLDER}")
        return

    print("Found the following videos:")
    for v in videos:
        print("  -", v.name)

    # Make sure output folder exists
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model (PyTorch) – this will automatically use the GPU when device=0
    print("\nLoading YOLOv8 model on GPU...")
    model = YOLO(str(MODEL_PATH))

    print("\nStarting GPU processing...\n")

    for v in videos:
        run_name = v.stem + "_gpu"
        print(f"=== Processing {v.name} ===")

        # Ultralytics handles reading video, running on GPU, and saving annotated video
        model.predict(
            source=str(v),
            device=0,              # <-- THIS makes it use your RTX 5090
            save=True,
            project=str(OUT_ROOT),
            name=run_name,
            vid_stride=VID_STRIDE,
            stream=False,
            verbose=False,
        )

        print(f"Finished {v.name}, saved under: {OUT_ROOT / run_name}\n")

    print("All videos processed on GPU.")


if __name__ == "__main__":
    main()