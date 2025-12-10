import cv2
from pathlib import Path
import time
from ultralytics import YOLO

# ========= CONFIG =========

# Folder where your imported videos live (from process_sd_card.py)
VIDEO_FOLDER = Path(r"C:\max 2 jerms")

# YOLO model weights (CPU)
MODEL_PATH = Path(r"C:\models\yolov8x.pt")  # we already downloaded this earlier

CONF_THRESH = 0.3  # minimum confidence to draw box
IOU_THRESH = 0.45  # NMS iou (Ultralytics handles this internally)

# ==========================


def process_video(model: YOLO, video_path: Path, output_folder: Path):
    """Run YOLO on one video and save annotated MP4."""
    print(f"\n=== Processing video: {video_path.name} ===")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_folder / f"{video_path.stem}_yolo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Ultralytics does preprocessing, NMS etc. internally
        results = model.predict(
            source=frame,
            verbose=False,
            conf=CONF_THRESH,
            iou=IOU_THRESH
        )

        # Draw detections on the frame
        annotated = results[0].plot()  # returns numpy array with boxes drawn
        out.write(annotated)

        if frame_idx % 50 == 0:
            dt = time.time() - t0
            fps_proc = frame_idx / dt if dt > 0 else 0
            print(f"  Processed {frame_idx} frames (~{fps_proc:.1f} FPS)")

    cap.release()
    out.release()

    dt = time.time() - t0
    print(f"Finished {video_path.name}: {frame_idx} frames in {dt:.1f}s "
          f"({frame_idx/dt:.1f} FPS). Output: {out_path}")


def main():
    if not VIDEO_FOLDER.exists():
        print("Video folder not found:", VIDEO_FOLDER)
        return

    videos = sorted(
        [p for p in VIDEO_FOLDER.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".360")]
    )
    if not videos:
        print("No video files found in:", VIDEO_FOLDER)
        return

    print("Found the following videos:")
    for v in videos:
        print(" -", v.name)

    # Output folder next to video folder
    output_folder = VIDEO_FOLDER / "annotated_cpu"
    output_folder.mkdir(exist_ok=True)
    print("\nAnnotated videos will be saved in:", output_folder)

    print("\nLoading YOLOv8x model on CPU...")
    model = YOLO(str(MODEL_PATH))
    model.to("cpu")

    for v in videos:
        process_video(model, v, output_folder)

    print("\nAll videos processed on CPU.")


if __name__ == "__main__":
    main()