import cv2
from ultralytics import YOLO
import time
from pathlib import Path
import csv

# ========= CONFIG =========
# Folder containing your stitched MP4 videos (360 or fixed cams)
VIDEO_FOLDER = Path(r"C:\max 2 jerms")   # <- change later if you want

# YOLOv8x weights (already downloaded)
MODEL_PATH = Path(r"C:\models\yolov8x.pt")
# ==========================


def process_video(model: YOLO, video_path: Path, output_folder: Path, log_folder: Path):
    print(f"\n=== Processing video: {video_path.name} ===")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    out_video_path = output_folder / f"{video_path.stem}_det.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    log_path = log_folder / f"{video_path.stem}_det.csv"
    log_file = open(log_path, mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        ["frame_idx", "time_sec", "class_id", "class_name",
         "confidence", "x1", "y1", "x2", "y2"]
    )

    frame_idx = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_idx += 1
        time_sec = frame_idx / fps

        results = model(frame, verbose=False, device="cpu")[0]
        annotated = results.plot()

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = model.names.get(cls_id, str(cls_id))
                csv_writer.writerow(
                    [frame_idx, f"{time_sec:.3f}", cls_id, class_name,
                     f"{conf:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"]
                )

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        live_fps = 1.0 / dt if dt > 0 else 0.0

        cv2.putText(
            annotated,
            f"{video_path.name} | YOLOv8x ~ {live_fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLOv8x - Multi-video processing", annotated)
        writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("User pressed 'q' - skipping rest of this video.")
            break

    cap.release()
    writer.release()
    log_file.close()
    print(f"Saved annotated video to: {out_video_path}")
    print(f"Saved detection log to:   {log_path}")


def main():
    if not VIDEO_FOLDER.exists():
        print(f"Video folder not found: {VIDEO_FOLDER}")
        return

    videos = sorted(
        [p for p in VIDEO_FOLDER.iterdir() if p.suffix.lower() == ".mp4"]
    )

    if not videos:
        print(f"No .mp4 files found in folder: {VIDEO_FOLDER}")
        return

    print("Found the following videos:")
    for v in videos:
        print("  -", v.name)

    output_folder = VIDEO_FOLDER / "annotated"
    log_folder = VIDEO_FOLDER / "logs"
    output_folder.mkdir(exist_ok=True)
    log_folder.mkdir(exist_ok=True)

    print("\nLoading YOLOv8X model on CPU...")
    model = YOLO(str(MODEL_PATH))
    model.to("cpu")

    for video_path in videos:
        process_video(model, video_path, output_folder, log_folder)

    cv2.destroyAllWindows()
    print("\nAll videos processed.")


if __name__ == "__main__":
    main()