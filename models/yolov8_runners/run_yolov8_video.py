import cv2
from ultralytics import YOLO
import time
from pathlib import Path

# === CONFIG ===
# CHANGE this to match your actual video path if needed:
VIDEO_PATH = Path(r"C:\max 2 jerms\GS010011.mp4")
MODEL_PATH = Path(r"C:\models\yolov8x.pt")  # already downloaded earlier

def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    print("Loading YOLOv8x model...")
    model = YOLO(str(MODEL_PATH))

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS (from file): {fps}")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error.")
            break

        frame_count += 1

        # Run YOLO inference (model handles resize/preprocess)
        results = model(frame, verbose=False)[0]

        # Draw detections on frame
        annotated = results.plot()

        # Calculate simple live FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        live_fps = 1.0 / dt if dt > 0 else 0.0

        # Put FPS text on frame
        cv2.putText(
            annotated,
            f"YOLOv8x ~ {live_fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLOv8x - Shed detection", annotated)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User pressed 'q' - exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")

if __name__ == "__main__":
    main()