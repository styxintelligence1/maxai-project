import cv2
import numpy as np
import tensorrt as trt
import cuda  # from the cuda-python package
from pathlib import Path
import time

# Try to get the CUDA Runtime API in a version-agnostic way
try:
    import cuda.cudart as cudart
except Exception:
    # Some versions expose everything via cuda.cuda instead
    from cuda import cuda as cudart


# ------------------------------------------------------------------
# CONFIG: adjust these if you want
# ------------------------------------------------------------------
ENGINE_PATH = r"C:\models\yolov8x_fp16.engine"
INPUT_FOLDER = r"C:\test_videos"
OUTPUT_FOLDER = r"C:\max 2 jerms\trt_runs_b"
CONF_THRESH = 0.35
IOU_THRESH = 0.5
CLASS_NAMES = None  # we can plug COCO80 names later if you like


# ------------------------------------------------------------------
# TensorRT helpers
# ------------------------------------------------------------------

class TrtYoloRunner:
    def __init__(self, engine_path: str, device_id: int = 0):
        self.engine_path = engine_path
        self.device_id = device_id

        print(f"[TRT] Loading engine: {engine_path}")
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        cudart.cudaSetDevice(device_id)

        # We expect bindings: images (input), output0 (output)
        self.input_idx = self.engine.get_binding_index("images")
        self.output_idx = self.engine.get_binding_index("output0")

        input_shape = self.engine.get_binding_shape(self.input_idx)
        # If dynamic, set concrete shape
        if -1 in input_shape:
            input_shape = (1, 3, 640, 640)
            self.context.set_binding_shape(self.input_idx, input_shape)

        self.input_shape = tuple(input_shape)  # NCHW
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))

        print(f"[TRT] Input  shape: {self.input_shape}")
        print(f"[TRT] Output shape: {self.output_shape}")

        # Allocate device buffers
        self.input_nbytes = int(np.prod(self.input_shape) * np.dtype(np.float16).itemsize)
        self.output_nbytes = int(np.prod(self.output_shape) * np.dtype(np.float16).itemsize)

        _, self.d_input = cudart.cudaMalloc(self.input_nbytes)
        _, self.d_output = cudart.cudaMalloc(self.output_nbytes)

        # Host buffers (page-locked for speed)
        _, self.h_input = cudart.cudaHostAlloc(self.input_nbytes, cudart.cudaHostAllocDefault)
        _, self.h_output = cudart.cudaHostAlloc(self.output_nbytes, cudart.cudaHostAllocDefault)

        # Wrap host buffers with numpy
        self.h_input_np = np.frombuffer(
            (ctypes.c_ubyte * self.input_nbytes).from_address(self.h_input),
            dtype=np.float16,
        )
        self.h_output_np = np.frombuffer(
            (ctypes.c_ubyte * self.output_nbytes).from_address(self.h_output),
            dtype=np.float16,
        )

        # CUDA stream
        _, self.stream = cudart.cudaStreamCreate()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        input_tensor: float16 numpy array with shape self.input_shape
        returns: float16 numpy array with shape self.output_shape
        """
        assert input_tensor.shape == self.input_shape
        assert input_tensor.dtype == np.float16

        # copy input into pinned host buffer
        np.copyto(self.h_input_np.view(np.float16).reshape(self.input_shape), input_tensor)

        # H2D
        cudart.cudaMemcpyAsync(
            self.d_input,
            self.h_input,
            self.input_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

        # execute
        bindings = [0] * self.engine.num_io_tensors
        bindings[self.input_idx] = int(self.d_input)
        bindings[self.output_idx] = int(self.d_output)

        self.context.execute_async_v2(bindings, self.stream)

        # D2H
        cudart.cudaMemcpyAsync(
            self.h_output,
            self.d_output,
            self.output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)

        return self.h_output_np.view(np.float16).reshape(self.output_shape)


# Need ctypes after class definition
import ctypes  # noqa: E402


# ------------------------------------------------------------------
# YOLOv8-ish postprocessing
# ------------------------------------------------------------------

def xywh_to_xyxy(boxes):
    # boxes: (N,4) in cx,cy,w,h (image coords)
    x_c, y_c, w, h = np.split(boxes, 4, axis=1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.concatenate([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, iou_thresh):
    # boxes: (N,4) xyxy, scores: (N,)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thresh]
    return np.array(keep, dtype=np.int32)


def box_iou(box1, box2):
    # box1: (N,4), box2: (M,4)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # (N,M,2)
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # (N,M,2)

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / np.clip(union, 1e-6, None)


def decode_yolo_output(output, img_w, img_h, conf_thresh=0.35, iou_thresh=0.5):
    """
    output: (1, 84, 8400) float16 -> we'll treat as:
      0:4  = cx,cy,w,h
      4:   = 80 class scores (no explicit objectness)
    """
    out = output.astype(np.float32)[0]  # (84, 8400)
    num_preds = out.shape[1]

    boxes = out[0:4, :].T  # (8400,4)
    class_scores = out[4:, :]  # (80, 8400)
    cls_ids = np.argmax(class_scores, axis=0)  # (8400,)
    cls_scores = class_scores[cls_ids, np.arange(num_preds)]

    mask = cls_scores >= conf_thresh
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    boxes = boxes[mask]
    cls_ids = cls_ids[mask]
    cls_scores = cls_scores[mask]

    # resize from 640x640 to original image size
    gain_w = img_w / 640.0
    gain_h = img_h / 640.0
    boxes[:, 0] *= gain_w
    boxes[:, 2] *= gain_w
    boxes[:, 1] *= gain_h
    boxes[:, 3] *= gain_h

    boxes_xyxy = xywh_to_xyxy(boxes)

    keep = nms(boxes_xyxy, cls_scores, iou_thresh)
    return boxes_xyxy[keep], cls_ids[keep], cls_scores[keep]


# ------------------------------------------------------------------
# Preproc / drawing
# ------------------------------------------------------------------

def preprocess_frame(frame_bgr, input_shape):
    """
    frame_bgr: HxWx3 uint8
    input_shape: (1,3,640,640)
    """
    _, _, in_h, in_w = input_shape
    h, w, _ = frame_bgr.shape

    # letterbox to 640x640 (simpler: just resize for now)
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    chw = chw[np.newaxis, ...]  # add batch dim
    return chw.astype(np.float16)


def draw_detections(frame_bgr, boxes, cls_ids, scores):
    for box, cid, score in zip(boxes, cls_ids, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if CLASS_NAMES is not None and 0 <= cid < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[cid]} {score:.2f}"
        else:
            label = f"id{cid} {score:.2f}"

        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame_bgr, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame_bgr


# ------------------------------------------------------------------
# Main video loop
# ------------------------------------------------------------------

def process_video(trt_runner: TrtYoloRunner, in_path: Path, out_path: Path):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {in_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    infer_time_total = 0.0
    frames_counted = 0

    print(f"\n=== Processing {in_path.name} ({w}x{h} @ {fps:.1f} FPS) ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Preprocess
        inp = preprocess_frame(frame, trt_runner.input_shape)

        # Inference
        t_infer_start = time.time()
        out = trt_runner.infer(inp)
        infer_time_total += (time.time() - t_infer_start)
        frames_counted += 1

        # Postprocess
        boxes, cls_ids, scores = decode_yolo_output(out, w, h, CONF_THRESH, IOU_THRESH)

        # Draw
        frame = draw_detections(frame, boxes, cls_ids, scores)
        writer.write(frame)

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Frame {frame_idx:5d} | elapsed {elapsed:6.1f}s | "
                f"avg infer FPS ~ {frames_counted / infer_time_total:6.1f}"
            )

    cap.release()
    writer.release()

    total_elapsed = time.time() - t0
    if infer_time_total > 0:
        avg_infer_fps = frames_counted / infer_time_total
    else:
        avg_infer_fps = 0.0

    print(
        f"=== Done {in_path.name} -> {out_path.name} | "
        f"video time {total_elapsed:.1f}s | "
        f"avg infer FPS ~ {avg_infer_fps:.1f} ==="
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default=ENGINE_PATH)
    parser.add_argument("--input_folder", type=str, default=INPUT_FOLDER)
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    in_dir = Path(args.input_folder)
    out_root = Path(args.output_folder)

    print(f"[SETUP] Engine: {args.engine}")
    print(f"[SETUP] Input folder:  {in_dir}")
    print(f"[SETUP] Output folder: {out_root}")

    trt_runner = TrtYoloRunner(args.engine)

    videos = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".mp4"])
    if not videos:
        print(f"[WARN] No .mp4 files found in {in_dir}")
        return

    print("\nFound videos:")
    for v in videos:
        print("  -", v.name)

    for v in videos:
        out_path = out_root / (v.stem + "_trt.mp4")
        process_video(trt_runner, v, out_path)

    print("\nAll videos processed with TensorRT on GPU.")


if __name__ == "__main__":
    main()