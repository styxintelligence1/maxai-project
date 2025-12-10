import cv2
import numpy as np
from pathlib import Path
import time

import tensorrt as trt
import pycuda.autoinit  # noqa: F401, this sets up a CUDA context
import pycuda.driver as cuda


# ========= CONFIG =========

ENGINE_PATH = Path(r"C:\models\yolov8x_fp16.engine")
VIDEO_PATH = Path(r"C:\max 2 jerms\GS010011.mp4")  # <-- change to one test video
OUTPUT_PATH = VIDEO_PATH.with_name(VIDEO_PATH.stem + "_trt_det.mp4")

INPUT_SIZE = (640, 640)  # YOLOv8 default export size
CONF_THRESH = 0.3
IOU_THRESH = 0.45

# =========================


def load_engine(engine_path: Path):
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def preprocess(frame, input_size):
    """
    Resize + letterbox to 640x640, convert to CHW float32 in [0,1].
    """
    h, w = frame.shape[:2]
    new_w, new_h = input_size

    # letterbox
    scale = min(new_w / w, new_h / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    dw = (new_w - resized.shape[1]) // 2
    dh = (new_h - resized.shape[0]) // 2
    canvas[dh:dh + resized.shape[0], dw:dw + resized.shape[1], :] = resized

    # BGR -> RGB, HWC -> CHW, normalize
    img = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return img, scale, dw, dh, (h, w)


def nms(boxes, scores, iou_thresh):
    """
    Simple NMS for boxes in xyxy format.
    boxes: (N, 4), scores: (N,)
    """
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = (
            (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            - inter
        )
        iou = inter / (union + 1e-6)

        idxs = rest[iou <= iou_thresh]

    return keep


def main():
    if not ENGINE_PATH.exists():
        print(f"Engine not found: {ENGINE_PATH}")
        return
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    print(f"Loading engine: {ENGINE_PATH}")
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Assume one input and one output
    input_idx = engine.get_binding_index(engine[0])
    output_idx = engine.get_binding_index(engine[1])

    input_shape = engine.get_binding_shape(input_idx)   # e.g. (1, 3, 640, 640)
    output_shape = engine.get_binding_shape(output_idx) # e.g. (1, N, 84)

    print("Input shape:", input_shape)
    print("Output shape:", output_shape)

    # Allocate host/device buffers
    input_size_bytes = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    output_size_bytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_size_bytes)
    d_output = cuda.mem_alloc(output_size_bytes)

    h_output = np.empty(trt.volume(output_shape), dtype=np.float32)

    bindings = [int(d_input), int(d_output)]

    # Set up video IO
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        img, scale, dw, dh, (orig_h, orig_w) = preprocess(frame, INPUT_SIZE)
        img_batch = np.expand_dims(img, axis=0).ravel()

        # Copy to device
        cuda.memcpy_htod(d_input, img_batch)

        # Run inference
        context.execute_v2(bindings)

        # Copy back
        cuda.memcpy_dtoh(h_output, d_output)
        output = h_output.reshape(output_shape)

        # Parse YOLOv8 output: assume (1, N, 84) = (x,y,w,h, conf, 80 class scores)
        preds = output[0]
        boxes = preds[:, 0:4]
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5:]

        # Get best class + score
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
        scores = obj_conf * class_conf

        # Filter by confidence
        mask = scores >= CONF_THRESH
        boxes = boxes[mask]
        scores_f = scores[mask]
        class_ids = class_ids[mask]

        if boxes.size > 0:
            # Convert xywh -> xyxy in 640x640 space
            xy = np.zeros_like(boxes)
            xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

            # Undo letterbox scaling back to original frame size
            sx = orig_w / INPUT_SIZE[0]
            sy = orig_h / INPUT_SIZE[1]

            xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) * sx
            xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) * sy

            # Clip to image
            xy[:, 0] = np.clip(xy[:, 0], 0, orig_w - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, orig_h - 1)
            xy[:, 2] = np.clip(xy[:, 2], 0, orig_w - 1)
            xy[:, 3] = np.clip(xy[:, 3], 0, orig_h - 1)

            # NMS
            keep = nms(xy, scores_f, IOU_THRESH)
            xy = xy[keep]
            scores_keep = scores_f[keep]
            class_ids_keep = class_ids[keep]

            # Draw boxes
            for (x1, y1, x2, y2), sc, cid in zip(xy, scores_keep, class_ids_keep):
                label = f"{int(cid)}:{sc:.2f}"
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    cuda.Context.pop()

    dt = time.time() - t0
    print(f"Done. {frame_count} frames in {dt:.2f}s -> {frame_count/dt:.2f} FPS")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()