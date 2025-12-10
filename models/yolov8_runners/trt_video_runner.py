import cv2
import numpy as np
import time
import tensorrt as trt

ENGINE_PATH = "C:/models/yolov8x_fp16.engine"
VIDEO_FOLDER = "C:/test_videos"

INPUT_VIDEOS = [
    "max_360_20s.mp4",
    "fixed_cam_20s.mp4"
]


# ==================================================
# LOAD TensorRT ENGINE
# ==================================================
def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# ==================================================
# ALLOCATE BUFFERS
# TensorRT v10 uses tensor names, not binding indexes
# ==================================================
def allocate_buffers(engine, context):
    bindings = []
    inputs = []
    outputs = []

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        size = trt.volume(shape)

        # Host memory
        host_mem = np.empty(size, dtype=dtype)

        # Device memory
        device_mem = trt.DeviceMemory(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append((tensor_name, host_mem, device_mem))
        else:
            outputs.append((tensor_name, host_mem, device_mem))

    return bindings, inputs, outputs


# ==================================================
# PREPROCESS
# ==================================================
def preprocess(frame, w, h):
    img = cv2.resize(frame, (w, h))
    img = img[:, :, ::-1].astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


# ==================================================
# MAIN VIDEO RUNNER
# ==================================================
def run_video(engine):

    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_name)

    input_h = input_shape[2]
    input_w = input_shape[3]

    for vid in INPUT_VIDEOS:

        video_path = f"{VIDEO_FOLDER}/{vid}"
        cap = cv2.VideoCapture(video_path)

        out_path = f"{VIDEO_FOLDER}/{vid}_trt_output.mp4"
        width  = int(cap.get(3))
        height = int(cap.get(4))
        fps    = cap.get(5)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f"\n=== Running TensorRT on {vid} ===")

        frame_id = 0
        t0 = time.time()

        # allocate buffers once
        bindings, inputs, outputs = allocate_buffers(engine, context)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            inp = preprocess(frame, input_w, input_h)

            # copy data into input buffer
            np.copyto(inputs[0][1], inp.ravel())

            # copy host → device
            inputs[0][2].copy_from(inputs[0][1].tobytes())

            # run inference
            context.set_tensor_address(inputs[0][0], inputs[0][2])
            for name, _, dev_mem in outputs:
                context.set_tensor_address(name, dev_mem)

            context.execute_async_v3(0)

            # copy device → host
            for name, host_mem, dev_mem in outputs:
                dev_mem.copy_to(host_mem)

            # Simple visual confirmation
            cv2.putText(frame, "TensorRT RUNNING (no decode)", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            writer.write(frame)
            frame_id += 1

        cap.release()
        writer.release()

        dt = time.time() - t0
        print(f"Processed {frame_id} frames in {dt:.2f}s "
              f"({frame_id/dt:.2f} FPS)")


if __name__ == "__main__":
    print("Loading engine...")
    engine = load_engine(ENGINE_PATH)
    print("Running videos...")
    run_video(engine)
    print("\nDone.")