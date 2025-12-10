import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart  # from cuda-python

# ==== CONFIG ====
ENGINE_PATH = r"C:\models\yolov8x_fp16.engine"
VIDEO_DIR = r"C:\test_videos"
MAX_FRAMES_PER_VIDEO = 600
# =================

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def check_cuda_err(err):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA Runtime Error: {err}")
    else:
        raise RuntimeError(f"Unknown CUDA error type: {err}")

def cuda_call(call):
    err, *rest = call
    check_cuda_err(err)
    if len(rest) == 1:
        return rest[0]
    return rest

class HostDeviceMem:
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * np.dtype(dtype).itemsize
        host_ptr = cuda_call(cudart.cudaMallocHost(nbytes))
        self.host = np.ctypeslib.as_array(
            np.ctypeslib.as_ctypes_type(dtype).from_address(host_ptr),
            shape=(size,),
        )
        self.device = cuda_call(cudart.cudaMalloc(nbytes))
        self.nbytes = nbytes

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

def allocate_buffers(engine: trt.ICudaEngine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in tensor_names:
        shape = engine.get_tensor_shape(name)
        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        mem = HostDeviceMem(size, dtype)
        bindings.append(int(mem.device))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append((name, shape, mem))
        else:
            outputs.append((name, shape, mem))
    return inputs, outputs, bindings, stream

def run_inference(context, bindings, inputs, outputs, stream):
    for _, _, mem in inputs:
        cuda_call(
            cudart.cudaMemcpyAsync(
                mem.device, mem.host, mem.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
            )
        )
    context.execute_v2(bindings)
    for _, _, mem in outputs:
        cuda_call(
            cudart.cudaMemcpyAsync(
                mem.host, mem.device, mem.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
            )
        )
    cuda_call(cudart.cudaStreamSynchronize(stream))

def preprocess_frame(frame_bgr, input_shape):
    _, c, h, w = input_shape
    frame_resized = cv2.resize(frame_bgr, (w, h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main():
    engine_path = Path(ENGINE_PATH)
    video_dir = Path(VIDEO_DIR)

    print("Engine:", engine_path)
    print("Videos folder:", video_dir)

    videos = sorted([p for p in video_dir.iterdir() if p.suffix.lower() == ".mp4"])
    if not videos:
        print("NO VIDEOS FOUND.")
        return

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    input_name, input_shape, input_mem = inputs[0]
    print("Model input shape:", input_shape)

    for vid in videos:
        print("\n=== Running:", vid.name, "===")
        cap = cv2.VideoCapture(str(vid))
        frames = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            if frames > MAX_FRAMES_PER_VIDEO:
                print("Hit testing limit, stopping this video early.")
                break

            img = preprocess_frame(frame, input_shape)
            np.copyto(input_mem.host, img.ravel())
            run_inference(context, bindings, inputs, outputs, stream)

            if frames % 50 == 0:
                fps = frames / (time.time() - t0)
                print(f"  {frames} frames, ~{fps:.1f} FPS")

        cap.release()
        t1 = time.time()
        if frames > 0:
            print(f"Done {vid.name}: {frames} frames in {t1 - t0:.2f}s -> {frames/(t1-t0):.1f} FPS")

    for _, _, mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))

if __name__ == "__main__":
    main()