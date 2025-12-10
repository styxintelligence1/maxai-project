FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Ultralytics + common deps
RUN pip3 install --no-cache-dir ultralytics opencv-python numpy pyyaml tqdm

COPY . /app

CMD ["bash"]
