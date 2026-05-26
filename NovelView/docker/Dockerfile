# Training image for the geometric refiner (Task B: Novel View Synthesis).
# Base: PyTorch 2.4.1 + CUDA 12.1 (matches A5000 / modern drivers).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV (headless) + scipy runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements.txt /workspace/requirements.txt
# Robust to flaky/slow networks during build (build with: docker build --network=host)
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r /workspace/requirements.txt

# Project code (geo_warp/ + refiner/). Data and cache are mounted, not baked in.
COPY geo_warp /workspace/geo_warp
COPY refiner /workspace/refiner

# tini gives clean signal handling so `docker stop` / Ctrl-C kill training properly
ENTRYPOINT ["tini", "--"]
CMD ["python", "-u", "refiner/train.py", "--help"]
