FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir \
      "transformers==4.57.1" \
      diffusers \
      accelerate \
      einops \
      numpy \
      pillow \
      tqdm \
      scipy \
      flask \
      openai \
      python-dotenv \
      safetensors \
      sentencepiece \
      protobuf \
      "huggingface_hub[hf_xet]"

# Clone the upstream HiDream-O1-Image repository at a pinned commit.
# flash-attn is not built for aarch64 + CUDA 13; patch pipeline.py to use SDPA.
# Clone upstream and force the SDPA path: aarch64 + CUDA 13 has no flash_attn
# wheel, and the model gates on a `use_flash_attn` kwarg in pipeline.py.
RUN git clone https://github.com/HiDream-ai/HiDream-O1-Image.git /app/HiDream-O1-Image && \
    cd /app/HiDream-O1-Image && \
    sed -i 's/"use_flash_attn": True/"use_flash_attn": False/g' models/pipeline.py

# Pre-download Dev weights at build time so the container is offline-first.
ARG HF_TOKEN=""
RUN HF_TOKEN="${HF_TOKEN}" python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='HiDream-ai/HiDream-O1-Image-Dev', \
                  local_dir='/models/HiDream-O1-Image-Dev', \
                  local_dir_use_symlinks=False)"

RUN mkdir -p /data/output

WORKDIR /app/HiDream-O1-Image

EXPOSE 7860
