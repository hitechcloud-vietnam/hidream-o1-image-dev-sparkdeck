# HiDream-O1-Image-Dev — Spark AI Hub build

Native ARM64 + CUDA 13 image for the DGX Spark, packaging the upstream
[HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) Flask UI
with the distilled `HiDream-O1-Image-Dev` checkpoint baked in.

## Build

```
docker build --build-arg HF_TOKEN=$HF_TOKEN \
  -t ghcr.io/waxacabytes/hidream-o1-image-dev-spark-ai-hub:latest .
```

## Notes

- `flash-attn` is intentionally not installed (no aarch64 + CUDA 13 wheel);
  `models/pipeline.py` is patched to fall back to PyTorch SDPA.
- Weights are pre-downloaded into `/models/HiDream-O1-Image-Dev` so the
  container starts fully offline (`HF_HUB_OFFLINE=1` by default).
- UI listens on port 7860 inside the container; the recipe maps it to
  host port 7863.
