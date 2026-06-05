FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the community-maintained VibeVoice repo (Microsoft removed the original).
RUN git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git /app/VibeVoice

WORKDIR /app/VibeVoice

# Install vibevoice and runpod + tooling.
RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir \
    runpod==1.7.7 \
    soundfile==0.12.1 \
    "huggingface_hub>=0.24,<1.0"

WORKDIR /app

# Pre-download model weights at build time so cold start skips the ~6GB download.
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='vibevoice/VibeVoice-1.5B', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
