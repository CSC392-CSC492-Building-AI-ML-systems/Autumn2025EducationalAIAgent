# ---- Base: CUDA + PyTorch (RunPod official)
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# vLLM + deps
RUN pip install --no-cache-dir \
    runpod \
    vllm \
    transformers \
    accelerate \
    huggingface_hub \
    hf_transfer \
    einops

# Copy your handler
COPY handler.py /app/handler.py

# Common envs you can override in the RunPod console
ENV MODEL1_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    MODEL1_MAX_NEW_TOKENS=1024 \
    MODEL1_TEMPERATURE=0.0 \
    MODEL1_TOP_P=1.0 \
    MODEL1_GPU_UTIL=0.90 \
    MODEL1_MAX_MODEL_LEN=8192