FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04   

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone Forge
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge webui
WORKDIR /app/webui

# Install PyTorch with CUDA 12.1
RUN pip3 install torch torchvision torchaudio joblib --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip3 install -r requirements_versions.txt

EXPOSE 7860

CMD ["python3", "launch.py", "--listen", "--port", "7860", "--skip-prepare-environment"]
