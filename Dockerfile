# The only line that ever gets cached is next
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# we do not want interactive anything
ENV DEBIAN_FRONTEND=noninteractive

# dummy to hold timestamp during build to bust the cache
# the lines probably were not cached, but we leave for debug anyways
# pretty much nothing caches and the build takes long
# especially the "exporting image" and "exporting layers" steps.
ARG DUMMY=

# Install system deps
RUN apt-get update && apt-get install -y \
    python3 git wget nano curl htop libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade system jeez
RUN apt-get upgrade -y && apt-get dist-upgrade -y

# Download and install the latest pip directly using get-pip.py
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install rustup without modifying profile files
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

# Add to PATH
ENV PATH="/root/.cargo/bin:/root/.rustup/bin:${PATH}"

# Force rustup to use 1.70.0 globally
ENV RUSTUP_TOOLCHAIN=1.70.0

# Suppress common harmless warnings in production
# Does not affect errors or app-level logging
# Also exists in startup-webui.sh
ENV PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# Install and set default toolchain
RUN rustup install 1.70.0 && rustup default 1.70.0

# Verify
RUN echo "Cache bust: $DUMMY AFTER:" && rustc --version

# DEBUG - you need to add ```--progress=plain --build-arg DUMMY=$(date +%s)``` to your docker build cmd
RUN echo "Cache bust: $DUMMY AFTER INSTALL:" && rustc --version
RUN echo "Cache bust: $DUMMY AFTER INSTALL:" && which python3 && which pip3
RUN echo "Cache bust: $DUMMY AFTER INSTALL:" && python3 --version
RUN echo "Cache bust: $DUMMY AFTER INSTALL:" && pip3 --version

WORKDIR /app

COPY start-webui.sh /app/start-webui.sh

# Clone Forge
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge webui
WORKDIR /app/webui

# Install PyTorch with CUDA 12.1
RUN pip3 install --root-user-action ignore torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
RUN pip3 install --root-user-action ignore -r requirements_versions.txt

RUN pip3 install --root-user-action ignore joblib 
#insightface

#
## Install the git repositories manually, so we can cutback startup time
##
## Any repos not installed by this Dockerfile _should_ auto install on start
##
# 

# UPGRADE PIP & SETUPTOOLS
RUN pip3 install --root-user-action ignore --upgrade pip && \
    pip install --root-user-action ignore --upgrade pip && \
    pip3 install --root-user-action ignore --force-reinstall --no-deps "setuptools>=62.4"

# modules/launch_utils.py contains the repos and hashes
RUN mkdir -p /app/webui/repositories && \
    cd /app/webui/repositories && \
    git clone --config core.filemode=false https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git && \
    git clone --config core.filemode=false https://github.com/lllyasviel/huggingface_guess.git && \
    git clone --config core.filemode=false https://github.com/salesforce/BLIP.git

RUN echo "Cache bust: $DUMMY" && cd /app/webui/repositories/stable-diffusion-webui-assets && \
    git checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917

RUN cd /app/webui/repositories/huggingface_guess && \
    git checkout 84826248b49bb7ca754c73293299c4d4e23a548d

RUN cd /app/webui/repositories/BLIP && \
    git checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    pip3 install --root-user-action ignore -r requirements.txt

# modules/launch_utils.py contains the repos and hashes
# kept for reference :)
#assets_commit_hash = os.environ.get('ASSETS_COMMIT_HASH', "6f7db241d2f8ba7457bac5ca9753331f0c266917")
#huggingface_guess_commit_hash = os.environ.get('', "84826248b49bb7ca754c73293299c4d4e23a548d")
#blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

#
## Startup SD WebUI Forge
#

EXPOSE 7860
CMD ["/app/start-webui.sh"]
