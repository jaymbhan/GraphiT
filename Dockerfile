# ── Dockerfile for M4 Mac (ARM64) ──────────────────────────────────────────────
# Note: Docker on M4 Mac runs ARM64 containers natively
FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies for building C++ extensions and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt /app/
COPY setup.py /app/
COPY setup_torch.py /app/

# Install Python dependencies
# Using modern PyTorch with ARM64 support
RUN pip install --upgrade pip wheel setuptools && \
    pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install torch-geometric==2.5.0 && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the entire project
COPY . /app/

# Build C++ extensions
RUN python setup.py build_ext --inplace && \
    python setup_torch.py build_ext --inplace

CMD ["python"]
