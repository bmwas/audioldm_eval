ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy the repository
COPY . .

# Create virtual environment and install dependencies
RUN python3.10 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    # Install ssr_eval without dependencies to avoid MySQL-python issues
    pip install ssr_eval --no-deps && \
    # Install audioldm_eval without dependencies first
    pip install -e . --no-deps && \
    # Install remaining dependencies
    pip install torch torchaudio transformers scikit-image torchlibrosa absl-py scipy tqdm librosa && \
    # Install audio processing dependencies
    pip install resampy && \
    # Install API dependencies
    pip install fastapi uvicorn python-multipart aiofiles

# Create directories for uploads and results
RUN mkdir -p /app/uploads /app/results

# Expose port
EXPOSE 2600

# Activate virtual environment and start the API
CMD ["/bin/bash", "-c", "source venv/bin/activate && python app.py"]
