# Base image with Python 3.7 and 3.11
FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.7 \
    python3.7-venv \
    python3.11 \
    python3.11-venv \
    wget \
    curl \
    unzip

# Install pip for Python 3.7 using the specific version link
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py -O get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

# Install pip for Python 3.11 without relying on the system's pip
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Set working directory
WORKDIR /git-repos

# Copy the project files
COPY . .

# Create virtual environments for Python 3.7 (CARLA setup) and Python 3.11 (VAE, Masking, Counterfactual)
RUN python3.7 -m venv /opt/venv && \
    python3.11 -m venv /opt/venv-project

# Install dependencies for Python 3.7 (CARLA-related tasks)
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install carla==0.9.15 pygame==2.1.0 numpy==1.21.0

# Install dependencies for Python 3.11 (VAE, masking, counterfactual explanations)
RUN /opt/venv-project/bin/pip install --upgrade pip && \
    /opt/venv-project/bin/pip install torch torchvision numpy==2.0.2 \
    matplotlib==3.9.2 scikit-learn==1.5.1 lime==0.2.0.1 opencv-python==4.10.0.84 pandas==2.2.2

# Set environment variable for both virtual environments (default to Python 3.11 environment)
ENV PATH="/opt/venv/bin:/opt/venv-project/bin:$PATH"

# Default command: start a shell
CMD ["/bin/bash"]
