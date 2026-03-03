FROM ubuntu:22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    jiwer \
    tqdm \
    Pillow

CMD ["/bin/bash"]
