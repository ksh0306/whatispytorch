FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 1. pip root 경고 무시 및 파이썬 출력 최적화
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 3. SSH 설정 (비번: password123)
RUN mkdir /var/run/sshd
RUN echo 'root:1q2w3e4r' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 4. PyTorch 설치 (CUDA 12.1 대응)
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]