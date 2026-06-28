# 使用NVIDIA CUDA基础镜像（nvcr.io 官方源，避免 docker hub mirror 缓存残缺）
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG UBUNTU_MIRROR
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app

# NVIDIA GPU相关环境变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional mirror override for hosts whose proxy intermittently returns 502
# against archive.ubuntu.com.
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g; s|http://security.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
    fi

# Use Ubuntu 22.04's native Python 3.10 to avoid the deadsnakes PPA and keep
# the apt transaction small. The CUDA devel base already contains compilers;
# Python/materials dependencies are installed from wheels whenever possible.
RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置 python 命令
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 创建必要的目录并设置权限
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# 升级pip和安装wheel
RUN pip install --upgrade pip setuptools wheel

# 复制主项目requirements文件并安装依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 安装PyTorch和CrystaLLM特定依赖
RUN pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
RUN pip install omegaconf==2.3.0 pyzmq==25.1.1

# 复制CrystaLLM目录并安装
RUN mkdir -p /app/tools
COPY tools/CrystaLLM /app/tools/CrystaLLM
WORKDIR /app/tools/CrystaLLM
RUN pip install -e .
WORKDIR /app

# 复制脚本文件并设置权限
COPY scripts/model_manager.sh /usr/local/bin/model_manager.sh
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/model_manager.sh /usr/local/bin/entrypoint.sh

# 复制应用代码
COPY --chown=appuser:appuser . .
RUN chmod -R a+rX /app

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=300s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT:-8000}/ || exit 1

# 使用启动脚本作为入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "server.py", "--transport", "sse"]
