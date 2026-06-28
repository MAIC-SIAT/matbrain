# Use the full Python image to avoid apt-get during build.
FROM python:3.11

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# 设置工作目录
WORKDIR /app

# 先复制requirements.txt（利用Docker层缓存）
COPY requirements.txt .

# Install the CUDA-enabled PyTorch stack first so later dependency resolution
# does not downgrade it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install PyG wheels that match the torch build above.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Route the remaining packages through the mirror to avoid read timeouts.
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=120

# Install torchdata/torch-geometric explicitly before the rest of the stack so
# pip does not try to upgrade the CUDA torch build during later resolution.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torchdata==0.8.0 torch-geometric==2.8.0

# Install the application dependencies. MatGL 3.x is PyG-first, so DGL is gone.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the project after dependency layers for better cache reuse.
COPY . .

# Create runtime directories.
RUN mkdir -p /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "server.py", "--transport", "sse"]
