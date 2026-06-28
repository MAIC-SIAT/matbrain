# ================================
# Stage 1: Builder - 构建阶段
# ================================
FROM python:3.11-slim AS builder

# 设置构建时环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装构建依赖
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip和安装wheel
RUN pip install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt /tmp/requirements.txt

# 安装Python依赖到虚拟环境
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# 清理虚拟环境，移除不必要的文件
RUN find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyo" -delete && \
    find /opt/venv -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# ================================
# Stage 2: Runtime - 运行阶段
# ================================
FROM python:3.11-slim AS runtime

# 设置运行时环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 只安装运行时必需的系统依赖
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /app

# 创建必要的目录并设置权限
RUN mkdir -p /data/mp_cifs /data/mp_props /app/logs && \
    chown -R appuser:appuser /app /data

# 复制应用代码
COPY --chown=appuser:appuser . .

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=300s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT:-8000}/ || exit 1

# 启动命令
CMD ["python", "server.py", "--transport", "sse"]
