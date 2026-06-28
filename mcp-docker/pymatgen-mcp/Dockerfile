# 使用Python slim镜像
FROM python:3.11-slim

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

# 默认 PyPI 在这台 host 上会 read timeout（files.pythonhosted.org），改用清华 mirror + 拉长 timeout
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=120

# 使用缓存挂载安装Python依赖，既快速又不增加镜像大小
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --no-cache-dir

# 复制项目文件（放在最后，代码变更不会影响依赖层）
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "server.py", "--transport", "sse"]
