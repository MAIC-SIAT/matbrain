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

WORKDIR /app
COPY requirements.txt .

ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1000 --retries 10 torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/model-cache /app/logs

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MACE_CACHE_DIR=/app/model-cache
ENV XDG_CACHE_HOME=/app/model-cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

CMD ["python", "server.py", "--transport", "sse"]
