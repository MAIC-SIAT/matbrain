#!/bin/bash
set -e

echo "=== CrystaLLM 容器启动 ==="

# 运行模型管理器
/usr/local/bin/model_manager.sh

# 验证模型文件
if [[ ! -f "${CRYSTALLM_MODEL_DIR:-/app/models}/ckpt.pt" ]]; then
    echo "错误: 模型文件未准备就绪"
    exit 1
fi

echo "=== 启动应用服务 ==="

# 执行原始命令
exec "$@"
