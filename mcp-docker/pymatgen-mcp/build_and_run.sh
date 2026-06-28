#!/bin/bash

# PyMatGen-MCP Docker 构建和运行脚本

set -e

echo "🚀 PyMatGen-MCP Docker 构建和运行脚本"
echo "=================================="

# 检查 .env 文件是否存在
if [ ! -f ".env" ]; then
    echo "❌ 错误: .env 文件不存在"
    echo "请确保 .env 文件存在并包含必要的环境变量"
    exit 1
fi

# 加载环境变量
source .env

echo "📋 当前配置:"
echo "  - 服务端口: ${HOST_PORT:-5673}"
echo "  - 路径前缀: ${MCP_PATH_PREFIX:-/pymatgen}"
echo "  - 日志级别: ${LOG_LEVEL:-INFO}"
echo ""

# 停止并删除现有容器（如果存在）
echo "🛑 停止现有容器..."
docker-compose down --remove-orphans 2>/dev/null || true

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ 服务启动成功!"
    echo ""
    echo "📡 服务信息:"
    echo "  - 服务地址: http://localhost:${HOST_PORT:-5673}${MCP_PATH_PREFIX:-/pymatgen}"
    echo "  - 健康检查: http://localhost:${HOST_PORT:-5673}${MCP_PATH_PREFIX:-/pymatgen}/health"
    echo "  - SSE 端点: http://localhost:${HOST_PORT:-5673}${MCP_PATH_PREFIX:-/pymatgen}/sse"
    echo ""
    echo "🔧 可用工具:"
    echo "  - analyze_thermodynamic_stability_pymatgen: 热力学稳定性分析"
    echo "  - check_chemical_formula_valence_pymatgen: 化学式价态验证"
    echo "  - check_structure_atomic_geometry_pymatgen: 原子几何检查"
    echo ""
    echo "📊 查看日志: docker-compose logs -f"
    echo "🛑 停止服务: docker-compose down"
else
    echo "❌ 服务启动失败!"
    echo "查看日志: docker-compose logs"
    exit 1
fi
