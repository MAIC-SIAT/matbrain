#!/bin/bash

# pymatgen-mcp一键启动脚本
# 功能：检测matgl-mcp依赖，创建网络，构建并启动pymatgen-mcp

set -e  # 遇到错误立即退出

# 配置变量
NETWORK_NAME="mcp-shared-network"
MATGL_CONTAINER="matgl-mcp"
MATGL_PORT="5668"
PYMATGEN_DIR="."

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否运行
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker未运行，请先启动Docker"
        exit 1
    fi
    print_success "Docker运行正常"
}

# 检查matgl-mcp是否运行
check_matgl_mcp() {
    print_info "检查matgl-mcp服务状态..."

    # 检查容器是否存在且运行
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "^${MATGL_CONTAINER}.*Up"; then
        print_success "matgl-mcp容器运行中"

        # 检查端口是否可访问
        if docker exec $MATGL_CONTAINER curl -f http://localhost:$MATGL_PORT/ >/dev/null 2>&1; then
            print_success "matgl-mcp服务响应正常 (端口:$MATGL_PORT)"
            return 0
        else
            print_warning "matgl-mcp容器运行中，但服务未响应"
            return 1
        fi
    else
        print_error "matgl-mcp容器未运行"
        echo "请先启动matgl-mcp服务："
        echo "  cd matgl-mcp && docker compose up -d"
        return 1
    fi
}

# 创建共享网络
create_network() {
    print_info "检查/创建共享网络: $NETWORK_NAME"

    if docker network ls --format "{{.Name}}" | grep -q "^${NETWORK_NAME}$"; then
        print_success "共享网络已存在"
    else
        if docker network create $NETWORK_NAME; then
            print_success "共享网络创建成功"
        else
            print_error "共享网络创建失败"
            exit 1
        fi
    fi
}

# 检查matgl-mcp是否在共享网络中
check_matgl_network() {
    print_info "检查matgl-mcp网络连接..."

    # 获取matgl-mcp容器的网络信息
    matgl_networks=$(docker inspect $MATGL_CONTAINER --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' 2>/dev/null || echo "")

    if echo "$matgl_networks" | grep -q "$NETWORK_NAME"; then
        print_success "matgl-mcp已连接到共享网络"
        return 0
    else
        print_warning "matgl-mcp未连接到共享网络，尝试连接..."
        if docker network connect $NETWORK_NAME $MATGL_CONTAINER 2>/dev/null; then
            print_success "matgl-mcp已连接到共享网络"
            return 0
        else
            print_error "无法将matgl-mcp连接到共享网络"
            echo "建议重启matgl-mcp服务并使用共享网络配置"
            return 1
        fi
    fi
}

# 构建pymatgen-mcp镜像
build_pymatgen() {
    print_info "构建pymatgen-mcp镜像..."

    if docker compose build; then
        print_success "pymatgen-mcp镜像构建成功"
    else
        print_error "pymatgen-mcp镜像构建失败"
        exit 1
    fi
}

# 启动pymatgen-mcp服务
start_pymatgen() {
    print_info "启动pymatgen-mcp服务..."

    if docker compose up -d; then
        print_success "pymatgen-mcp服务启动成功"

        # 等待服务启动
        print_info "等待服务完全启动..."
        sleep 10

        # 检查服务状态
        if docker compose ps | grep -q "pymatgen-mcp.*Up"; then
            print_success "pymatgen-mcp服务运行正常"

            # 显示服务信息
            echo
            echo "=== 服务信息 ==="
            echo "pymatgen-mcp端口: 5672"
            echo "matgl-mcp连接: http://matgl-mcp:5668"
            echo "健康检查: http://localhost:5672/pymatgen/health"
            echo "================"
        else
            print_error "pymatgen-mcp服务启动异常"
            echo "查看日志: docker compose logs"
        fi
    else
        print_error "pymatgen-mcp服务启动失败"
        exit 1
    fi
}

# 测试连接
test_connection() {
    print_info "测试pymatgen-mcp与matgl-mcp的连接..."

    # 等待一段时间让服务完全启动
    sleep 5

    # 检查pymatgen-mcp健康状态
    if curl -f http://localhost:5672/pymatgen/health >/dev/null 2>&1; then
        print_success "pymatgen-mcp健康检查通过"
    else
        print_warning "pymatgen-mcp健康检查失败，但服务可能仍在启动中"
    fi

    # 从pymatgen-mcp容器内测试连接matgl-mcp
    if docker exec pymatgen-mcp curl -f http://matgl-mcp:5668/ >/dev/null 2>&1; then
        print_success "pymatgen-mcp可以访问matgl-mcp服务"
    else
        print_error "pymatgen-mcp无法访问matgl-mcp服务"
        echo "请检查网络配置和matgl-mcp服务状态"
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "    pymatgen-mcp 一键启动脚本"
    echo "========================================"
    echo

    # 1. 检查Docker
    check_docker

    # 2. 检查matgl-mcp依赖
    if ! check_matgl_mcp; then
        print_error "matgl-mcp依赖检查失败，无法继续"
        exit 1
    fi

    # 3. 创建共享网络
    create_network

    # 4. 检查matgl-mcp网络连接
    check_matgl_network

    # 5. 构建pymatgen-mcp
    build_pymatgen

    # 6. 启动pymatgen-mcp
    start_pymatgen

    # 7. 测试连接
    test_connection

    echo
    print_success "pymatgen-mcp启动完成！"
    echo
    echo "使用方法："
    echo "  查看日志: docker compose logs -f"
    echo "  停止服务: docker compose down"
    echo "  健康检查: curl http://localhost:5672/pymatgen/health"
}

# 执行主函数
main "$@"
