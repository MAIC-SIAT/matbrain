#!/bin/bash

# CrystaLLM MCP 轻量化镜像构建脚本
# 针对科学计算应用优化的Docker构建工具

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
IMAGE_NAME="crystallm-mcp"
TAG="latest"
DOCKERFILE="Dockerfile"
COMPOSE_FILE="docker-compose.yml"

# 函数：打印带颜色的消息
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

# 函数：获取镜像大小
get_image_size() {
    local image_tag=$1
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "$image_tag" | awk '{print $2}' || echo "N/A"
}

# 函数：检查必要文件
check_requirements() {
    print_info "检查必要文件..."

    local missing_files=()

    if [[ ! -f "$DOCKERFILE" ]]; then
        missing_files+=("$DOCKERFILE")
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        missing_files+=("$COMPOSE_FILE")
    fi

    if [[ ! -f "requirements.txt" ]]; then
        missing_files+=("requirements.txt")
    fi

    if [[ ! -d "tools/CrystaLLM" ]]; then
        missing_files+=("tools/CrystaLLM/")
    fi

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "缺少必要文件："
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi

    print_success "所有必要文件检查通过"
}

# 函数：构建镜像
build_image() {
    print_info "构建 CrystaLLM MCP 轻量化镜像..."
    print_info "镜像标签: ${IMAGE_NAME}:${TAG}"

    # 记录构建开始时间
    local start_time=$(date +%s)

    # 构建镜像
    if docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${TAG}" .; then
        local end_time=$(date +%s)
        local build_time=$((end_time - start_time))
        print_success "镜像构建完成！"
        print_info "构建时间: ${build_time}秒"

        # 获取镜像大小
        local image_size=$(get_image_size "${IMAGE_NAME}:${TAG}")
        print_info "镜像大小: $image_size"

        return 0
    else
        print_error "镜像构建失败！"
        return 1
    fi
}

# 函数：启动服务
start_service() {
    print_info "启动 CrystaLLM MCP 服务..."

    if docker-compose -f "$COMPOSE_FILE" up -d; then
        print_success "服务启动成功！"
        print_info "服务状态："
        docker-compose -f "$COMPOSE_FILE" ps

        # 显示日志
        print_info "查看服务日志（按 Ctrl+C 退出）："
        sleep 2
        docker-compose -f "$COMPOSE_FILE" logs -f
    else
        print_error "服务启动失败！"
        return 1
    fi
}

# 函数：停止服务
stop_service() {
    print_info "停止 CrystaLLM MCP 服务..."

    if docker-compose -f "$COMPOSE_FILE" down; then
        print_success "服务已停止"
    else
        print_error "停止服务失败！"
        return 1
    fi
}

# 函数：重启服务
restart_service() {
    print_info "重启 CrystaLLM MCP 服务..."
    stop_service
    sleep 2
    start_service
}

# 函数：查看服务状态
show_status() {
    print_info "CrystaLLM MCP 服务状态："
    echo "----------------------------------------"

    # 检查容器状态
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        print_success "服务正在运行"
        docker-compose -f "$COMPOSE_FILE" ps

        # 显示端口信息
        local port=$(grep "HOST_PORT" .env 2>/dev/null | cut -d'=' -f2 || echo "5669")
        print_info "服务地址: http://localhost:${port}"
        print_info "健康检查: http://localhost:${port}/health"

    else
        print_warning "服务未运行"
    fi

    echo "----------------------------------------"

    # 显示镜像信息
    local image_size=$(get_image_size "${IMAGE_NAME}:${TAG}")
    if [[ "$image_size" != "N/A" ]]; then
        print_info "镜像信息: ${IMAGE_NAME}:${TAG} (${image_size})"
    fi
}

# 函数：查看日志
show_logs() {
    print_info "查看 CrystaLLM MCP 服务日志："
    docker-compose -f "$COMPOSE_FILE" logs -f
}

# 函数：清理资源
cleanup() {
    print_info "清理 Docker 资源..."

    # 停止服务
    docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true

    # 清理悬空镜像
    docker image prune -f > /dev/null 2>&1 || true

    # 可选：删除镜像
    if [[ "$1" == "--remove-image" ]]; then
        docker rmi "${IMAGE_NAME}:${TAG}" 2>/dev/null || true
        print_info "已删除镜像: ${IMAGE_NAME}:${TAG}"
    fi

    print_success "清理完成"
}

# 函数：显示使用说明
show_usage() {
    echo "CrystaLLM MCP Docker 管理工具"
    echo ""
    echo "使用方法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  build              构建镜像（默认）"
    echo "  start              启动服务"
    echo "  stop               停止服务"
    echo "  restart            重启服务"
    echo "  status             查看服务状态"
    echo "  logs               查看服务日志"
    echo "  cleanup            清理Docker资源"
    echo "  help               显示此帮助信息"
    echo ""
    echo "选项:"
    echo "  --remove-image     清理时同时删除镜像"
    echo ""
    echo "示例:"
    echo "  $0                 构建镜像"
    echo "  $0 build           构建镜像"
    echo "  $0 start           启动服务"
    echo "  $0 status          查看状态"
    echo "  $0 cleanup --remove-image  清理所有资源"
}

# 主函数
main() {
    local command="${1:-build}"
    local option="$2"

    print_info "CrystaLLM MCP Docker 管理工具"
    print_info "================================"

    # 检查 Docker 是否运行
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker 未运行或无法访问"
        exit 1
    fi

    case $command in
        "build")
            check_requirements
            build_image
            ;;
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            restart_service
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup "$option"
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            print_error "未知命令: $command"
            show_usage
            exit 1
            ;;
    esac

    print_success "操作完成！"
}

# 执行主函数
main "$@"
