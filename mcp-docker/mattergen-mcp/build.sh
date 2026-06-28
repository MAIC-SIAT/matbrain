#!/bin/bash

# MatterGen-MCP Docker 构建脚本
# 使用方法: ./build.sh [选项]

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

# 显示帮助信息
show_help() {
    echo "MatterGen-MCP Docker 构建脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --clean    清理现有镜像和容器"
    echo "  -r, --run      构建后立即运行"
    echo "  --no-cache     不使用缓存构建"
    echo "  --pull         构建前拉取最新基础镜像"
    echo ""
    echo "示例:"
    echo "  $0                    # 标准构建"
    echo "  $0 --clean --run     # 清理后构建并运行"
    echo "  $0 --no-cache        # 无缓存构建"
}

# 检查 Docker 和 Docker Compose
check_requirements() {
    print_info "检查系统要求..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose 未安装或不在 PATH 中"
        exit 1
    fi

    # 检查 NVIDIA Docker 支持
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker 支持可能未正确配置"
        print_warning "请确保已安装 nvidia-docker2 或 nvidia-container-toolkit"
    fi

    print_success "系统要求检查通过"
}

# 清理现有镜像和容器
clean_docker() {
    print_info "清理现有 Docker 资源..."

    # 停止并删除容器
    if docker ps -a --format "table {{.Names}}" | grep -q "mattergen-mcp"; then
        print_info "停止并删除现有容器..."
        docker-compose down --remove-orphans 2>/dev/null || true
        docker rm -f mattergen-mcp 2>/dev/null || true
    fi

    # 删除镜像
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "mattergen-mcp"; then
        print_info "删除现有镜像..."
        docker rmi mattergen-mcp:latest 2>/dev/null || true
    fi

    # 清理悬空镜像
    if [ "$(docker images -f "dangling=true" -q)" ]; then
        print_info "清理悬空镜像..."
        docker image prune -f
    fi

    print_success "Docker 资源清理完成"
}

# 验证配置文件
validate_config() {
    print_info "验证配置文件..."

    if [ ! -f ".env" ]; then
        print_error ".env 文件不存在"
        exit 1
    fi

    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml 文件不存在"
        exit 1
    fi

    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile 文件不存在"
        exit 1
    fi

    # 检查模型目录
    source .env
    if [ ! -d "$HOST_MODEL_DIR" ]; then
        print_warning "模型目录不存在: $HOST_MODEL_DIR"
        print_warning "请确保模型文件已正确放置"
    fi

    print_success "配置文件验证通过"
}

# 构建 Docker 镜像
build_image() {
    print_info "开始构建 MatterGen-MCP Docker 镜像..."

    local build_args=""

    if [ "$NO_CACHE" = true ]; then
        build_args="$build_args --no-cache"
        print_info "使用无缓存构建"
    fi

    if [ "$PULL" = true ]; then
        build_args="$build_args --pull"
        print_info "拉取最新基础镜像"
    fi

    # 显示构建信息
    print_info "构建参数: $build_args"
    print_info "这可能需要几分钟时间，请耐心等待..."

    # 执行构建
    if docker-compose build $build_args; then
        print_success "Docker 镜像构建成功"
    else
        print_error "Docker 镜像构建失败"
        exit 1
    fi
}

# 运行容器
run_container() {
    print_info "启动 MatterGen-MCP 容器..."

    if docker-compose up -d; then
        print_success "容器启动成功"
        print_info "容器状态:"
        docker-compose ps
        print_info ""
        print_info "查看日志: docker-compose logs -f"
        print_info "停止服务: docker-compose down"

        # 显示服务信息
        source .env
        print_success "MatterGen-MCP 服务已启动"
        print_info "服务地址: http://localhost:${HOST_PORT:-5670}"
        print_info "健康检查: http://localhost:${HOST_PORT:-5670}/health"
    else
        print_error "容器启动失败"
        exit 1
    fi
}

# 主函数
main() {
    local CLEAN=false
    local RUN=false
    local NO_CACHE=false
    local PULL=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            -r|--run)
                RUN=true
                shift
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --pull)
                PULL=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_info "开始 MatterGen-MCP Docker 构建流程"
    print_info "========================================"

    # 执行构建流程
    check_requirements
    validate_config

    if [ "$CLEAN" = true ]; then
        clean_docker
    fi

    build_image

    if [ "$RUN" = true ]; then
        run_container
    fi

    print_success "构建流程完成!"
    print_info "========================================"

    if [ "$RUN" = false ]; then
        print_info "要启动服务，请运行: docker-compose up -d"
    fi
}

# 执行主函数
main "$@"
