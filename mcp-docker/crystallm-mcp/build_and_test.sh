#!/bin/bash
"""
CrystaLLM MCP Docker 构建和测试脚本
自动化构建、启动和测试 GPU 功能
"""

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要的命令
check_prerequisites() {
    log_info "检查系统环境..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose 未安装或不在 PATH 中"
        exit 1
    fi

    # 检查 NVIDIA Docker 支持
    if ! docker info | grep -q "nvidia"; then
        log_warning "未检测到 NVIDIA Docker 运行时，GPU 功能可能不可用"
    fi

    log_success "系统环境检查完成"
}

# 清理旧容器和镜像
cleanup() {
    log_info "清理旧的容器和镜像..."

    # 停止并删除容器
    if docker ps -a | grep -q "crystallm-mcp"; then
        docker stop crystallm-mcp 2>/dev/null || true
        docker rm crystallm-mcp 2>/dev/null || true
        log_info "已删除旧容器"
    fi

    # 可选：删除旧镜像（取消注释以启用）
    # if docker images | grep -q "crystallm-mcp"; then
    #     docker rmi crystallm-mcp:latest 2>/dev/null || true
    #     log_info "已删除旧镜像"
    # fi
}

# 构建镜像
build_image() {
    log_info "开始构建 CrystaLLM MCP Docker 镜像..."

    # 检查 .env 文件
    if [ ! -f ".env" ]; then
        log_warning ".env 文件不存在，请确保环境变量配置正确"
    fi

    # 构建镜像
    if docker-compose build --no-cache; then
        log_success "镜像构建成功"
    else
        log_error "镜像构建失败"
        exit 1
    fi
}

# 启动容器
start_container() {
    log_info "启动 CrystaLLM MCP 容器..."

    if docker-compose up -d; then
        log_success "容器启动成功"

        # 等待容器完全启动
        log_info "等待容器初始化..."
        sleep 10

        # 检查容器状态
        if docker ps | grep -q "crystallm-mcp"; then
            log_success "容器运行正常"
        else
            log_error "容器启动失败"
            docker-compose logs
            exit 1
        fi
    else
        log_error "容器启动失败"
        exit 1
    fi
}

# 运行 GPU 测试
test_gpu() {
    log_info "运行 GPU 功能测试..."

    # 复制测试脚本到容器
    if docker cp test_gpu.py crystallm-mcp:/tmp/test_gpu.py; then
        log_info "测试脚本已复制到容器"
    else
        log_error "无法复制测试脚本"
        exit 1
    fi

    # 运行测试
    log_info "执行 GPU 测试..."
    if docker exec crystallm-mcp python /tmp/test_gpu.py; then
        log_success "GPU 测试通过"
    else
        log_error "GPU 测试失败"
        return 1
    fi
}

# 显示容器信息
show_info() {
    log_info "容器信息："
    echo "----------------------------------------"
    docker-compose ps
    echo "----------------------------------------"

    log_info "查看日志："
    echo "docker-compose logs -f"

    log_info "进入容器："
    echo "docker exec -it crystallm-mcp bash"

    log_info "停止容器："
    echo "docker-compose down"
}

# 主函数
main() {
    echo "========================================"
    echo "CrystaLLM MCP Docker 构建和测试工具"
    echo "========================================"

    # 解析命令行参数
    SKIP_BUILD=false
    SKIP_TEST=false
    CLEANUP_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-test)
                SKIP_TEST=true
                shift
                ;;
            --cleanup-only)
                CLEANUP_ONLY=true
                shift
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --skip-build    跳过镜像构建"
                echo "  --skip-test     跳过 GPU 测试"
                echo "  --cleanup-only  仅清理，不构建"
                echo "  -h, --help      显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done

    # 执行步骤
    check_prerequisites
    cleanup

    if [ "$CLEANUP_ONLY" = true ]; then
        log_success "清理完成"
        exit 0
    fi

    if [ "$SKIP_BUILD" = false ]; then
        build_image
    fi

    start_container

    if [ "$SKIP_TEST" = false ]; then
        if test_gpu; then
            log_success "所有测试通过！CrystaLLM MCP GPU 环境配置正确。"
        else
            log_warning "GPU 测试失败，但容器已启动。请检查配置。"
        fi
    fi

    show_info

    echo "========================================"
    log_success "CrystaLLM MCP 部署完成！"
    echo "========================================"
}

# 运行主函数
main "$@"
