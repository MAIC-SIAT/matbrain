#!/bin/bash
set -e

# 配置
ZENODO_URL="https://zenodo.org/records/10642388/files"
MODEL_ARCHIVE="crystallm_v1_small.tar.gz"
MODEL_MD5="0221fbcd166bddb17f75be8a610892f3"
TEMP_DIR="/tmp/crystallm_model"
MODEL_DIR="${CRYSTALLM_MODEL_DIR:-/app/models}"
MODEL_FILE="$MODEL_DIR/ckpt.pt"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查模型是否存在
check_model_exists() {
    if [[ -f "$MODEL_FILE" ]]; then
        log_info "模型文件已存在: $MODEL_FILE"
        return 0
    else
        log_warn "模型文件不存在: $MODEL_FILE"
        return 1
    fi
}

# 创建必要目录
setup_directories() {
    log_info "创建目录结构..."
    mkdir -p "$MODEL_DIR"
    mkdir -p "$TEMP_DIR"
}

# 下载模型
download_model() {
    local archive_path="$TEMP_DIR/$MODEL_ARCHIVE"
    local download_url="$ZENODO_URL/$MODEL_ARCHIVE"

    log_info "开始下载模型: $download_url"

    # 使用curl下载，支持断点续传和进度显示
    if ! curl -L -C - --progress-bar "$download_url" -o "$archive_path"; then
        log_error "下载失败"
        return 1
    fi

    log_info "下载完成: $archive_path"
}

# 验证文件完整性
verify_download() {
    local archive_path="$TEMP_DIR/$MODEL_ARCHIVE"

    log_info "验证文件完整性..."

    if ! command -v md5sum >/dev/null 2>&1; then
        log_warn "md5sum不可用，跳过校验"
        return 0
    fi

    local actual_md5=$(md5sum "$archive_path" | cut -d' ' -f1)

    if [[ "$actual_md5" == "$MODEL_MD5" ]]; then
        log_info "MD5校验通过: $actual_md5"
        return 0
    else
        log_error "MD5校验失败! 期望: $MODEL_MD5, 实际: $actual_md5"
        return 1
    fi
}

# 解压模型
extract_model() {
    local archive_path="$TEMP_DIR/$MODEL_ARCHIVE"
    local extract_dir="$TEMP_DIR/extract"

    log_info "解压模型文件..."

    mkdir -p "$extract_dir"

    if ! tar -xzf "$archive_path" -C "$extract_dir"; then
        log_error "解压失败"
        return 1
    fi

    # 查找ckpt.pt文件
    local ckpt_file=$(find "$extract_dir" -name "ckpt.pt" -type f | head -1)

    if [[ -z "$ckpt_file" ]]; then
        log_error "解压后未找到ckpt.pt文件"
        return 1
    fi

    log_info "找到模型文件: $ckpt_file"

    # 移动到目标位置
    if ! mv "$ckpt_file" "$MODEL_FILE"; then
        log_error "移动模型文件失败"
        return 1
    fi

    log_info "模型文件已部署到: $MODEL_FILE"
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    rm -rf "$TEMP_DIR"
}

# 主函数
main() {
    log_info "=== CrystaLLM 模型管理器 ==="

    # 检查环境变量控制
    if [[ "${MODEL_AUTO_DOWNLOAD:-true}" != "true" ]]; then
        log_info "自动下载已禁用 (MODEL_AUTO_DOWNLOAD=false)"
        return 0
    fi

    # 检查是否强制重新下载
    if [[ "${MODEL_FORCE_REDOWNLOAD:-false}" == "true" ]]; then
        log_warn "强制重新下载模式"
        rm -f "$MODEL_FILE"
    fi

    # 检查是否已存在
    if check_model_exists; then
        log_info "模型已准备就绪，无需下载"
        return 0
    fi

    # 设置陷阱，确保清理
    trap cleanup EXIT

    # 执行下载流程
    setup_directories
    download_model

    # 可选跳过MD5检查
    if [[ "${MODEL_SKIP_MD5_CHECK:-false}" != "true" ]]; then
        verify_download
    else
        log_warn "跳过MD5校验"
    fi

    extract_model

    log_info "=== 模型准备完成 ==="
}

# 执行主函数
main "$@"
