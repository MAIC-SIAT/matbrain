# MatterGen-MCP Docker 部署

这是 MatterGen-MCP 项目的 Docker 化版本，使用 CUDA 基础镜像支持 GPU 加速的材料生成。

## 🚀 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA Docker 支持 (nvidia-docker2 或 nvidia-container-toolkit)
- 至少 8GB GPU 内存
- 支持 CUDA 11.8 的 NVIDIA GPU

### 构建和运行

1. **克隆项目并进入目录**
   ```bash
   cd mattergen-mcp
   ```

2. **检查配置文件**
   ```bash
   cat .env
   ```
   确保 `HOST_MODEL_DIR` 指向正确的模型文件路径。

3. **构建 Docker 镜像**
   ```bash
   ./build.sh
   ```

4. **运行服务**
   ```bash
   ./build.sh --run
   ```
   或者
   ```bash
   docker-compose up -d
   ```

5. **检查服务状态**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

6. **访问服务**
   - 服务地址: http://localhost:5670
   - 健康检查: http://localhost:5670/health
   - SSE 端点: http://localhost:5670/sse

## 📁 项目结构

```
mattergen-mcp/
├── Dockerfile              # Docker 镜像定义
├── docker-compose.yml      # Docker Compose 配置
├── .env                    # 环境变量配置
├── .dockerignore          # Docker 构建忽略文件
├── build.sh               # 构建脚本
├── requirements.txt       # Python 依赖
├── server.py              # MCP 服务器主文件
├── mattergen/             # MatterGen 核心库
├── mattergen_ckpt/        # 模型检查点文件
├── config/                # 配置模块
├── core/                  # 核心功能模块
└── tools/                 # 工具函数模块
```

## ⚙️ 配置说明

### 环境变量 (.env)

```bash
# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=5670
HOST_PORT=5670

# 模型路径配置
HOST_MODEL_DIR=./mattergen_ckpt
MATTERGEN_ROOT=/app/mattergen
MATTERGENMODEL_ROOT=/app/mattergen_ckpt

# 生成参数
MATTERGEN_BATCH_SIZE=2
MATTERGEN_NUM_BATCHES=1
MATTERGEN_GUIDANCE_FACTOR=2.0
MATTERGEN_DEVICE=cuda

# 其他配置
TOOL_CALL_TIMEOUT=360
LOG_LEVEL=INFO
```

### GPU 配置

项目支持多 GPU 配置，不同模型会自动分配到不同的 GPU：

- GPU 0-7: 支持 8 个 GPU 并行处理
- 自动 GPU 映射: 不同的 MatterGen 模型使用不同的 GPU

## 🔧 构建脚本使用

```bash
# 显示帮助信息
./build.sh --help

# 标准构建
./build.sh

# 清理后构建并运行
./build.sh --clean --run

# 无缓存构建
./build.sh --no-cache

# 拉取最新基础镜像后构建
./build.sh --pull
```

## 🛠️ 可用工具

MatterGen-MCP 提供以下材料生成工具：

1. **无条件生成**
   - `generate_material_unconditional_MatterGen`

2. **单属性条件生成**
   - `generate_material_by_dft_band_gap_MatterGen`
   - `generate_material_by_chemical_system_MatterGen`
   - `generate_material_by_space_group_MatterGen`
   - `generate_material_by_formation_energy_MatterGen`
   - `generate_material_by_bulk_modulus_MatterGen`

3. **多属性条件生成**
   - `generate_material_by_band_gap_and_space_group_MatterGen`
   - `generate_material_by_chemical_system_and_formation_energy_MatterGen`
   - `generate_material_by_custom_properties_MatterGen`

## 📊 监控和日志

### 查看日志
```bash
# 查看实时日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs mattergen-mcp

# 查看最近的日志
docker-compose logs --tail=100 mattergen-mcp
```

### 健康检查
```bash
# 检查容器状态
docker-compose ps

# 手动健康检查
curl http://localhost:5670/health
```

## 🔍 故障排除

### 常见问题

1. **GPU 不可用**
   ```bash
   # 检查 NVIDIA Docker 支持
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **模型文件缺失**
   ```bash
   # 检查模型目录
   ls -la ./mattergen_ckpt
   ```

3. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 5670
   ```

4. **内存不足**
   ```bash
   # 检查 GPU 内存
   nvidia-smi
   ```

### 重新构建

如果遇到问题，可以完全重新构建：

```bash
# 清理并重新构建
./build.sh --clean --no-cache --run
```

## 📝 开发说明

### 修改配置

1. 修改 `.env` 文件中的环境变量
2. 重新启动容器：
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### 更新代码

1. 修改代码后重新构建镜像：
   ```bash
   ./build.sh --clean
   ```

2. 启动新容器：
   ```bash
   docker-compose up -d
   ```

## 🔒 安全配置

- 容器以非 root 用户运行
- 只读文件系统（除临时目录）
- 网络隔离
- 最小权限原则

## 📄 许可证

请参考 MatterGen 项目的原始许可证。

## �� 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请查看：
1. 项目日志: `docker-compose logs -f`
2. 健康检查: `curl http://localhost:5670/health`
3. GPU 状态: `nvidia-smi`
