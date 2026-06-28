# MatGL MCP 服务器

基于 MatGL (Materials Graph Library) 的材料计算 MCP 服务器，提供晶体结构优化、性质预测等功能。

## 项目结构

```
matgl-mcp/
├── core/                    # 核心模块
│   ├── __init__.py
│   └── core.py             # LLM工具装饰器和错误处理
├── config/                  # 配置模块
│   ├── __init__.py
│   └── model_config.py     # 模型路径配置
├── tools/                   # 工具函数
│   ├── common_utils.py     # 公共工具函数
│   └── matgl_tools.py      # MatGL工具函数
├── models/                  # 预训练模型
│   ├── M3GNet-MP-2021.2.8-PES/
│   ├── M3GNet-MP-2018.6.1-Eform/
│   └── MEGNet-MP-2019.4.1-BandGap-mfi/
├── test_tools.py           # 测试脚本
└── README.md
```

## 功能特性

### 1. 结构优化
- **函数**: `relax_crystal_structure_MatGL`
- **功能**: 使用M3GNet通用势能优化晶体结构几何构型
- **参数**: CIF字符串、力收敛阈值

### 2. 形成能预测
- **函数**: `predict_formation_energy_MatGL`
- **功能**: 预测晶体结构的形成能
- **参数**: CIF字符串、是否优化结构、力收敛阈值

### 3. 分子动力学模拟
- **函数**: `run_molecular_dynamics_MatGL`
- **功能**: 运行分子动力学模拟
- **参数**: CIF字符串、温度、步数、是否优化结构、力收敛阈值

### 4. 单点能计算
- **函数**: `calculate_single_point_energy_MatGL`
- **功能**: 计算晶体结构的单点能
- **参数**: CIF字符串、是否优化结构、力收敛阈值

### 5. 多保真度带隙预测
- **函数**: `predict_multi_fidelity_band_gap_MatGL`
- **功能**: 使用多种DFT方法预测带隙（PBE、GLLB-SC、HSE、SCAN）
- **参数**: CIF字符串、是否优化结构、力收敛阈值

## 设计改进

### 1. 统一错误处理
- 所有错误信息使用统一格式：`Error：{具体错误信息}`
- 自动应用错误处理装饰器

### 2. 参数简化
- 将 `structure_source` 改为更直观的 `cif_string`
- 移除复杂的 `methods` 参数，默认使用所有DFT方法

### 3. 输出优化
- 使用结构化Markdown格式
- 移除冗余的原子位置表格
- 保留关键的结构信息和计算结果

### 4. 项目独立性
- 本地模型路径配置
- 不依赖外部配置文件
- 完整的项目目录结构

## 测试

运行测试脚本验证功能：

```bash
cd matgl-mcp
python test_tools.py
```

## 核心模块复用

`core/core.py` 模块可以复用到其他MCP项目中，提供：
- `@llm_tool` 装饰器
- 统一错误处理机制
- 工具函数注册和管理
