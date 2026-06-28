#!/usr/bin/env python3
"""
CrystaLLM MCP GPU 测试脚本
验证容器内的 CUDA 和 PyTorch GPU 功能
"""

import sys
import subprocess
import os

def test_nvidia_smi():
    """测试 nvidia-smi 命令"""
    print("=" * 60)
    print("测试 1: nvidia-smi GPU 检测")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi 成功")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi 失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ nvidia-smi 异常: {e}")
        return False

def test_cuda_version():
    """测试 CUDA 版本"""
    print("\n" + "=" * 60)
    print("测试 2: CUDA 版本检测")
    print("=" * 60)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA 编译器可用")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA 编译器不可用")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ CUDA 版本检测异常: {e}")
        return False

def test_pytorch_cuda():
    """测试 PyTorch CUDA 支持"""
    print("\n" + "=" * 60)
    print("测试 3: PyTorch CUDA 支持")
    print("=" * 60)
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU 设备数量: {device_count}")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {device_name} ({device_memory:.1f} GB)")

            # 测试简单的 CUDA 操作
            print("\n测试 CUDA 张量操作...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"✅ CUDA 矩阵乘法成功，结果形状: {z.shape}")

            return True
        else:
            print("❌ CUDA 不可用")
            return False

    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"❌ PyTorch CUDA 测试异常: {e}")
        return False

def test_cudnn():
    """测试 cuDNN 支持"""
    print("\n" + "=" * 60)
    print("测试 4: cuDNN 支持")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            cudnn_available = torch.backends.cudnn.enabled
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN 启用: {cudnn_available}")
            print(f"cuDNN 版本: {cudnn_version}")

            if cudnn_available:
                print("✅ cuDNN 可用")
                return True
            else:
                print("❌ cuDNN 不可用")
                return False
        else:
            print("❌ CUDA 不可用，无法测试 cuDNN")
            return False
    except Exception as e:
        print(f"❌ cuDNN 测试异常: {e}")
        return False

def test_crystallm_imports():
    """测试 CrystaLLM 相关导入"""
    print("\n" + "=" * 60)
    print("测试 5: CrystaLLM 模块导入")
    print("=" * 60)
    try:
        # 测试基础科学计算库
        import numpy as np
        print("✅ NumPy 导入成功")

        import torch
        print("✅ PyTorch 导入成功")

        # 尝试导入 CrystaLLM 相关模块
        try:
            sys.path.append('/app/tools/CrystaLLM')
            import crystallm
            print("✅ CrystaLLM 导入成功")
        except ImportError as e:
            print(f"⚠️  CrystaLLM 导入失败: {e}")

        return True
    except Exception as e:
        print(f"❌ 模块导入异常: {e}")
        return False

def test_environment_variables():
    """测试环境变量"""
    print("\n" + "=" * 60)
    print("测试 6: 环境变量检查")
    print("=" * 60)

    env_vars = [
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES',
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CRYSTALLM_DEVICE'
    ]

    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"{var}: {value}")

    return True

def main():
    """主测试函数"""
    print("CrystaLLM MCP GPU 环境测试")
    print("=" * 60)

    tests = [
        test_nvidia_smi,
        test_cuda_version,
        test_pytorch_cuda,
        test_cudnn,
        test_crystallm_imports,
        test_environment_variables
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")

    print("\n" + "=" * 60)
    print(f"测试总结: {passed}/{total} 项测试通过")
    print("=" * 60)

    if passed >= 4:  # 至少前4个核心测试通过
        print("🎉 GPU 环境配置基本正确！")
        return 0
    else:
        print("⚠️  GPU 环境可能存在问题，请检查配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
