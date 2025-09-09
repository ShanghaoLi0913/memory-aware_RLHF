#!/usr/bin/env python3
"""
核心依赖安装脚本 - 只安装RQ2实验必需的包
避免复杂的依赖冲突，专注于核心功能
"""
import subprocess
import sys
import os


def run_pip_install(packages, description):
    """安装pip包"""
    print(f"📦 {description}...")
    
    for package in packages:
        print(f"  安装 {package}...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  check=True, capture_output=True, text=True)
            print(f"  ✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {package} 安装失败: {e.stderr}")
            return False
    return True


def main():
    """主安装流程"""
    print("🚀 安装RQ2实验核心依赖包")
    print("="*40)
    
    # 检查环境
    if os.environ.get('CONDA_DEFAULT_ENV') != 'MARLHF':
        print("❌ 请确保在MARLHF环境中运行")
        return False
    
    print(f"✅ 当前环境: {os.environ.get('CONDA_DEFAULT_ENV')}")
    
    # 1. 核心科学计算包
    core_packages = [
        "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "tqdm"
    ]
    if not run_pip_install(core_packages, "核心科学计算包"):
        return False
    
    # 2. PyTorch (根据Python版本选择兼容版本)
    python_version = sys.version_info
    print(f"📦 安装PyTorch (Python {python_version.major}.{python_version.minor})...")
    
    if python_version >= (3, 9):
        # Python 3.9+ 使用最新版本
        torch_cmd = [sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cpu"]
    else:
        # Python 3.8 使用兼容版本
        torch_cmd = [sys.executable, "-m", "pip", "install", 
                    "torch==1.13.1", "torchvision==0.14.1", "torchaudio==0.13.1",
                    "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    try:
        result = subprocess.run(torch_cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch 安装成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch 安装失败: {e.stderr}")
        # 尝试不指定index-url
        try:
            simple_cmd = [sys.executable, "-m", "pip", "install", 
                         "torch==1.13.1", "torchvision==0.14.1", "torchaudio==0.13.1"]
            result = subprocess.run(simple_cmd, check=True, capture_output=True, text=True)
            print("✅ PyTorch 安装成功 (使用备选方案)")
        except subprocess.CalledProcessError as e2:
            print(f"❌ PyTorch 备选安装也失败: {e2.stderr}")
            return False
    
    # 3. Transformers生态系统 (兼容Python 3.8)
    if python_version >= (3, 9):
        transformers_packages = [
            "transformers>=4.30.0", "datasets>=2.12.0", "accelerate", "tokenizers"
        ]
    else:
        # Python 3.8 兼容版本
        transformers_packages = [
            "transformers==4.21.0", "datasets==2.5.0", "tokenizers", "accelerate"
        ]
    
    if not run_pip_install(transformers_packages, "Transformers生态系统"):
        return False
    
    # 4. 实验必需包
    experiment_packages = [
        "jsonlines", "rouge-score", "nltk"
    ]
    if not run_pip_install(experiment_packages, "实验工具"):
        return False
    
    # 5. 可选的API包 (如果需要)
    optional_packages = [
        "openai", "tiktoken"
    ]
    print("📦 安装可选API包...")
    for package in optional_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                         check=True, capture_output=True, text=True)
            print(f"  ✅ {package} 安装成功")
        except:
            print(f"  ⚠️ {package} 安装失败，跳过...")
    
    # 验证安装
    print("\n🔍 验证核心包...")
    try:
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        
        print("✅ 核心包导入成功")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Transformers: {transformers.__version__}")
        print(f"  Datasets: {datasets.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 核心依赖安装完成！")
        print("\n📋 下一步:")
        print("1. 测试数据加载: python experiments/test_rq2_framework.py")
        print("2. 如需要更多包，可单独安装: pip install package_name")
    else:
        print("\n❌ 安装过程中出现错误")
    
    sys.exit(0 if success else 1)
