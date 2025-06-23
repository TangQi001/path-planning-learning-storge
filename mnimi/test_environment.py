#!/usr/bin/env python3
"""
B-样条曲线 Manim 演示 - 环境测试脚本

用于验证所有必要的依赖是否正确安装，以及 Manim 是否能正常工作。
"""

import sys
import subprocess
from typing import List, Tuple


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """测试模块导入"""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name} 导入成功"
    except ImportError as e:
        return False, f"✗ {package_name or module_name} 导入失败: {e}"


def test_manim_command() -> Tuple[bool, str]:
    """测试 Manim 命令行工具"""
    try:
        result = subprocess.run(
            ["manim", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"✓ Manim 命令行工具可用: {version}"
        else:
            return False, f"✗ Manim 命令行工具错误: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "✗ Manim 命令超时"
    except FileNotFoundError:
        return False, "✗ 找不到 manim 命令，请确保 Manim 已正确安装"
    except Exception as e:
        return False, f"✗ Manim 命令测试失败: {e}"


def test_scipy_bspline() -> Tuple[bool, str]:
    """测试 SciPy B-spline 功能"""
    try:
        from scipy.interpolate import BSpline
        import numpy as np
        
        # 创建一个简单的 B-spline 测试
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        coeffs = [[1, 0], [0, 1], [1, 1], [0, 0]]
        bspline = BSpline(knots, coeffs, 2)
        
        # 测试求值
        result = bspline(1.5)
        
        return True, "✓ SciPy B-spline 功能正常"
    except Exception as e:
        return False, f"✗ SciPy B-spline 测试失败: {e}"


def test_manim_basic_scene() -> Tuple[bool, str]:
    """测试基本的 Manim 场景创建"""
    try:
        from manim import Scene, Circle, Text
        import numpy as np
        
        # 创建一个简单的测试场景
        class TestScene(Scene):
            def construct(self):
                circle = Circle()
                text = Text("Test")
        
        # 尝试实例化场景
        scene = TestScene()
        
        return True, "✓ Manim 基本场景创建成功"
    except Exception as e:
        return False, f"✗ Manim 基本场景创建失败: {e}"


def main():
    """主测试函数"""
    print("=" * 60)
    print("B-样条曲线 Manim 演示 - 环境测试")
    print("=" * 60)
    
    # 测试项目列表
    tests = [
        # 基础 Python 模块
        ("numpy", "NumPy"),
        ("scipy", "SciPy"), 
        ("scipy.interpolate", "SciPy 插值模块"),
        ("matplotlib", "Matplotlib"),
        
        # Manim 相关
        ("manim", "Manim"),
        
        # 可选模块
        ("PIL", "Pillow"),
    ]
    
    print("\n📦 检查 Python 模块导入...")
    print("-" * 40)
    
    failed_imports = []
    for module, name in tests:
        success, message = test_import(module, name)
        print(message)
        if not success:
            failed_imports.append(name)
    
    print("\n🔧 检查 Manim 命令行工具...")
    print("-" * 40)
    manim_success, manim_message = test_manim_command()
    print(manim_message)
    
    print("\n🧮 检查 SciPy B-spline 功能...")
    print("-" * 40)
    bspline_success, bspline_message = test_scipy_bspline()
    print(bspline_message)
    
    print("\n🎬 检查 Manim 场景创建...")
    print("-" * 40)
    scene_success, scene_message = test_manim_basic_scene()
    print(scene_message)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if not failed_imports and manim_success and bspline_success and scene_success:
        print("🎉 所有测试通过！环境设置正确。")
        print("\n✅ 您可以运行以下命令开始演示：")
        print("   python run_bspline_demo.py --preview")
        return 0
    else:
        print("❌ 存在问题，需要解决以下问题：")
        
        if failed_imports:
            print(f"\n📦 缺少模块: {', '.join(failed_imports)}")
            print("   解决方案: pip install -r requirements.txt")
        
        if not manim_success:
            print("\n🔧 Manim 命令行工具问题")
            print("   解决方案: 重新安装 Manim 或检查 PATH 环境变量")
        
        if not bspline_success:
            print("\n🧮 SciPy B-spline 功能问题")
            print("   解决方案: pip install --upgrade scipy")
        
        if not scene_success:
            print("\n🎬 Manim 场景创建问题")
            print("   解决方案: 检查 Manim 安装和系统依赖")
        
        print("\n📚 更多帮助:")
        print("   - Manim 安装指南: https://docs.manim.community/en/stable/installation.html")
        print("   - 项目 README: 查看 README.md 文件")
        
        return 1


if __name__ == "__main__":
    sys.exit(main()) 