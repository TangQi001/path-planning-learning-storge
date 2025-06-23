#!/usr/bin/env python3
"""
B-样条曲线演示视频运行脚本

使用方法：
1. 确保已安装 manim 和 scipy：
   pip install manim scipy

2. 运行完整演示：
   python run_bspline_demo.py

3. 运行单个场景：
   python run_bspline_demo.py --scene BSplineDemo
   python run_bspline_demo.py --scene BSplineBasicFunctions
   python run_bspline_demo.py --scene BSplineInteractive

4. 高质量渲染：
   python run_bspline_demo.py --quality high

5. 预览模式（低质量，快速渲染）：
   python run_bspline_demo.py --preview
"""

import subprocess
import sys
import argparse


def check_dependencies():
    """检查必要的依赖包"""
    try:
        import manim
        import scipy
        import numpy
        print("✓ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install manim scipy numpy")
        return False


def run_manim_command(scene_name=None, quality="medium", preview=False):
    """构建并运行 manim 命令"""
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bspline_file = os.path.join(script_dir, "bspline_demo.py")
    
    # 检查文件是否存在
    if not os.path.exists(bspline_file):
        print(f"✗ 找不到文件: {bspline_file}")
        return False
    
    # 基础命令
    cmd = ["manim"]
    
    # 质量设置
    if preview:
        cmd.extend(["-p", "-ql"])  # 预览模式，低质量
    elif quality == "high":
        cmd.extend(["-p", "-qh"])  # 高质量
    elif quality == "medium":
        cmd.extend(["-p", "-qm"])  # 中等质量
    else:
        cmd.extend(["-p", "-ql"])  # 默认低质量
    
    # 文件名（使用绝对路径）
    cmd.append(bspline_file)
    
    # 场景名称
    if scene_name:
        cmd.append(scene_name)
    else:
        cmd.append("BSplineDemo")  # 默认场景
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"文件路径: {bspline_file}")
    
    try:
        # 在脚本目录中运行命令
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("✓ 视频渲染完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 渲染失败: {e}")
        return False
    except FileNotFoundError:
        print("✗ 找不到 manim 命令。请确保 manim 已正确安装。")
        print("安装命令: pip install manim")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="B-样条曲线演示视频渲染工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--scene", 
        choices=["BSplineDemo", "BSplineBasicFunctions", "BSplineInteractive", "BSplineComplete"],
        default="BSplineDemo",
        help="选择要渲染的场景"
    )
    
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="渲染质量"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="预览模式（低质量，快速渲染）"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="仅检查依赖包"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("B-样条曲线 Manim 演示视频")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("依赖检查完成！")
        return
    
    print(f"场景: {args.scene}")
    print(f"质量: {args.quality}")
    if args.preview:
        print("模式: 预览")
    
    # 运行渲染
    success = run_manim_command(
        scene_name=args.scene,
        quality=args.quality,
        preview=args.preview
    )
    
    if success:
        print("\n🎉 渲染成功！视频文件保存在 media/videos/ 目录中")
        print("\n场景说明：")
        print("- BSplineDemo: 基础B-样条演示，包含控制点、曲线生成和交互")
        print("- BSplineBasicFunctions: B-样条基函数的可视化")
        print("- BSplineInteractive: 动态交互式B-样条演示")
        print("- BSplineComplete: 完整演示（包含所有场景）")
    else:
        print("\n❌ 渲染失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main() 