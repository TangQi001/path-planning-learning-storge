#!/usr/bin/env python3
"""
B-样条曲线演示启动脚本
从主目录启动 mnimi 子目录中的演示
"""

import os
import sys
import subprocess

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # mnimi 子目录路径
    mnimi_dir = os.path.join(current_dir, "mnimi")
    
    # 运行脚本路径
    run_script = os.path.join(mnimi_dir, "run_bspline_demo.py")
    
    # 检查文件是否存在
    if not os.path.exists(run_script):
        print(f"✗ 找不到运行脚本: {run_script}")
        print("请确保 mnimi/run_bspline_demo.py 文件存在")
        sys.exit(1)
    
    print("🚀 启动 B-样条曲线演示...")
    print(f"工作目录: {mnimi_dir}")
    
    # 构建命令
    cmd = [sys.executable, run_script] + sys.argv[1:]
    
    try:
        # 在 mnimi 目录中运行
        result = subprocess.run(cmd, cwd=mnimi_dir)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 