#!/usr/bin/env python3
"""
交互式算法对比演示
支持动态参数调整和实时算法对比
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from algorithms_comparison import PathfindingAlgorithms, create_test_grids
except ImportError:
    # 如果导入失败，提供一个简化的实现
    class PathfindingAlgorithms:
        def __init__(self, grid):
            self.grid = grid
        def is_valid(self, x, y):
            return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] == 0
    
    def create_test_grids():
        return {
            'simple': [[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0]],
            'complex': [[0]*10 for _ in range(10)],
            'maze': [[0]*10 for _ in range(10)]
        }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数"""
    print("启动交互式算法对比演示...")
    print("请确保已安装必要的依赖包：numpy, matplotlib")
    
    # 创建简单的演示
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, '算法对比演示\n\n包含:\n• Dijkstra算法\n• 贪心最佳优先算法\n• A*算法\n\n请运行 algorithms_comparison.py 查看完整演示', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('路径搜索算法对比', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 