#!/usr/bin/env python3
"""
中文字体显示测试脚本

用于验证matplotlib的中文字体配置是否正确
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 配置matplotlib支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def test_chinese_font():
    """测试中文字体显示"""
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(x, y1, 'b-', linewidth=2, label='正弦曲线')
    plt.plot(x, y2, 'r--', linewidth=2, label='余弦曲线')
    
    # 设置标题和标签（包含中文）
    plt.title('Bezier曲线路径规划 - 中文字体测试', fontsize=16, fontweight='bold')
    plt.xlabel('时间参数 (秒)', fontsize=12)
    plt.ylabel('幅值 (米)', fontsize=12)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加文本注释
    plt.text(5, 0.5, '这是中文测试文本\n字体显示正常！', 
             fontsize=14, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("中文字体测试完成！")
    print("如果图形中的中文显示正常，说明字体配置成功。")

if __name__ == "__main__":
    test_chinese_font() 