#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euler螺旋曲线（缓和曲线）基础计算和可视化

作者：AI助手
日期：2025年6月
功能：
1. 计算Euler螺旋曲线坐标
2. 可视化曲线形状
3. 分析曲率变化特性
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EulerSpiral:
    """Euler螺旋曲线类"""
    
    def __init__(self, A=1.0):
        """
        初始化Euler螺旋曲线
        
        参数:
        A: 缓和曲线参数 (clothoid parameter)
        """
        self.A = A
        
    def compute_coordinates(self, t_max=3.0, num_points=1000):
        """
        计算Euler螺旋曲线坐标
        
        参数:
        t_max: 参数t的最大值
        num_points: 计算点数
        
        返回:
        t, x, y: 参数数组和对应的x, y坐标
        """
        t = np.linspace(-t_max, t_max, num_points)
        
        # 使用scipy的fresnel函数计算坐标
        S, C = fresnel(t * np.sqrt(2/np.pi))
        
        # 缩放到正确的参数A
        x = self.A * C * np.sqrt(np.pi/2)
        y = self.A * S * np.sqrt(np.pi/2)
        
        return t, x, y
    
    def compute_curvature(self, t):
        """
        计算给定参数t处的曲率
        
        参数:
        t: 参数值或数组
        
        返回:
        κ: 曲率值
        """
        # 对于标准化的Euler螺旋：κ = t
        return t / self.A
    
    def compute_arc_length(self, t):
        """
        计算从原点到参数t的弧长
        
        参数:
        t: 参数值
        
        返回:
        s: 弧长
        """
        return self.A * abs(t)
    
    def plot_spiral(self, t_max=3.0, num_points=1000, figsize=(12, 10)):
        """
        绘制Euler螺旋曲线及其特性
        
        参数:
        t_max: 参数范围
        num_points: 绘图点数
        figsize: 图形大小
        """
        # 计算坐标
        t, x, y = self.compute_coordinates(t_max, num_points)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Euler螺旋曲线分析 (A = {self.A})', fontsize=16, fontweight='bold')
        
        # 1. 主曲线图
        ax1.plot(x, y, 'b-', linewidth=2, label='Euler螺旋')
        ax1.plot(0, 0, 'ro', markersize=8, label='起点')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title('Euler螺旋曲线形状')
        ax1.legend()
        ax1.axis('equal')
        
        # 添加方向箭头
        for i in range(0, len(x), len(x)//10):
            if i < len(x)-1:
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                ax1.arrow(x[i], y[i], dx*10, dy*10, 
                         head_width=0.1, head_length=0.1, 
                         fc='red', ec='red', alpha=0.7)
        
        # 2. 曲率变化图
        t_curvature = np.linspace(-t_max, t_max, 200)
        curvature = self.compute_curvature(t_curvature)
        ax2.plot(t_curvature, curvature, 'g-', linewidth=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('参数 t')
        ax2.set_ylabel('曲率 κ')
        ax2.set_title('曲率随参数变化')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. 弧长关系图
        t_positive = np.linspace(0, t_max, 100)
        arc_lengths = [self.compute_arc_length(t_val) for t_val in t_positive]
        ax3.plot(t_positive, arc_lengths, 'purple', linewidth=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('参数 t')
        ax3.set_ylabel('弧长 s')
        ax3.set_title('弧长随参数变化')
        
        # 4. 参数坐标图
        ax4.plot(t, x, 'r-', linewidth=2, label='X(t)')
        ax4.plot(t, y, 'b-', linewidth=2, label='Y(t)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('参数 t')
        ax4.set_ylabel('坐标值')
        ax4.set_title('坐标随参数变化')
        ax4.legend()
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig

def compare_different_parameters():
    """比较不同参数A的Euler螺旋"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 不同的A值
    A_values = [0.5, 1.0, 1.5, 2.0]
    colors = ['red', 'blue', 'green', 'orange']
    
    for A, color in zip(A_values, colors):
        spiral = EulerSpiral(A)
        t, x, y = spiral.compute_coordinates(3.0, 500)
        
        # 绘制曲线
        ax1.plot(x, y, color=color, linewidth=2, label=f'A = {A}')
        
        # 绘制曲率
        t_range = np.linspace(-3, 3, 200)
        curvature = spiral.compute_curvature(t_range)
        ax2.plot(t_range, curvature, color=color, linewidth=2, label=f'A = {A}')
    
    # 设置第一个子图
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.set_title('不同参数A的Euler螺旋对比')
    ax1.legend()
    ax1.axis('equal')
    
    # 设置第二个子图
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('参数 t')
    ax2.set_ylabel('曲率 κ')
    ax2.set_title('不同参数A的曲率对比')
    ax2.legend()
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def road_application_demo():
    """道路应用演示"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 模拟道路设计：直线 -> 缓和曲线 -> 圆弧 -> 缓和曲线 -> 直线
    
    # 设计参数
    R = 100  # 圆弧半径 (m)
    L = 50   # 缓和曲线长度 (m)
    A = np.sqrt(R * L)  # 缓和曲线参数
    
    print(f"道路设计参数：")
    print(f"圆弧半径 R = {R} m")
    print(f"缓和曲线长度 L = {L} m")
    print(f"缓和曲线参数 A = {A:.2f} m")
    
    # 创建缓和曲线
    spiral = EulerSpiral(A)
    
    # 计算缓和曲线参数范围
    t_max = np.sqrt(2 * L / A)  # 对应缓和曲线长度L
    t_range = np.linspace(0, t_max, 100)
    
    # 计算缓和曲线坐标
    S, C = fresnel(t_range * np.sqrt(2/np.pi))
    x_spiral = A * C * np.sqrt(np.pi/2)
    y_spiral = A * S * np.sqrt(np.pi/2)
    
    # 绘制入口缓和曲线
    x_offset = -L
    ax.plot(x_offset + x_spiral, y_spiral, 'b-', linewidth=3, label='入口缓和曲线')
    
    # 绘制圆弧部分
    theta_start = y_spiral[-1] / R  # 缓和曲线终点的切线角
    theta_arc = np.linspace(theta_start, np.pi/3, 100)  # 60度圆弧
    x_arc = x_offset + x_spiral[-1] + R * (np.sin(theta_arc) - np.sin(theta_start))
    y_arc = y_spiral[-1] + R * (np.cos(theta_start) - np.cos(theta_arc))
    ax.plot(x_arc, y_arc, 'r-', linewidth=3, label='圆弧段')
    
    # 绘制出口缓和曲线（镜像）
    x_exit = x_arc[-1] + x_spiral[::-1] * np.cos(theta_arc[-1]) - y_spiral[::-1] * np.sin(theta_arc[-1])
    y_exit = y_arc[-1] + x_spiral[::-1] * np.sin(theta_arc[-1]) + y_spiral[::-1] * np.cos(theta_arc[-1])
    ax.plot(x_exit, y_exit, 'g-', linewidth=3, label='出口缓和曲线')
    
    # 绘制直线段
    ax.plot([-200, x_offset], [0, 0], 'k-', linewidth=3, label='直线段')
    ax.plot([x_exit[-1], x_exit[-1] + 100], [y_exit[-1], y_exit[-1]], 'k-', linewidth=3)
    
    # 添加标注
    ax.annotate('直线段', xy=(-150, 5), fontsize=12, ha='center')
    ax.annotate('缓和曲线', xy=(-25, 10), fontsize=12, ha='center')
    ax.annotate(f'圆弧 R={R}m', xy=(30, 40), fontsize=12, ha='center')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('距离 (m)')
    ax.set_title('道路缓和曲线应用示例')
    ax.legend()
    ax.axis('equal')
    
    return fig

def main():
    """主函数"""
    print("=" * 60)
    print("Euler螺旋曲线（缓和曲线）分析程序")
    print("=" * 60)
    
    # 1. 基础螺旋分析
    print("\n1. 创建基础Euler螺旋曲线...")
    spiral = EulerSpiral(A=1.0)
    fig1 = spiral.plot_spiral()
    fig1.savefig('../images/euler_spiral_basic.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: ../images/euler_spiral_basic.png")
    
    # 2. 参数对比
    print("\n2. 不同参数对比...")
    fig2 = compare_different_parameters()
    fig2.savefig('../images/euler_spiral_comparison.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: ../images/euler_spiral_comparison.png")
    
    # 3. 道路应用演示
    print("\n3. 道路应用演示...")
    fig3 = road_application_demo()
    fig3.savefig('../images/road_application_demo.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: ../images/road_application_demo.png")
    
    # 4. 数值计算示例
    print("\n4. 数值计算示例:")
    print("-" * 40)
    
    A = 100  # 缓和曲线参数
    spiral = EulerSpiral(A)
    
    # 计算几个关键点的值
    t_values = [0, 0.5, 1.0, 1.5, 2.0]
    print(f"{'参数t':>8} {'弧长s':>10} {'曲率κ':>10} {'X坐标':>10} {'Y坐标':>10}")
    print("-" * 50)
    
    for t in t_values:
        s = spiral.compute_arc_length(t)
        kappa = spiral.compute_curvature(t)
        
        # 计算坐标
        S_val, C_val = fresnel(t * np.sqrt(2/np.pi))
        x = A * C_val * np.sqrt(np.pi/2)
        y = A * S_val * np.sqrt(np.pi/2)
        
        print(f"{t:8.1f} {s:10.2f} {kappa:10.4f} {x:10.2f} {y:10.2f}")
    
    print("\n程序运行完成！")
    print("请查看生成的图形文件以了解Euler螺旋曲线的特性。")
    
    # 显示图形（如果在交互环境中）
    plt.show()

if __name__ == "__main__":
    main() 