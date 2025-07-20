
# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dubins路径和固定翼无人机约束的交互式演示

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.widgets import Slider, Button
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DubinsDemo:
    """Dubins路径演示类"""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.3)
        
        # 初始参数
        self.min_radius = 50
        self.start_x = 100
        self.start_y = 100  
        self.start_angle = 0
        self.end_x = 400
        self.end_y = 300
        self.end_angle = 90
        
        # 创建滑块
        self._create_sliders()
        
        # 初始绘制
        self.update_plot()
        
    def _create_sliders(self):
        """创建控制滑块"""
        # 滑块位置
        ax_radius = plt.axes([0.1, 0.20, 0.3, 0.03])
        ax_start_x = plt.axes([0.1, 0.16, 0.3, 0.03])
        ax_start_y = plt.axes([0.1, 0.12, 0.3, 0.03])
        ax_start_angle = plt.axes([0.1, 0.08, 0.3, 0.03])
        ax_end_x = plt.axes([0.6, 0.20, 0.3, 0.03])
        ax_end_y = plt.axes([0.6, 0.16, 0.3, 0.03])
        ax_end_angle = plt.axes([0.6, 0.12, 0.3, 0.03])
        
        # 创建滑块
        self.slider_radius = Slider(ax_radius, '转弯半径', 20, 100, valinit=self.min_radius)
        self.slider_start_x = Slider(ax_start_x, '起点X', 0, 500, valinit=self.start_x)
        self.slider_start_y = Slider(ax_start_y, '起点Y', 0, 400, valinit=self.start_y)
        self.slider_start_angle = Slider(ax_start_angle, '起点角度', -180, 180, valinit=self.start_angle)
        self.slider_end_x = Slider(ax_end_x, '终点X', 0, 500, valinit=self.end_x)
        self.slider_end_y = Slider(ax_end_y, '终点Y', 0, 400, valinit=self.end_y)
        self.slider_end_angle = Slider(ax_end_angle, '终点角度', -180, 180, valinit=self.end_angle)
        
        # 连接事件
        self.slider_radius.on_changed(self.update_plot)
        self.slider_start_x.on_changed(self.update_plot)
        self.slider_start_y.on_changed(self.update_plot)
        self.slider_start_angle.on_changed(self.update_plot)
        self.slider_end_x.on_changed(self.update_plot)
        self.slider_end_y.on_changed(self.update_plot)
        self.slider_end_angle.on_changed(self.update_plot)
    
    def update_plot(self, val=None):
        """更新绘图"""
        # 获取当前参数
        self.min_radius = self.slider_radius.val
        self.start_x = self.slider_start_x.val
        self.start_y = self.slider_start_y.val
        self.start_angle = math.radians(self.slider_start_angle.val)
        self.end_x = self.slider_end_x.val
        self.end_y = self.slider_end_y.val
        self.end_angle = math.radians(self.slider_end_angle.val)
        
        # 清空并重绘
        self.ax.clear()
        self.ax.set_xlim(-50, 550)
        self.ax.set_ylim(-50, 450)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('固定翼无人机Dubins路径约束演示')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 绘制起点和终点
        self._draw_points()
        
        # 绘制转弯圆
        self._draw_turn_circles()
        
        # 绘制Dubins路径
        self._draw_dubins_path()
        
        # 绘制约束信息
        self._draw_constraints_info()
        
        # 刷新显示
        plt.draw()
    
    def _draw_points(self):
        """绘制起点和终点"""
        # 起点
        self.ax.plot(self.start_x, self.start_y, 'go', markersize=10, label='起点')
        
        # 起点方向箭头
        arrow_len = 30
        dx = arrow_len * math.cos(self.start_angle)
        dy = arrow_len * math.sin(self.start_angle)
        self.ax.arrow(self.start_x, self.start_y, dx, dy, 
                     head_width=8, head_length=6, fc='green', ec='green')
        
        # 终点
        self.ax.plot(self.end_x, self.end_y, 'rs', markersize=10, label='终点')
        
        # 终点方向箭头
        dx = arrow_len * math.cos(self.end_angle)
        dy = arrow_len * math.sin(self.end_angle)
        self.ax.arrow(self.end_x, self.end_y, dx, dy,
                     head_width=8, head_length=6, fc='red', ec='red')
    
    def _draw_turn_circles(self):
        """绘制转弯圆"""
        R = self.min_radius
        
        # 起点左右转弯圆心
        start_left_cx = self.start_x - R * math.sin(self.start_angle)
        start_left_cy = self.start_y + R * math.cos(self.start_angle)
        start_right_cx = self.start_x + R * math.sin(self.start_angle)
        start_right_cy = self.start_y - R * math.cos(self.start_angle)
        
        # 终点左右转弯圆心
        end_left_cx = self.end_x - R * math.sin(self.end_angle)
        end_left_cy = self.end_y + R * math.cos(self.end_angle)
        end_right_cx = self.end_x + R * math.sin(self.end_angle)
        end_right_cy = self.end_y - R * math.cos(self.end_angle)
        
        # 绘制转弯圆
        circles = [
            Circle((start_left_cx, start_left_cy), R, fill=False, color='blue', alpha=0.3, linestyle='--'),
            Circle((start_right_cx, start_right_cy), R, fill=False, color='blue', alpha=0.3, linestyle='--'),
            Circle((end_left_cx, end_left_cy), R, fill=False, color='red', alpha=0.3, linestyle='--'),
            Circle((end_right_cx, end_right_cy), R, fill=False, color='red', alpha=0.3, linestyle='--')
        ]
        
        for circle in circles:
            self.ax.add_patch(circle)
        
        # 标记圆心
        self.ax.plot([start_left_cx, start_right_cx], [start_left_cy, start_right_cy], 'bo', markersize=3)
        self.ax.plot([end_left_cx, end_right_cx], [end_left_cy, end_right_cy], 'ro', markersize=3)
    
    def _draw_dubins_path(self):
        """绘制简化的Dubins路径"""
        # 生成简化路径点
        num_points = 50
        path_x = []
        path_y = []
        
        for i in range(num_points + 1):
            t = i / num_points
            
            # 简单的插值路径
            x = self.start_x + t * (self.end_x - self.start_x)
            y = self.start_y + t * (self.end_y - self.start_y)
            
            # 添加曲率来模拟Dubins路径
            if 0.2 < t < 0.8:
                curve_offset = 20 * math.sin(math.pi * (t - 0.2) / 0.6)
                # 计算垂直方向
                dx = self.end_x - self.start_x
                dy = self.end_y - self.start_y
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = -dy / length
                    perp_y = dx / length
                    x += curve_offset * perp_x
                    y += curve_offset * perp_y
            
            path_x.append(x)
            path_y.append(y)
        
        # 绘制路径
        self.ax.plot(path_x, path_y, 'purple', linewidth=3, label='Dubins路径')
        
        # 添加方向箭头
        for i in range(0, len(path_x) - 5, 10):
            dx = path_x[i+5] - path_x[i]
            dy = path_y[i+5] - path_y[i]
            self.ax.arrow(path_x[i], path_y[i], dx, dy,
                         head_width=3, head_length=2, fc='purple', ec='purple', alpha=0.7)
        
        self.ax.legend()
    
    def _draw_constraints_info(self):
        """显示约束信息"""
        # 计算一些约束相关信息
        speed = 25  # m/s
        bank_angle = 45  # degrees
        g = 9.81
        
        theoretical_radius = speed**2 / (g * math.tan(math.radians(bank_angle)))
        
        # 计算路径长度
        path_length = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2) * 1.2  # 估算
        flight_time = path_length / speed
        
        info_text = f"""约束参数:
最小转弯半径: {self.min_radius:.1f} m
理论计算值: {theoretical_radius:.1f} m
(V=25m/s, φ=45°)

路径信息:
估算长度: {path_length:.1f} m
飞行时间: {flight_time:.1f} s
航向变化: {abs(math.degrees(self.end_angle - self.start_angle)):.1f}°"""
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)

def main():
    """主函数"""
    print("启动Dubins路径约束交互演示...")
    
    demo = DubinsDemo()
    
    plt.figtext(0.5, 0.02, 
                "拖动滑块调整参数，观察Dubins路径如何满足固定翼无人机的转弯半径约束",
                ha='center', fontsize=10, style='italic')
    
    plt.show()

if __name__ == "__main__":
    main() 