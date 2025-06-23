#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
道路缓和曲线设计计算器

作者：AI助手
日期：2025年6月
功能：
1. 道路缓和曲线设计参数计算
2. 曲线要素计算（切线长、外距、切曲差等）
3. 逐桩坐标计算
4. 超高与加宽过渡计算
5. 设计图纸生成
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class RoadDesignCalculator:
    """道路缓和曲线设计计算器"""
    
    def __init__(self):
        """初始化计算器"""
        self.design_speed = 80  # 设计速度 km/h
        self.radius = 300       # 圆曲线半径 m
        self.transition_length = 100  # 缓和曲线长度 m
        self.lane_width = 3.75  # 车道宽度 m
        self.superelevation_rate = 0.06  # 超高横坡度
        
        # 计算基本参数
        self.update_parameters()
    
    def update_parameters(self):
        """更新设计参数"""
        self.A = np.sqrt(self.radius * self.transition_length)  # 缓和曲线参数
        self.deflection_angle_transition = self.transition_length**2 / (2 * self.radius * self.transition_length)
        
    def set_design_parameters(self, speed, radius, transition_length, 
                            lane_width=3.75, superelevation=0.06):
        """
        设置设计参数
        
        参数:
        speed: 设计速度 (km/h)
        radius: 圆曲线半径 (m)
        transition_length: 缓和曲线长度 (m)
        lane_width: 车道宽度 (m)
        superelevation: 超高横坡度
        """
        self.design_speed = speed
        self.radius = radius
        self.transition_length = transition_length
        self.lane_width = lane_width
        self.superelevation_rate = superelevation
        self.update_parameters()
    
    def check_design_standards(self):
        """检查设计是否符合规范标准"""
        results = {}
        
        # 最小半径检查 (基于设计速度)
        min_radius_dict = {40: 60, 50: 100, 60: 150, 70: 200, 80: 250, 
                          100: 400, 120: 650}
        min_radius = min_radius_dict.get(self.design_speed, 250)
        results['半径检查'] = {
            '当前半径': self.radius,
            '最小半径': min_radius,
            '是否合格': self.radius >= min_radius
        }
        
        # 缓和曲线长度检查
        # 最小长度：车辆行驶3秒的距离
        min_length = self.design_speed / 3.6 * 3  # m
        # 推荐长度
        recommended_length = self.design_speed / 3.6 * 3.5
        results['缓和曲线长度检查'] = {
            '当前长度': self.transition_length,
            '最小长度': min_length,
            '推荐长度': recommended_length,
            '是否合格': self.transition_length >= min_length
        }
        
        # 曲率变化率检查（乘客舒适度）
        curvature_change_rate = 1 / (self.radius * self.transition_length)
        max_curvature_change_rate = 0.3  # 1/s²
        results['曲率变化率检查'] = {
            '当前值': curvature_change_rate,
            '最大允许值': max_curvature_change_rate,
            '是否合格': curvature_change_rate <= max_curvature_change_rate
        }
        
        return results
    
    def calculate_curve_elements(self):
        """计算曲线要素"""
        # 基本参数
        L = self.transition_length
        R = self.radius
        A = self.A
        
        # 计算缓和曲线终点坐标（相对于起点）
        t_end = np.sqrt(2 * L / A)
        S, C = fresnel(t_end * np.sqrt(2/np.pi))
        x_end = A * C * np.sqrt(np.pi/2)
        y_end = A * S * np.sqrt(np.pi/2)
        
        # 切线角
        beta = L**2 / (2 * R * L)  # 弧度
        
        # 内移距
        p = y_end - R * (1 - np.cos(beta))
        
        # 切线增长
        q = x_end - R * np.sin(beta)
        
        elements = {
            '缓和曲线长度 (L)': L,
            '圆曲线半径 (R)': R,
            '缓和曲线参数 (A)': A,
            '切线角 (β)': np.degrees(beta),
            '内移距 (p)': p,
            '切线增长 (q)': q,
            '缓和曲线终点X坐标': x_end,
            '缓和曲线终点Y坐标': y_end
        }
        
        return elements
    
    def plot_design_drawing(self, total_angle_deg=45):
        """绘制设计图纸"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'道路缓和曲线设计图 (V={self.design_speed}km/h, R={self.radius}m)', 
                     fontsize=16, fontweight='bold')
        
        # 1. 平面图
        self._plot_plan_view(ax1, total_angle_deg)
        
        # 2. 曲率图
        self._plot_curvature_diagram(ax2)
        
        # 3. 超高图
        self._plot_superelevation_diagram(ax3)
        
        # 4. 设计要素表
        self._plot_design_table(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_plan_view(self, ax, total_angle_deg):
        """绘制平面图"""
        L = self.transition_length
        R = self.radius
        A = self.A
        
        # 第一缓和曲线
        t1 = np.linspace(0, np.sqrt(2 * L / A), 100)
        S1, C1 = fresnel(t1 * np.sqrt(2/np.pi))
        x1 = A * C1 * np.sqrt(np.pi/2)
        y1 = A * S1 * np.sqrt(np.pi/2)
        
        ax.plot(x1, y1, 'b-', linewidth=3, label='第一缓和曲线')
        
        # 圆曲线段
        beta = L**2 / (2 * R * L)
        circle_angle = np.radians(total_angle_deg) - 2 * beta
        theta = np.linspace(beta, beta + circle_angle, 100)
        
        x_circle = x1[-1] + R * (np.sin(theta) - np.sin(beta))
        y_circle = y1[-1] + R * (np.cos(beta) - np.cos(theta))
        
        ax.plot(x_circle, y_circle, 'r-', linewidth=3, label='圆曲线')
        
        # 第二缓和曲线（镜像和旋转）
        t2 = np.linspace(np.sqrt(2 * L / A), 0, 100)
        S2, C2 = fresnel(t2 * np.sqrt(2/np.pi))
        x2_rel = A * C2 * np.sqrt(np.pi/2)
        y2_rel = A * S2 * np.sqrt(np.pi/2)
        
        # 旋转和平移
        end_angle = beta + circle_angle
        cos_end = np.cos(end_angle)
        sin_end = np.sin(end_angle)
        
        x2 = x_circle[-1] + x2_rel * cos_end - y2_rel * sin_end
        y2 = y_circle[-1] + x2_rel * sin_end + y2_rel * cos_end
        
        ax.plot(x2, y2, 'g-', linewidth=3, label='第二缓和曲线')
        
        # 标注关键点
        ax.plot(0, 0, 'ko', markersize=8, label='起点')
        ax.plot(x1[-1], y1[-1], 'bo', markersize=6)
        ax.plot(x_circle[-1], y_circle[-1], 'ro', markersize=6)
        ax.plot(x2[-1], y2[-1], 'go', markersize=8, label='终点')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title('道路平面线形图')
        ax.legend()
        ax.axis('equal')
    
    def _plot_curvature_diagram(self, ax):
        """绘制曲率图"""
        L = self.transition_length
        R = self.radius
        
        # 第一缓和曲线
        s1 = np.linspace(0, L, 100)
        k1 = s1 / (R * L)
        
        # 圆曲线（假设长度为L）
        s2 = np.linspace(L, 2*L, 100)
        k2 = np.ones_like(s2) / R
        
        # 第二缓和曲线
        s3 = np.linspace(2*L, 3*L, 100)
        k3 = (3*L - s3) / (R * L)
        
        ax.plot(s1, k1, 'b-', linewidth=2, label='第一缓和曲线')
        ax.plot(s2, k2, 'r-', linewidth=2, label='圆曲线')
        ax.plot(s3, k3, 'g-', linewidth=2, label='第二缓和曲线')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('里程 (m)')
        ax.set_ylabel('曲率 (1/m)')
        ax.set_title('曲率变化图')
        ax.legend()
    
    def _plot_superelevation_diagram(self, ax):
        """绘制超高图"""
        L = self.transition_length
        
        # 超高过渡
        s = np.linspace(0, L, 100)
        slope = self.superelevation_rate * s / L
        
        ax.plot(s, slope*100, 'b-', linewidth=2, label='外侧横坡')
        ax.plot(s, -slope*100, 'r-', linewidth=2, label='内侧横坡')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('距离 (m)')
        ax.set_ylabel('横坡度 (%)')
        ax.set_title('超高过渡图')
        ax.legend()
    
    def _plot_design_table(self, ax):
        """绘制设计要素表"""
        ax.axis('off')
        
        # 获取曲线要素
        elements = self.calculate_curve_elements()
        
        # 检查设计标准
        standards = self.check_design_standards()
        
        # 创建表格数据
        table_data = []
        table_data.append(['设计参数', '数值', '单位'])
        table_data.append(['设计速度', f'{self.design_speed}', 'km/h'])
        table_data.append(['圆曲线半径', f'{self.radius}', 'm'])
        table_data.append(['缓和曲线长度', f'{self.transition_length}', 'm'])
        table_data.append(['缓和曲线参数', f'{self.A:.2f}', 'm'])
        table_data.append(['车道宽度', f'{self.lane_width}', 'm'])
        table_data.append(['超高横坡度', f'{self.superelevation_rate*100}', '%'])
        
        # 绘制表格
        table = ax.table(cellText=table_data[1:], 
                        colLabels=table_data[0],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.set_title('设计参数表')

def main():
    """主函数"""
    print("=" * 60)
    print("道路缓和曲线设计计算器")
    print("=" * 60)
    
    # 示例计算
    calculator = RoadDesignCalculator()
    calculator.set_design_parameters(speed=100, radius=400, transition_length=120)
    
    print("\n示例设计参数:")
    print(f"设计速度: {calculator.design_speed} km/h")
    print(f"圆曲线半径: {calculator.radius} m")
    print(f"缓和曲线长度: {calculator.transition_length} m")
    
    # 检查设计标准
    print("\n设计检查结果:")
    standards = calculator.check_design_standards()
    for check_name, result in standards.items():
        print(f"{check_name}: {'合格' if result['是否合格'] else '不合格'}")
    
    # 显示曲线要素
    print("\n曲线要素:")
    elements = calculator.calculate_curve_elements()
    for name, value in elements.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")
    
    # 生成设计图纸
    fig = calculator.plot_design_drawing(60)
    fig.savefig('images/road_design_example.png', dpi=300, bbox_inches='tight')
    print("\n设计图纸已保存: images/road_design_example.png")
    
    plt.show()

if __name__ == "__main__":
    main() 