
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
"""
Bezier曲线路径规划可视化模块

Author: AI Assistant
Date: 2024
Description: 提供Bezier路径规划的各种可视化功能，包括静态图形、动态动画和交互式演示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Callable
import time
import sys
import os

# 配置matplotlib支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 添加上级目录到路径，以便导入core_algorithm
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02_代码实现'))
from core_algorithm import Point, BezierCurve, BezierPath, PathPlanner, Constraints

class PathVisualizer:
    """Bezier路径可视化器"""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8), dpi: int = 100):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
            dpi: 图形分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        self._color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self._current_color_idx = 0
    
    def _setup_figure(self, title: str = "Bezier路径规划可视化"):
        """设置图形环境"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        self.ax.clear()
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X坐标 (m)', fontsize=12)
        self.ax.set_ylabel('Y坐标 (m)', fontsize=12)
    
    def _get_next_color(self) -> str:
        """获取下一个颜色"""
        color = self._color_cycle[self._current_color_idx]
        self._current_color_idx = (self._current_color_idx + 1) % len(self._color_cycle)
        return color
    
    def plot_waypoints(self, waypoints: List[Point], style: str = 'ro-', 
                      markersize: int = 8, linewidth: float = 2, 
                      label: str = '航点', alpha: float = 0.8):
        """
        绘制航点
        
        Args:
            waypoints: 航点列表
            style: 绘制样式
            markersize: 标记大小
            linewidth: 线宽
            label: 图例标签
            alpha: 透明度
        """
        if not waypoints:
            return
        
        x_coords = [wp.x for wp in waypoints]
        y_coords = [wp.y for wp in waypoints]
        
        self.ax.plot(x_coords, y_coords, style, markersize=markersize, 
                    linewidth=linewidth, label=label, alpha=alpha)
        
        # 添加航点编号
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            self.ax.annotate(f'W{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    def plot_bezier_path(self, path: BezierPath, style: str = 'b-', 
                        num_points: int = 100, linewidth: float = 2, 
                        label: str = 'Bezier路径', alpha: float = 1.0):
        """
        绘制Bezier路径
        
        Args:
            path: Bezier路径对象
            style: 绘制样式
            num_points: 每段采样点数
            linewidth: 线宽
            label: 图例标签
            alpha: 透明度
        """
        if not path.segments:
            return
        
        all_x, all_y = [], []
        
        for segment in path.segments:
            points = segment.sample_points(num_points)
            x_coords = [p.x for p in points]
            y_coords = [p.y for p in points]
            all_x.extend(x_coords)
            all_y.extend(y_coords)
        
        self.ax.plot(all_x, all_y, style, linewidth=linewidth, label=label, alpha=alpha)
    
    def plot_control_points(self, segments: List[BezierCurve], 
                           show_polygon: bool = True, show_points: bool = True,
                           point_style: str = 'o', polygon_style: str = '--',
                           point_color: str = 'orange', polygon_color: str = 'gray'):
        """
        绘制控制点和控制多边形
        
        Args:
            segments: Bezier曲线段列表
            show_polygon: 是否显示控制多边形
            show_points: 是否显示控制点
            point_style: 控制点样式
            polygon_style: 控制多边形样式
            point_color: 控制点颜色
            polygon_color: 控制多边形颜色
        """
        for i, segment in enumerate(segments):
            cp = segment.control_points
            x_coords = [p.x for p in cp]
            y_coords = [p.y for p in cp]
            
            if show_polygon:
                self.ax.plot(x_coords, y_coords, polygon_style, 
                           color=polygon_color, alpha=0.5, linewidth=1)
            
            if show_points:
                self.ax.plot(x_coords, y_coords, point_style, 
                           color=point_color, markersize=6, alpha=0.8)
                
                # 标注控制点
                for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                    self.ax.annotate(f'P{i}.{j}', (x, y), xytext=(-5, -15), 
                                   textcoords='offset points', fontsize=8, 
                                   alpha=0.7)
    
    def plot_curvature_distribution(self, path: BezierPath, num_samples: int = 200,
                                  max_curvature: Optional[float] = None):
        """
        绘制曲率分布图
        
        Args:
            path: Bezier路径
            num_samples: 采样点数量
            max_curvature: 最大曲率限制线
        """
        # 计算曲率数据
        curvatures = []
        distances = []
        current_distance = 0
        
        for segment in path.segments:
            segment_length = segment.arc_length()
            t_values = np.linspace(0, 1, num_samples // len(path.segments))
            
            for t in t_values:
                curvature = segment.curvature(t)
                curvatures.append(curvature)
                distances.append(current_distance + t * segment_length)
            
            current_distance += segment_length
        
        # 创建子图
        if hasattr(self, '_curvature_fig'):
            plt.figure(self._curvature_fig.number)
        else:
            self._curvature_fig, self._curvature_ax = plt.subplots(figsize=(12, 4))
        
        self._curvature_ax.clear()
        self._curvature_ax.plot(distances, curvatures, 'b-', linewidth=2, label='路径曲率')
        
        if max_curvature is not None:
            self._curvature_ax.axhline(y=max_curvature, color='r', linestyle='--', 
                                     linewidth=2, label=f'最大曲率限制 ({max_curvature:.3f})')
        
        self._curvature_ax.set_xlabel('沿路径距离 (m)')
        self._curvature_ax.set_ylabel('曲率 (1/m)')
        self._curvature_ax.set_title('路径曲率分布')
        self._curvature_ax.grid(True, alpha=0.3)
        self._curvature_ax.legend()
        
        plt.tight_layout()
    
    def plot_comparison(self, waypoints: List[Point], 
                       paths: List[Tuple[BezierPath, str, str]] = None,
                       show_control_points: bool = True):
        """
        绘制多种方法的对比
        
        Args:
            waypoints: 航点列表
            paths: (路径, 标签, 颜色) 的列表
            show_control_points: 是否显示控制点
        """
        self._setup_figure("Bezier路径规划方法对比")
        
        # 绘制航点
        self.plot_waypoints(waypoints)
        
        # 如果没有提供路径，生成默认的对比路径
        if paths is None:
            planner = PathPlanner()
            path_tangent = planner.generate_smooth_path(waypoints, method="tangent")
            path_optimized = planner.generate_smooth_path(waypoints, method="optimized")
            
            paths = [
                (path_tangent, "切线方法", "blue"),
                (path_optimized, "优化方法", "green")
            ]
        
        # 绘制路径
        for path, label, color in paths:
            self.plot_bezier_path(path, style=f'{color[0]}-', label=label)
            
            if show_control_points:
                self.plot_control_points(path.segments, 
                                       point_color=color, 
                                       polygon_color=color)
        
        self.ax.legend()
        plt.tight_layout()
    
    def animate_path_generation(self, waypoints: List[Point], 
                              method: str = "tangent", 
                              save_gif: bool = False, 
                              filename: str = "bezier_animation.gif"):
        """
        动画展示路径生成过程
        
        Args:
            waypoints: 航点列表
            method: 生成方法
            save_gif: 是否保存为GIF
            filename: 保存文件名
        """
        self._setup_figure("Bezier路径生成动画")
        
        # 生成路径
        planner = PathPlanner()
        path = planner.generate_smooth_path(waypoints, method=method)
        
        # 准备动画数据
        all_points = []
        for segment in path.segments:
            points = segment.sample_points(50)
            all_points.extend(points)
        
        # 动画函数
        def animate(frame):
            self.ax.clear()
            self._setup_figure("Bezier路径生成动画")
            
            # 绘制航点
            self.plot_waypoints(waypoints)
            
            # 绘制控制点（逐渐显示）
            if frame > 20:
                segment_idx = min((frame - 20) // 10, len(path.segments) - 1)
                segments_to_show = path.segments[:segment_idx + 1]
                self.plot_control_points(segments_to_show)
            
            # 绘制路径（逐步生成）
            if frame > 40:
                points_to_show = min(frame - 40, len(all_points))
                if points_to_show > 0:
                    x_coords = [p.x for p in all_points[:points_to_show]]
                    y_coords = [p.y for p in all_points[:points_to_show]]
                    self.ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.8)
        
        # 创建动画
        frames = 40 + len(all_points) + 20
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=50, repeat=True)
        
        if save_gif:
            print(f"正在保存动画到 {filename}...")
            anim.save(filename, writer='pillow', fps=20)
            print("动画保存完成！")
        
        return anim
    
    def animate_de_casteljau(self, control_points: List[Point], 
                           save_gif: bool = False,
                           filename: str = "de_casteljau_animation.gif"):
        """
        动画展示De Casteljau算法过程
        
        Args:
            control_points: 控制点列表
            save_gif: 是否保存为GIF
            filename: 保存文件名
        """
        self._setup_figure("De Casteljau算法可视化")
        
        curve = BezierCurve(control_points)
        
        def animate(frame):
            self.ax.clear()
            self._setup_figure("De Casteljau算法可视化")
            
            t = frame / 100.0  # t从0到1
            
            # 绘制控制多边形
            x_coords = [p.x for p in control_points]
            y_coords = [p.y for p in control_points]
            self.ax.plot(x_coords, y_coords, 'ko--', alpha=0.5, 
                        markersize=8, label='控制点')
            
            # De Casteljau递归过程
            current_points = control_points[:]
            colors = ['red', 'orange', 'yellow', 'green']
            
            for level in range(len(control_points) - 1):
                new_points = []
                for i in range(len(current_points) - 1):
                    # 线性插值
                    p1, p2 = current_points[i], current_points[i + 1]
                    new_point = Point(
                        p1.x * (1 - t) + p2.x * t,
                        p1.y * (1 - t) + p2.y * t
                    )
                    new_points.append(new_point)
                
                # 绘制这一层的点和连线
                if new_points:
                    color = colors[level % len(colors)]
                    nx = [p.x for p in new_points]
                    ny = [p.y for p in new_points]
                    
                    if len(new_points) > 1:
                        self.ax.plot(nx, ny, 'o-', color=color, alpha=0.7,
                                   markersize=6, label=f'第{level+1}层')
                    else:
                        self.ax.plot(nx, ny, 'o', color=color, markersize=10,
                                   label='最终点')
                
                current_points = new_points
            
            # 绘制完整曲线（淡显示）
            t_values = np.linspace(0, 1, 100)
            curve_points = [curve.evaluate(t_val) for t_val in t_values]
            cx = [p.x for p in curve_points]
            cy = [p.y for p in curve_points]
            self.ax.plot(cx, cy, 'b-', alpha=0.3, linewidth=2, label='Bezier曲线')
            
            # 显示当前t值
            self.ax.text(0.02, 0.98, f't = {t:.2f}', 
                        transform=self.ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.ax.legend()
        
        # 创建动画
        anim = animation.FuncAnimation(self.fig, animate, frames=101, 
                                     interval=100, repeat=True)
        
        if save_gif:
            print(f"正在保存De Casteljau动画到 {filename}...")
            anim.save(filename, writer='pillow', fps=10)
            print("动画保存完成！")
        
        return anim
    
    def parameter_sensitivity_analysis(self, waypoints: List[Point], 
                                     param_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        参数敏感性分析
        
        Args:
            waypoints: 航点列表
            param_range: 参数α的取值范围
        """
        self._setup_figure("参数敏感性分析")
        
        # 绘制航点
        self.plot_waypoints(waypoints)
        
        planner = PathPlanner()
        
        for i, alpha in enumerate(param_range):
            # 手动生成控制点
            segments_cp = planner.generate_control_points_tangent(waypoints, alpha=alpha)
            segments = [BezierCurve(cp) for cp in segments_cp]
            path = BezierPath(segments)
            
            # 使用不同颜色绘制
            color = self._get_next_color()
            self.plot_bezier_path(path, style=f'-', linewidth=2, 
                                label=f'α = {alpha}')
        
        self.ax.legend()
        plt.tight_layout()
    
    def show(self):
        """显示图形"""
        plt.show()
    
    def save(self, filename: str, dpi: int = 300):
        """
        保存图形
        
        Args:
            filename: 文件名
            dpi: 分辨率
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图形已保存到: {filename}")

class InteractiveDemo:
    """交互式演示"""
    
    def __init__(self):
        """初始化交互式演示"""
        self.waypoints = []
        self.planner = PathPlanner()
        self.visualizer = PathVisualizer(figsize=(14, 10))
        self.current_path = None
        
        # 参数
        self.alpha = 0.3
        self.method = "tangent"
        
        # UI组件
        self.sliders = {}
        self.buttons = {}
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主图和控制面板
        self.fig = plt.figure(figsize=(16, 10))
        
        # 主绘图区域
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        
        # 曲率图区域
        self.ax_curvature = plt.subplot2grid((4, 4), (3, 0), colspan=3)
        
        # 控制面板
        self.ax_controls = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        self.ax_controls.axis('off')
        
        # α参数滑块
        ax_alpha = plt.axes([0.8, 0.7, 0.15, 0.03])
        self.sliders['alpha'] = Slider(ax_alpha, 'α系数', 0.1, 1.0, valinit=self.alpha)
        self.sliders['alpha'].on_changed(self.update_path)
        
        # 方法切换按钮
        ax_method = plt.axes([0.8, 0.6, 0.1, 0.04])
        self.buttons['method'] = Button(ax_method, '切换方法')
        self.buttons['method'].on_clicked(self.toggle_method)
        
        # 清除按钮
        ax_clear = plt.axes([0.8, 0.5, 0.1, 0.04])
        self.buttons['clear'] = Button(ax_clear, '清除')
        self.buttons['clear'].on_clicked(self.clear_waypoints)
        
        # 保存按钮
        ax_save = plt.axes([0.8, 0.4, 0.1, 0.04])
        self.buttons['save'] = Button(ax_save, '保存图像')
        self.buttons['save'].on_clicked(self.save_image)
        
        # 鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 添加说明文字
        self.ax_controls.text(0.1, 0.9, '操作说明:', fontsize=12, fontweight='bold')
        self.ax_controls.text(0.1, 0.85, '• 左键点击添加航点', fontsize=10)
        self.ax_controls.text(0.1, 0.8, '• 右键点击删除最近航点', fontsize=10)
        self.ax_controls.text(0.1, 0.75, '• 调节α系数改变形状', fontsize=10)
        self.ax_controls.text(0.1, 0.7, '• 切换方法比较效果', fontsize=10)
        
        # 当前状态显示
        self.status_text = self.ax_controls.text(0.1, 0.3, '', fontsize=10)
    
    def on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax_main:
            return
        
        if event.button == 1:  # 左键添加航点
            new_point = Point(event.xdata, event.ydata)
            self.waypoints.append(new_point)
            self.update_display()
        
        elif event.button == 3:  # 右键删除最近航点
            if self.waypoints:
                click_point = Point(event.xdata, event.ydata)
                min_dist = float('inf')
                closest_idx = -1
                
                for i, wp in enumerate(self.waypoints):
                    dist = wp.distance_to(click_point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx >= 0:
                    self.waypoints.pop(closest_idx)
                    self.update_display()
    
    def update_path(self, val=None):
        """更新路径"""
        if val is not None:
            self.alpha = self.sliders['alpha'].val
        
        if len(self.waypoints) >= 2:
            if self.method == "tangent":
                segments_cp = self.planner.generate_control_points_tangent(
                    self.waypoints, alpha=self.alpha)
                segments = [BezierCurve(cp) for cp in segments_cp]
                self.current_path = BezierPath(segments)
            else:
                self.current_path = self.planner.generate_smooth_path(
                    self.waypoints, method="optimized")
        else:
            self.current_path = None
        
        self.update_display()
    
    def toggle_method(self, event):
        """切换生成方法"""
        self.method = "optimized" if self.method == "tangent" else "tangent"
        self.update_path()
    
    def clear_waypoints(self, event):
        """清除所有航点"""
        self.waypoints.clear()
        self.current_path = None
        self.update_display()
    
    def save_image(self, event):
        """保存当前图像"""
        filename = f"bezier_interactive_{int(time.time())}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {filename}")
    
    def update_display(self):
        """更新显示"""
        # 清空主图
        self.ax_main.clear()
        self.ax_main.set_title('交互式Bezier路径规划', fontsize=14)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # 绘制航点
        if self.waypoints:
            x_coords = [wp.x for wp in self.waypoints]
            y_coords = [wp.y for wp in self.waypoints]
            self.ax_main.plot(x_coords, y_coords, 'ro-', markersize=8, 
                            linewidth=2, label='航点', alpha=0.8)
        
        # 绘制路径
        if self.current_path:
            all_x, all_y = [], []
            for segment in self.current_path.segments:
                points = segment.sample_points(50)
                x_coords = [p.x for p in points]
                y_coords = [p.y for p in points]
                all_x.extend(x_coords)
                all_y.extend(y_coords)
            
            color = 'blue' if self.method == "tangent" else 'green'
            self.ax_main.plot(all_x, all_y, color=color, linewidth=2, 
                            label=f'{self.method}方法')
            
            # 绘制控制点
            for segment in self.current_path.segments:
                cp = segment.control_points
                x_coords = [p.x for p in cp]
                y_coords = [p.y for p in cp]
                self.ax_main.plot(x_coords, y_coords, 'o--', 
                                color='orange', alpha=0.5, markersize=4)
        
        self.ax_main.legend()
        
        # 更新曲率图
        self.ax_curvature.clear()
        if self.current_path:
            # 计算曲率
            curvatures = []
            distances = []
            current_distance = 0
            
            for segment in self.current_path.segments:
                segment_length = segment.arc_length()
                t_values = np.linspace(0, 1, 50)
                
                for t in t_values:
                    curvature = segment.curvature(t)
                    curvatures.append(curvature)
                    distances.append(current_distance + t * segment_length)
                
                current_distance += segment_length
            
            self.ax_curvature.plot(distances, curvatures, 'b-', linewidth=2)
            self.ax_curvature.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            self.ax_curvature.set_xlabel('沿路径距离')
            self.ax_curvature.set_ylabel('曲率')
            self.ax_curvature.set_title('曲率分布')
            self.ax_curvature.grid(True, alpha=0.3)
        
        # 更新状态信息
        status = f"航点数: {len(self.waypoints)}\n"
        status += f"方法: {self.method}\n"
        status += f"α: {self.alpha:.2f}\n"
        if self.current_path:
            status += f"路径长度: {self.current_path.total_length():.2f}m"
        
        self.status_text.set_text(status)
        
        plt.draw()
    
    def run(self):
        """运行交互式演示"""
        self.setup_ui()
        self.update_display()
        print("交互式演示已启动！")
        print("• 左键点击添加航点")
        print("• 右键点击删除最近航点")
        print("• 调节参数观察效果变化")
        plt.show()

def demo_visualizations():
    """演示各种可视化功能"""
    print("=== Bezier路径规划可视化演示 ===")
    
    # 创建测试数据
    waypoints = [
        Point(0, 0),
        Point(3, 2),
        Point(7, 1),
        Point(10, 4),
        Point(15, 3)
    ]
    
    # 创建可视化器
    viz = PathVisualizer()
    
    # 演示1：基础路径对比
    print("1. 基础路径对比...")
    viz.plot_comparison(waypoints)
    viz.show()
    
    # 演示2：参数敏感性分析
    print("2. 参数敏感性分析...")
    viz.parameter_sensitivity_analysis(waypoints)
    viz.show()
    
    # 演示3：De Casteljau算法可视化
    print("3. De Casteljau算法可视化...")
    control_points = [Point(0, 0), Point(2, 4), Point(6, 4), Point(8, 0)]
    anim = viz.animate_de_casteljau(control_points)
    plt.show()
    
    # 演示4：曲率分布分析
    print("4. 曲率分布分析...")
    planner = PathPlanner()
    path = planner.generate_smooth_path(waypoints)
    viz.plot_curvature_distribution(path, max_curvature=0.5)
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bezier路径规划可视化")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    parser.add_argument("--interactive", action="store_true", help="启动交互式演示")
    
    args = parser.parse_args()
    
    if args.interactive:
        demo = InteractiveDemo()
        demo.run()
    elif args.demo:
        demo_visualizations()
    else:
        print("使用 --demo 运行演示，或 --interactive 启动交互式演示") 