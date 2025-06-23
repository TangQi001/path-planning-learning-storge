"""
Voronoi图路径规划 - 交互式演示
==============================

本模块提供交互式的Voronoi图路径规划演示，支持：
1. 鼠标交互编辑障碍物
2. 拖拽起点和终点
3. 实时参数调节
4. 多场景切换

使用方法：
python interactive_demo.py

操作说明：
- 鼠标左键：添加障碍物
- 鼠标右键：删除障碍物  
- 拖拽绿色圆圈：移动起点
- 拖拽红色方块：移动终点
- 滚轮：调整新障碍物大小

作者：AI教程生成器
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches
from matplotlib.patches import Circle
import sys
import os

# 添加核心模块路径，以便导入核心模块
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(os.path.dirname(current_dir), '02_代码实现')
sys.path.insert(0, core_path)
from core_voronoi import VoronoiPathPlanner, Point, Obstacle

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveVoronoiDemo:
    """交互式Voronoi图路径规划演示"""
    
    def __init__(self, bounds=(0, 0, 20, 15)):
        self.bounds = bounds
        self.planner = None
        self.current_path = None
        self.current_distance = 0
        
        # 交互状态
        self.start_point = Point(2, 2)
        self.goal_point = Point(18, 13)
        self.new_obstacle_radius = 1.0
        self.safety_margin = 0.3
        self.dragging_start = False
        self.dragging_goal = False
        
        # 预设场景
        self.scenarios = {
            '简单场景': [
                (5, 5, 1.5),
                (12, 8, 2.0),
                (15, 3, 1.8)
            ],
            '复杂迷宫': [
                (4, 4, 1.0), (4, 8, 1.0), (4, 12, 1.0),
                (8, 2, 1.0), (8, 6, 1.0), (8, 10, 1.0),
                (12, 4, 1.0), (12, 8, 1.0), (12, 12, 1.0),
                (16, 2, 1.0), (16, 6, 1.0), (16, 10, 1.0)
            ],
            '稀疏障碍': [
                (6, 7, 2.0),
                (14, 9, 1.5)
            ],
            '密集障碍': [
                (3, 3, 0.8), (6, 4, 0.9), (9, 3, 0.7),
                (12, 4, 1.0), (15, 3, 0.8), (17, 6, 0.9),
                (16, 9, 0.8), (13, 10, 0.7), (10, 11, 0.9),
                (7, 10, 0.8), (4, 9, 0.7), (2, 6, 0.9)
            ]
        }
        
        self.current_scenario = '简单场景'
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主图形
        self.fig = plt.figure(figsize=(16, 10))
        
        # 主绘图区域
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        
        # 控制面板区域
        self.ax_controls = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis('off')
        
        # 状态显示区域
        self.ax_status = plt.subplot2grid((4, 4), (3, 0), colspan=3)
        self.ax_status.axis('off')
        
        # 初始化控件
        self.setup_controls()
        
        # 设置事件处理
        self.setup_events()
        
        # 初始化场景
        self.load_scenario(self.current_scenario)
        
    def setup_controls(self):
        """设置控制面板"""
        # 场景选择
        scenarios_ax = plt.axes([0.77, 0.8, 0.2, 0.15])
        self.radio_scenarios = RadioButtons(scenarios_ax, list(self.scenarios.keys()))
        self.radio_scenarios.on_clicked(self.load_scenario)
        
        # 安全边距滑块
        margin_ax = plt.axes([0.77, 0.65, 0.2, 0.03])
        self.slider_margin = Slider(margin_ax, '安全边距', 0.1, 1.0, 
                                   valinit=self.safety_margin, valfmt='%.2f')
        self.slider_margin.on_changed(self.update_safety_margin)
        
        # 障碍物大小滑块
        radius_ax = plt.axes([0.77, 0.6, 0.2, 0.03])
        self.slider_radius = Slider(radius_ax, '障碍物大小', 0.5, 3.0,
                                   valinit=self.new_obstacle_radius, valfmt='%.1f')
        self.slider_radius.on_changed(self.update_obstacle_radius)
        
        # 控制按钮
        clear_ax = plt.axes([0.77, 0.5, 0.08, 0.04])
        self.btn_clear = Button(clear_ax, '清空')
        self.btn_clear.on_clicked(self.clear_obstacles)
        
        replan_ax = plt.axes([0.87, 0.5, 0.08, 0.04])
        self.btn_replan = Button(replan_ax, '重新规划')
        self.btn_replan.on_clicked(self.replan_path)
        
        # 显示模式选择
        mode_ax = plt.axes([0.77, 0.35, 0.15, 0.1])
        self.radio_mode = RadioButtons(mode_ax, ['基础模式', '详细模式'])
        self.radio_mode.on_clicked(self.change_display_mode)
        
    def setup_events(self):
        """设置事件处理"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def load_scenario(self, scenario_name):
        """加载预设场景"""
        self.current_scenario = scenario_name
        obstacles = self.scenarios[scenario_name]
        
        # 重新创建规划器
        self.planner = VoronoiPathPlanner(self.bounds, self.safety_margin)
        self.planner.add_obstacles(obstacles)
        
        # 重新规划路径
        self.update_visualization()
        
    def update_safety_margin(self, val):
        """更新安全边距"""
        self.safety_margin = val
        if self.planner:
            # 保存当前障碍物
            current_obstacles = [(obs.center.x, obs.center.y, 
                                obs.radius - self.planner.safety_margin) 
                               for obs in self.planner.obstacles]
            
            # 重新创建规划器
            self.planner = VoronoiPathPlanner(self.bounds, self.safety_margin)
            self.planner.add_obstacles(current_obstacles)
            
            self.update_visualization()
    
    def update_obstacle_radius(self, val):
        """更新新障碍物大小"""
        self.new_obstacle_radius = val
        
    def clear_obstacles(self, event):
        """清空所有障碍物"""
        self.planner = VoronoiPathPlanner(self.bounds, self.safety_margin)
        self.update_visualization()
        
    def replan_path(self, event):
        """重新规划路径"""
        self.update_visualization()
        
    def change_display_mode(self, mode):
        """切换显示模式"""
        self.display_mode = mode
        self.update_visualization()
        
    def on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes != self.ax_main:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # 检查是否点击了起点或终点
        start_dist = np.sqrt((x - self.start_point.x)**2 + (y - self.start_point.y)**2)
        goal_dist = np.sqrt((x - self.goal_point.x)**2 + (y - self.goal_point.y)**2)
        
        if start_dist < 0.5:  # 点击起点
            self.dragging_start = True
        elif goal_dist < 0.5:  # 点击终点
            self.dragging_goal = True
        elif event.button == 1:  # 左键添加障碍物
            self.add_obstacle(x, y)
        elif event.button == 3:  # 右键删除障碍物
            self.remove_obstacle(x, y)
            
    def on_release(self, event):
        """鼠标释放事件"""
        if self.dragging_start or self.dragging_goal:
            self.dragging_start = False
            self.dragging_goal = False
            self.update_visualization()
            
    def on_motion(self, event):
        """鼠标移动事件"""
        if event.inaxes != self.ax_main:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        if self.dragging_start:
            self.start_point = Point(x, y)
            self.update_visualization(update_path=False)
        elif self.dragging_goal:
            self.goal_point = Point(x, y)
            self.update_visualization(update_path=False)
            
    def on_scroll(self, event):
        """鼠标滚轮事件"""
        if event.inaxes != self.ax_main:
            return
            
        # 调整新障碍物大小
        if event.button == 'up':
            self.new_obstacle_radius = min(3.0, self.new_obstacle_radius + 0.1)
        else:
            self.new_obstacle_radius = max(0.3, self.new_obstacle_radius - 0.1)
            
        self.slider_radius.set_val(self.new_obstacle_radius)
        
    def add_obstacle(self, x, y):
        """添加障碍物"""
        if not self.planner:
            self.planner = VoronoiPathPlanner(self.bounds, self.safety_margin)
            
        self.planner.add_obstacle(Point(x, y), self.new_obstacle_radius)
        self.update_visualization()
        
    def remove_obstacle(self, x, y):
        """删除障碍物"""
        if not self.planner or not self.planner.obstacles:
            return
            
        # 找到最近的障碍物
        min_dist = float('inf')
        closest_idx = -1
        
        for i, obstacle in enumerate(self.planner.obstacles):
            dist = np.sqrt((x - obstacle.center.x)**2 + (y - obstacle.center.y)**2)
            if dist < obstacle.radius and dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        if closest_idx >= 0:
            self.planner.obstacles.pop(closest_idx)
            self.update_visualization()
            
    def update_visualization(self, update_path=True):
        """更新可视化"""
        self.ax_main.clear()
        
        # 设置坐标轴
        x_min, y_min, x_max, y_max = self.bounds
        self.ax_main.set_xlim(x_min - 1, x_max + 1)
        self.ax_main.set_ylim(y_min - 1, y_max + 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        
        # 绘制边界
        boundary = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, edgecolor='black', linewidth=2)
        self.ax_main.add_patch(boundary)
        
        # 绘制障碍物
        if self.planner and self.planner.obstacles:
            for obstacle in self.planner.obstacles:
                circle = Circle((obstacle.center.x, obstacle.center.y),
                              obstacle.radius, color='red', alpha=0.7)
                self.ax_main.add_patch(circle)
                
            # 构造Voronoi图和规划路径
            try:
                if update_path:
                    self.planner.construct_voronoi()
                    self.current_path, self.current_distance = self.planner.plan_path(
                        self.start_point, self.goal_point)
                
                # 绘制Voronoi图
                if hasattr(self, 'display_mode') and self.display_mode == '详细模式':
                    self.draw_detailed_voronoi()
                else:
                    self.draw_basic_voronoi()
                    
            except Exception as e:
                print(f"Voronoi构造或路径规划出错: {e}")
                self.current_path = None
                self.current_distance = float('inf')
        
        # 绘制起点和终点
        self.ax_main.scatter(self.start_point.x, self.start_point.y, 
                           c='green', s=150, marker='o', edgecolors='black', 
                           linewidth=2, label='起点', zorder=5)
        self.ax_main.scatter(self.goal_point.x, self.goal_point.y,
                           c='red', s=150, marker='s', edgecolors='black',
                           linewidth=2, label='终点', zorder=5)
        
        # 绘制路径
        if self.current_path and len(self.current_path) > 1:
            path_x = [p.x for p in self.current_path]
            path_y = [p.y for p in self.current_path]
            self.ax_main.plot(path_x, path_y, 'g-', linewidth=4, 
                            alpha=0.8, label=f'路径 (长度: {self.current_distance:.2f})')
            
        self.ax_main.legend(loc='upper left')
        self.ax_main.set_title('交互式Voronoi图路径规划\n'
                              '左键：添加障碍物，右键：删除障碍物，拖拽：移动起终点')
        
        # 更新状态信息
        self.update_status()
        
        self.fig.canvas.draw()
        
    def draw_basic_voronoi(self):
        """绘制基础Voronoi图"""
        if self.planner and self.planner.valid_edges:
            for edge in self.planner.valid_edges:
                p1, p2 = edge
                self.ax_main.plot([p1.x, p2.x], [p1.y, p2.y],
                                'b-', alpha=0.4, linewidth=1)
                                
    def draw_detailed_voronoi(self):
        """绘制详细Voronoi图"""
        if not self.planner or not self.planner.voronoi:
            return
            
        # 绘制Voronoi顶点
        vertices = self.planner.voronoi.vertices
        valid_vertices = []
        for vertex in vertices:
            if self.planner._point_in_bounds(vertex):
                valid_vertices.append(vertex)
                
        if valid_vertices:
            valid_vertices = np.array(valid_vertices)
            self.ax_main.scatter(valid_vertices[:, 0], valid_vertices[:, 1],
                               c='blue', s=30, alpha=0.7, label='Voronoi顶点')
            
        # 绘制有效边
        for edge in self.planner.valid_edges:
            p1, p2 = edge
            self.ax_main.plot([p1.x, p2.x], [p1.y, p2.y],
                            'b-', alpha=0.6, linewidth=1.5, label='Voronoi边')
            
    def update_status(self):
        """更新状态信息"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        status_text = []
        status_text.append(f"场景: {self.current_scenario}")
        status_text.append(f"障碍物数量: {len(self.planner.obstacles) if self.planner else 0}")
        status_text.append(f"安全边距: {self.safety_margin:.2f}")
        status_text.append(f"新障碍物大小: {self.new_obstacle_radius:.1f}")
        
        if self.current_path:
            status_text.append(f"路径长度: {self.current_distance:.2f}")
            status_text.append(f"路径点数: {len(self.current_path)}")
            
            # 计算直线距离
            direct_dist = np.sqrt((self.goal_point.x - self.start_point.x)**2 + 
                                (self.goal_point.y - self.start_point.y)**2)
            overhead = (self.current_distance / direct_dist - 1) * 100
            status_text.append(f"路径开销: {overhead:.1f}%")
        else:
            status_text.append("状态: 无可行路径")
            
        # 显示操作说明
        status_text.append("")
        status_text.append("操作说明:")
        status_text.append("• 左键点击：添加障碍物")
        status_text.append("• 右键点击：删除障碍物")
        status_text.append("• 拖拽绿圆：移动起点")
        status_text.append("• 拖拽红方：移动终点")
        status_text.append("• 滚轮：调整障碍物大小")
        
        for i, text in enumerate(status_text):
            self.ax_status.text(0.05, 0.95 - i * 0.08, text, 
                              transform=self.ax_status.transAxes,
                              fontsize=9, verticalalignment='top')
                              
    def run(self):
        """运行交互式演示"""
        self.display_mode = '基础模式'
        self.update_visualization()
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("启动Voronoi图路径规划交互式演示...")
    print("请在图形界面中进行交互操作")
    
    demo = InteractiveVoronoiDemo()
    demo.run()

if __name__ == "__main__":
    main() 