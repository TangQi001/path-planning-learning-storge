
# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

"""
RRT算法动态可视化演示

作者: AICP-7协议实现
功能: 实时显示RRT树的增长过程
特点: 动画效果、步骤展示、交互控制
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import sys
import os
import time
import math

# 添加代码实现目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_代码实现'))
from rrt_basic import Node, RRTBasic
from rrt_star import RRTStar

class RRTAnimator:
    """RRT算法动态可视化器"""
    
    def __init__(self, rrt_planner, interval=50, save_frames=False):
        """
        初始化动画器
        
        Args:
            rrt_planner: RRT规划器实例
            interval: 动画帧间隔(ms)
            save_frames: 是否保存动画帧
        """
        self.rrt = rrt_planner
        self.interval = interval
        self.save_frames = save_frames
        
        # 动画状态
        self.current_iteration = 0
        self.animation_data = []
        self.found_path = None
        
        # 设置图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
        
    def setup_plot(self):
        """设置绘图环境"""
        self.ax.clear()
        
        # 绘制边界
        boundary = self.rrt.boundary
        boundary_rect = patches.Rectangle(
            (boundary[0], boundary[2]), 
            boundary[1] - boundary[0], 
            boundary[3] - boundary[2],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(boundary_rect)
        
        # 绘制障碍物
        for (ox, oy, radius) in self.rrt.obstacle_list:
            circle = patches.Circle((ox, oy), radius, 
                                  facecolor='red', alpha=0.6, edgecolor='darkred')
            self.ax.add_patch(circle)
        
        # 绘制起点和终点
        self.ax.scatter(self.rrt.start.x, self.rrt.start.y, c='blue', s=150, 
                       marker='o', label='起点', zorder=10)
        self.ax.scatter(self.rrt.goal.x, self.rrt.goal.y, c='red', s=150, 
                       marker='*', label='目标', zorder=10)
        
        self.ax.set_xlabel('X坐标')
        self.ax.set_ylabel('Y坐标')
        self.ax.set_title('RRT算法动态演示')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')
        
        # 设置坐标轴范围
        margin = 1.0
        self.ax.set_xlim(boundary[0] - margin, boundary[1] + margin)
        self.ax.set_ylim(boundary[2] - margin, boundary[3] + margin)
    
    def run_planning_with_recording(self):
        """运行规划并记录每个步骤"""
        print("📹 开始记录RRT规划过程...")
        
        # 重置规划器
        self.rrt.node_list = [self.rrt.start]
        self.animation_data = []
        
        for i in range(self.rrt.max_iter):
            # 记录当前状态
            current_nodes = [Node(node.x, node.y) for node in self.rrt.node_list]
            for j, node in enumerate(current_nodes):
                if j > 0:  # 复制父节点信息
                    original_node = self.rrt.node_list[j]
                    if original_node.parent:
                        parent_index = self.rrt.node_list.index(original_node.parent)
                        node.parent = current_nodes[parent_index]
            
            self.animation_data.append({
                'iteration': i,
                'nodes': current_nodes,
                'new_node': None,
                'sampled_point': None
            })
            
            # 执行一步RRT
            rand_node = self.rrt.sample()
            nearest_node = self.rrt.get_nearest_node(rand_node)
            new_node = self.rrt.steer(nearest_node, rand_node)
            
            # 记录采样点
            self.animation_data[-1]['sampled_point'] = (rand_node.x, rand_node.y)
            
            if not self.rrt.check_collision(nearest_node, new_node):
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.rrt.distance(nearest_node, new_node)
                self.rrt.node_list.append(new_node)
                
                # 记录新节点
                self.animation_data[-1]['new_node'] = (new_node.x, new_node.y)
                
                # 检查是否到达目标
                if self.rrt.distance(new_node, self.rrt.goal) <= self.rrt.step_size:
                    final_node = self.rrt.steer(new_node, self.rrt.goal)
                    if not self.rrt.check_collision(new_node, final_node):
                        final_node.parent = new_node
                        self.found_path = self.rrt.generate_final_course(final_node)
                        print(f"✅ 在第 {i+1} 次迭代找到路径!")
                        break
        
        print(f"📊 记录完成: {len(self.animation_data)} 个步骤")
    
    def show_animation(self):
        """显示动画"""
        self.run_planning_with_recording()
        
        def animate_step(frame):
            if frame >= len(self.animation_data):
                return
            
            step_data = self.animation_data[frame]
            
            # 清除动态元素，保留静态背景
            for artist in self.ax.collections[2:]:
                artist.remove()
            for line in self.ax.lines[1:]:
                line.remove()
            
            # 绘制树
            nodes = step_data['nodes']
            for node in nodes:
                if node.parent:
                    self.ax.plot([node.x, node.parent.x], 
                               [node.y, node.parent.y], 
                               'g-', alpha=0.6, linewidth=1.5)
            
            # 绘制节点
            if len(nodes) > 1:
                node_x = [node.x for node in nodes[1:]]
                node_y = [node.y for node in nodes[1:]]
                self.ax.scatter(node_x, node_y, c='lightgreen', s=30, alpha=0.8)
            
            # 绘制采样点
            if step_data['sampled_point']:
                sx, sy = step_data['sampled_point']
                self.ax.scatter(sx, sy, c='orange', s=50, marker='x', alpha=0.8)
            
            # 高亮新节点
            if step_data['new_node']:
                nx, ny = step_data['new_node']
                self.ax.scatter(nx, ny, c='yellow', s=80, marker='o', 
                              edgecolor='black', linewidth=2)
            
            # 绘制找到的路径
            if self.found_path and frame == len(self.animation_data) - 1:
                path_x = [point[0] for point in self.found_path]
                path_y = [point[1] for point in self.found_path]
                self.ax.plot(path_x, path_y, 'b-', linewidth=4, alpha=0.8)
            
            self.ax.set_title(f'RRT算法动态演示 - 迭代 {step_data["iteration"]+1}')
        
        ani = animation.FuncAnimation(
            self.fig, animate_step,
            frames=len(self.animation_data),
            interval=self.interval,
            repeat=True
        )
        
        plt.show()
        return ani

def demo_basic_rrt_animation():
    """基础RRT动画演示"""
    print("🎥 RRT基础算法动画演示")
    
    start = (2, 2)
    goal = (18, 18)
    obstacles = [(8, 8, 2.0), (15, 5, 1.5), (5, 15, 1.8)]
    boundary = (0, 20, 0, 20)
    
    rrt = RRTBasic(
        start=start, goal=goal, obstacle_list=obstacles, boundary=boundary,
        step_size=1.5, goal_sample_rate=0.1, max_iter=200
    )
    
    animator = RRTAnimator(rrt, interval=100)
    return animator.show_animation()

if __name__ == "__main__":
    demo_basic_rrt_animation() 