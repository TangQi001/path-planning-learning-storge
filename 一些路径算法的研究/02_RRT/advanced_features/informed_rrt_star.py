
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
Informed RRT*算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse
import math
import random
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InformedRRTStar:
    """Informed RRT*算法类"""
    
    def __init__(self, start, goal, obstacles, bounds):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds
        
        # 算法参数
        self.max_iter = 1000
        self.step_size = 30
        self.goal_tolerance = 25
        self.rewire_radius = 50
        
        # Informed RRT*特有参数
        self.best_path_cost = float('inf')
        self.informed_sampling = False
        self.c_min = self._distance(start, goal)
        
        # 搜索树
        self.vertices = [start]
        self.edges = []
        self.parent = {0: None}
        self.costs = {0: 0.0}
        
        # 历史记录
        self.cost_history = []
        self.sample_counts = {'uniform': 0, 'informed': 0}
    
    def _distance(self, p1, p2):
        """计算距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _random_sample(self):
        """智能采样"""
        if not self.informed_sampling or random.random() < 0.1:
            # Uniform sampling
            self.sample_counts['uniform'] += 1
            return [random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])]
        else:
            # Informed sampling
            self.sample_counts['informed'] += 1
            return self._sample_from_ellipse()
    
    def _sample_from_ellipse(self):
        """从椭圆内采样"""
        if self.best_path_cost >= float('inf'):
            return [random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])]
        
        # 椭圆参数
        c_max = self.best_path_cost
        c_min = self.c_min
        
        # 椭圆中心
        center = [(self.start[0] + self.goal[0]) / 2,
                 (self.start[1] + self.goal[1]) / 2]
        
        # 椭圆方向角
        theta = math.atan2(self.goal[1] - self.start[1], 
                          self.goal[0] - self.start[0])
        
        # 椭圆半轴
        a = c_max / 2  # 长半轴
        b = math.sqrt(c_max**2 - c_min**2) / 2  # 短半轴
        
        if b <= 0:
            return [random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])]
        
        # 在椭圆内采样
        for _ in range(50):
            # 在单位圆内采样
            angle = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.uniform(0, 1))
            x_unit = r * math.cos(angle)
            y_unit = r * math.sin(angle)
            
            # 缩放到椭圆
            x_ellipse = a * x_unit
            y_ellipse = b * y_unit
            
            # 旋转和平移
            x_world = (x_ellipse * math.cos(theta) - 
                      y_ellipse * math.sin(theta) + center[0])
            y_world = (x_ellipse * math.sin(theta) + 
                      y_ellipse * math.cos(theta) + center[1])
            
            # 检查边界
            if (self.bounds[0] <= x_world <= self.bounds[1] and
                self.bounds[2] <= y_world <= self.bounds[3]):
                return [x_world, y_world]
        
        # 回退到uniform采样
        return [random.uniform(self.bounds[0], self.bounds[1]),
                random.uniform(self.bounds[2], self.bounds[3])]
    
    def _nearest_vertex(self, point):
        """找最近顶点"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(self.vertices):
            dist = self._distance(vertex, point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_point, to_point):
        """扩展函数"""
        direction = [to_point[0] - from_point[0], to_point[1] - from_point[1]]
        distance = math.sqrt(direction[0]**2 + direction[1]**2)
        
        if distance <= self.step_size:
            return to_point
        else:
            unit_vector = [direction[0] / distance, direction[1] / distance]
            return [
                from_point[0] + self.step_size * unit_vector[0],
                from_point[1] + self.step_size * unit_vector[1]
            ]
    
    def _is_collision_free(self, p1, p2):
        """碰撞检测"""
        num_checks = int(self._distance(p1, p2) / 5) + 1
        
        for i in range(num_checks + 1):
            t = i / num_checks if num_checks > 0 else 0
            check_point = [
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1])
            ]
            
            if self._point_in_obstacle(check_point):
                return False
        
        return True
    
    def _point_in_obstacle(self, point):
        """检查点是否在障碍物内"""
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                center = obs['center']
                radius = obs['radius']
                if self._distance(point, center) <= radius:
                    return True
            elif obs['type'] == 'rectangle':
                corner = obs['corner']
                width = obs['width']
                height = obs['height']
                if (corner[0] <= point[0] <= corner[0] + width and 
                    corner[1] <= point[1] <= corner[1] + height):
                    return True
        return False
    
    def _near_vertices(self, point):
        """找附近顶点"""
        near_indices = []
        for i, vertex in enumerate(self.vertices):
            if self._distance(vertex, point) <= self.rewire_radius:
                near_indices.append(i)
        return near_indices
    
    def _choose_parent(self, new_point, near_indices):
        """选择最优父节点"""
        best_cost = float('inf')
        best_parent = None
        
        for idx in near_indices:
            if self._is_collision_free(self.vertices[idx], new_point):
                cost = self.costs[idx] + self._distance(self.vertices[idx], new_point)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = idx
        
        return best_parent, best_cost
    
    def _rewire(self, new_idx, near_indices):
        """重连操作"""
        new_point = self.vertices[new_idx]
        
        for idx in near_indices:
            if idx == new_idx or idx == self.parent.get(new_idx):
                continue
            
            new_cost = self.costs[new_idx] + self._distance(new_point, self.vertices[idx])
            
            if (new_cost < self.costs[idx] and 
                self._is_collision_free(new_point, self.vertices[idx])):
                
                # 重连
                old_parent = self.parent[idx]
                self.parent[idx] = new_idx
                self.costs[idx] = new_cost
                
                # 更新边
                if old_parent is not None:
                    old_edge = (old_parent, idx)
                    if old_edge in self.edges:
                        self.edges.remove(old_edge)
                
                self.edges.append((new_idx, idx))
    
    def _extract_path(self):
        """提取路径"""
        # 找最近目标的节点
        best_idx = None
        min_dist = float('inf')
        
        for i, vertex in enumerate(self.vertices):
            dist = self._distance(vertex, self.goal)
            if dist < self.goal_tolerance and dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx is None:
            return []
        
        # 回溯路径
        path = []
        current_idx = best_idx
        
        while current_idx is not None:
            path.append(self.vertices[current_idx])
            current_idx = self.parent.get(current_idx)
        
        path.reverse()
        return path
    
    def _calculate_path_cost(self, path):
        """计算路径代价"""
        if len(path) < 2:
            return 0
        
        cost = 0
        for i in range(len(path) - 1):
            cost += self._distance(path[i], path[i+1])
        return cost
    
    def plan(self):
        """执行规划"""
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # 采样
            x_rand = self._random_sample()
            
            # 最近顶点
            nearest_idx = self._nearest_vertex(x_rand)
            x_nearest = self.vertices[nearest_idx]
            
            # 扩展
            x_new = self._steer(x_nearest, x_rand)
            
            # 碰撞检测
            if not self._is_collision_free(x_nearest, x_new):
                continue
            
            # 附近顶点
            near_indices = self._near_vertices(x_new)
            
            # 选择父节点
            best_parent_idx, best_cost = self._choose_parent(x_new, near_indices)
            if best_parent_idx is None:
                continue
            
            # 添加新顶点
            new_idx = len(self.vertices)
            self.vertices.append(x_new)
            self.parent[new_idx] = best_parent_idx
            self.costs[new_idx] = best_cost
            self.edges.append((best_parent_idx, new_idx))
            
            # 重连
            self._rewire(new_idx, near_indices)
            
            # 更新最优路径
            path = self._extract_path()
            if path:
                path_cost = self._calculate_path_cost(path)
                if path_cost < self.best_path_cost:
                    self.best_path_cost = path_cost
                    self.cost_history.append((iteration, path_cost))
                    
                    # 启用informed sampling
                    if not self.informed_sampling:
                        self.informed_sampling = True
                        print(f"在第 {iteration} 次迭代启用Informed采样")
        
        planning_time = time.time() - start_time
        print(f"规划完成！时间: {planning_time:.3f}s")
        
        return self._extract_path()
    
    def visualize(self, path=None):
        """可视化"""
        fig, (ax_main, ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 主图
        ax_main.set_xlim(self.bounds[0], self.bounds[1])
        ax_main.set_ylim(self.bounds[2], self.bounds[3])
        ax_main.set_xlabel('X (m)')
        ax_main.set_ylabel('Y (m)')
        ax_main.set_title('Informed RRT* 算法')
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.3)
        
        # 绘制障碍物
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.7)
                ax_main.add_patch(circle)
            elif obs['type'] == 'rectangle':
                rect = Rectangle(obs['corner'], obs['width'], obs['height'],
                               color='red', alpha=0.7)
                ax_main.add_patch(rect)
        
        # 绘制搜索树
        for edge in self.edges:
            p1 = self.vertices[edge[0]]
            p2 = self.vertices[edge[1]]
            ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        'b-', alpha=0.3, linewidth=0.5)
        
        # 绘制椭圆
        if self.informed_sampling and self.best_path_cost < float('inf'):
            center = [(self.start[0] + self.goal[0]) / 2,
                     (self.start[1] + self.goal[1]) / 2]
            
            a = self.best_path_cost / 2
            b = math.sqrt(self.best_path_cost**2 - self.c_min**2) / 2
            
            if b > 0:
                angle = math.degrees(math.atan2(self.goal[1] - self.start[1], 
                                              self.goal[0] - self.start[0]))
                
                ellipse = Ellipse(center, 2*a, 2*b, angle=angle,
                                fill=False, color='green', linewidth=2, 
                                linestyle='--')
                ax_main.add_patch(ellipse)
        
        # 绘制起点终点
        ax_main.plot(self.start[0], self.start[1], 'go', markersize=10, label='起点')
        ax_main.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='目标')
        
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax_main.plot(path_x, path_y, 'orange', linewidth=3, label='最优路径')
        
        ax_main.legend()
        
        # 收敛图
        if self.cost_history:
            iterations, costs = zip(*self.cost_history)
            ax_stats.plot(iterations, costs, 'g-', linewidth=2, marker='o', markersize=4)
            ax_stats.set_xlabel('迭代次数')
            ax_stats.set_ylabel('路径代价')
            ax_stats.set_title('收敛性分析')
            ax_stats.grid(True, alpha=0.3)
        
        # 统计信息
        total_samples = self.sample_counts['uniform'] + self.sample_counts['informed']
        stats_text = f"""统计信息:
节点数: {len(self.vertices)}
路径代价: {self.best_path_cost:.2f}
Uniform采样: {self.sample_counts['uniform']}
Informed采样: {self.sample_counts['informed']}
Informed比例: {self.sample_counts['informed']/total_samples:.1%}"""
        
        ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("=== Informed RRT* 算法演示 ===")
    
    # 环境设置
    start = [50, 200]
    goal = [550, 200]
    bounds = [0, 600, 0, 400]
    
    obstacles = [
        {'type': 'circle', 'center': [200, 150], 'radius': 40},
        {'type': 'circle', 'center': [400, 280], 'radius': 50},
        {'type': 'rectangle', 'corner': [150, 300], 'width': 100, 'height': 60},
    ]
    
    # 创建规划器
    planner = InformedRRTStar(start, goal, obstacles, bounds)
    
    # 执行规划
    path = planner.plan()
    
    # 可视化
    planner.visualize(path)

if __name__ == "__main__":
    main() 