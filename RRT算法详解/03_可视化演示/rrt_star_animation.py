#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRT*算法动态演示和教学程序

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import math
import random
from typing import List, Tuple, Optional
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RRTStarDemo:
    """RRT*算法演示类"""
    
    def __init__(self):
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 算法参数
        self.max_iter = 200
        self.step_size = 30
        self.goal_tolerance = 25
        self.rewire_radius = 50
        self.goal_sample_rate = 0.1
        
        # 搜索空间
        self.x_range = (0, 600)
        self.y_range = (0, 400)
        
        # 起点和终点
        self.start = [50, 200]
        self.goal = [550, 200]
        
        # 障碍物
        self.obstacles = [
            {'type': 'circle', 'center': [200, 150], 'radius': 40},
            {'type': 'circle', 'center': [400, 280], 'radius': 50},
            {'type': 'rectangle', 'corner': [150, 300], 'width': 100, 'height': 60},
            {'type': 'rectangle', 'corner': [350, 80], 'width': 80, 'height': 70},
        ]
        
        # 树结构
        self.vertices = []
        self.edges = []
        self.parent = {}
        self.costs = {}
        
        # 重置树
        self._reset_tree()
        
    def _reset_tree(self):
        """重置搜索树"""
        self.vertices = [self.start.copy()]
        self.edges = []
        self.parent = {0: None}
        self.costs = {0: 0.0}
        
    def _distance(self, p1, p2):
        """计算距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _random_sample(self):
        """随机采样"""
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()
        else:
            return [random.uniform(*self.x_range), random.uniform(*self.y_range)]
    
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
            return to_point.copy()
        else:
            unit_vector = [direction[0] / distance, direction[1] / distance]
            return [
                from_point[0] + self.step_size * unit_vector[0],
                from_point[1] + self.step_size * unit_vector[1]
            ]
    
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
        
        return best_parent
    
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
            path.append(self.vertices[current_idx].copy())
            current_idx = self.parent.get(current_idx)
        
        path.reverse()
        return path
    
    def step(self):
        """执行一步算法"""
        # 采样
        x_rand = self._random_sample()
        
        # 最近顶点
        nearest_idx = self._nearest_vertex(x_rand)
        x_nearest = self.vertices[nearest_idx]
        
        # 扩展
        x_new = self._steer(x_nearest, x_rand)
        
        # 碰撞检测
        if not self._is_collision_free(x_nearest, x_new):
            return False
        
        # 附近顶点
        near_indices = self._near_vertices(x_new)
        
        # 选择父节点
        best_parent_idx = self._choose_parent(x_new, near_indices)
        if best_parent_idx is None:
            return False
        
        # 添加新顶点
        new_idx = len(self.vertices)
        self.vertices.append(x_new)
        self.parent[new_idx] = best_parent_idx
        self.costs[new_idx] = (self.costs[best_parent_idx] + 
                              self._distance(self.vertices[best_parent_idx], x_new))
        
        # 添加边
        self.edges.append((best_parent_idx, new_idx))
        
        # 重连
        self._rewire(new_idx, near_indices)
        
        return True
    
    def animate(self, interval=100):
        """创建动画"""
        
        def animate_frame(frame):
            # 执行算法步骤
            if frame < self.max_iter:
                self.step()
            
            # 清空图形
            self.ax_main.clear()
            self.ax_stats.clear()
            
            # 设置主图
            self.ax_main.set_xlim(self.x_range)
            self.ax_main.set_ylim(self.y_range)
            self.ax_main.set_xlabel('X (m)')
            self.ax_main.set_ylabel('Y (m)')
            self.ax_main.set_title(f'RRT*算法演示 - 迭代: {frame}')
            self.ax_main.grid(True, alpha=0.3)
            self.ax_main.set_aspect('equal')
            
            # 绘制障碍物
            for obs in self.obstacles:
                if obs['type'] == 'circle':
                    circle = Circle(obs['center'], obs['radius'], 
                                  color='red', alpha=0.7)
                    self.ax_main.add_patch(circle)
                elif obs['type'] == 'rectangle':
                    rect = Rectangle(obs['corner'], obs['width'], obs['height'],
                                   color='red', alpha=0.7)
                    self.ax_main.add_patch(rect)
            
            # 绘制起点终点
            self.ax_main.plot(self.start[0], self.start[1], 'go', 
                            markersize=10, label='起点')
            self.ax_main.plot(self.goal[0], self.goal[1], 'r*', 
                            markersize=15, label='目标')
            
            # 绘制搜索树
            for edge in self.edges:
                p1 = self.vertices[edge[0]]
                p2 = self.vertices[edge[1]]
                self.ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                                'b-', alpha=0.3, linewidth=0.8)
            
            # 绘制顶点
            for vertex in self.vertices:
                self.ax_main.plot(vertex[0], vertex[1], 'bo', 
                                markersize=3, alpha=0.6)
            
            # 绘制路径
            path = self._extract_path()
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                self.ax_main.plot(path_x, path_y, 'orange', 
                                linewidth=3, label='当前路径')
            
            self.ax_main.legend()
            
            # 统计信息
            self.ax_stats.text(0.1, 0.8, f'迭代次数: {frame}', 
                             transform=self.ax_stats.transAxes, fontsize=12)
            self.ax_stats.text(0.1, 0.7, f'顶点数量: {len(self.vertices)}', 
                             transform=self.ax_stats.transAxes, fontsize=12)
            self.ax_stats.text(0.1, 0.6, f'边数量: {len(self.edges)}', 
                             transform=self.ax_stats.transAxes, fontsize=12)
            
            if path:
                path_length = sum(self._distance(path[i], path[i+1]) 
                                for i in range(len(path)-1))
                self.ax_stats.text(0.1, 0.5, f'路径长度: {path_length:.1f}m', 
                                 transform=self.ax_stats.transAxes, fontsize=12)
                self.ax_stats.text(0.1, 0.4, '路径状态: 已找到', 
                                 transform=self.ax_stats.transAxes, fontsize=12, color='green')
            else:
                self.ax_stats.text(0.1, 0.4, '路径状态: 搜索中', 
                                 transform=self.ax_stats.transAxes, fontsize=12, color='orange')
            
            self.ax_stats.set_title('算法统计信息')
            self.ax_stats.axis('off')
        
        anim = FuncAnimation(self.fig, animate_frame, frames=self.max_iter,
                           interval=interval, repeat=False, blit=False)
        
        plt.tight_layout()
        plt.show()
        return anim

def main():
    """主函数"""
    print("=== RRT*算法动态演示 ===")
    
    demo = RRTStarDemo()
    
    print("开始演示...")
    demo.animate(interval=100)

if __name__ == "__main__":
    main() 