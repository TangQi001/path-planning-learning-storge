#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRT算法对比分析程序
比较RRT、RRT*、Informed RRT*等算法的性能

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
import random
import time
from typing import List, Dict, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RRTComparison:
    """RRT算法对比类"""
    
    def __init__(self):
        # 环境设置
        self.start = [50, 200]
        self.goal = [550, 200]
        self.bounds = [0, 600, 0, 400]
        
        # 障碍物
        self.obstacles = [
            {'type': 'circle', 'center': [200, 150], 'radius': 40},
            {'type': 'circle', 'center': [400, 280], 'radius': 50},
            {'type': 'rectangle', 'corner': [150, 300], 'width': 100, 'height': 60},
        ]
        
        # 算法参数
        self.max_iter = 500
        self.step_size = 30
        self.goal_tolerance = 25
        self.rewire_radius = 50
        
        # 结果存储
        self.results = {}
    
    def _distance(self, p1, p2):
        """计算距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _random_sample(self):
        """随机采样"""
        return [random.uniform(self.bounds[0], self.bounds[1]), 
                random.uniform(self.bounds[2], self.bounds[3])]
    
    def _nearest_vertex(self, vertices, point):
        """找最近顶点"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(vertices):
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
    
    def _extract_path(self, vertices, parent):
        """提取路径"""
        # 找最近目标的节点
        best_idx = None
        min_dist = float('inf')
        
        for i, vertex in enumerate(vertices):
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
            path.append(vertices[current_idx])
            current_idx = parent.get(current_idx)
        
        path.reverse()
        return path
    
    def _calculate_path_length(self, path):
        """计算路径长度"""
        if len(path) < 2:
            return 0
        
        length = 0
        for i in range(len(path) - 1):
            length += self._distance(path[i], path[i+1])
        return length
    
    def run_basic_rrt(self):
        """运行基本RRT算法"""
        start_time = time.time()
        
        vertices = [self.start]
        edges = []
        parent = {0: None}
        
        for iteration in range(self.max_iter):
            # 采样
            x_rand = self._random_sample()
            
            # 最近顶点
            nearest_idx = self._nearest_vertex(vertices, x_rand)
            x_nearest = vertices[nearest_idx]
            
            # 扩展
            x_new = self._steer(x_nearest, x_rand)
            
            # 碰撞检测
            if not self._is_collision_free(x_nearest, x_new):
                continue
            
            # 添加新顶点
            new_idx = len(vertices)
            vertices.append(x_new)
            parent[new_idx] = nearest_idx
            edges.append((nearest_idx, new_idx))
            
            # 检查是否到达目标
            if self._distance(x_new, self.goal) < self.goal_tolerance:
                break
        
        planning_time = time.time() - start_time
        path = self._extract_path(vertices, parent)
        
        return {
            'algorithm': 'RRT',
            'vertices': vertices,
            'edges': edges,
            'path': path,
            'planning_time': planning_time,
            'path_length': self._calculate_path_length(path),
            'nodes_generated': len(vertices)
        }
    
    def run_rrt_star(self):
        """运行RRT*算法"""
        start_time = time.time()
        
        vertices = [self.start]
        edges = []
        parent = {0: None}
        costs = {0: 0.0}
        
        for iteration in range(self.max_iter):
            # 采样
            x_rand = self._random_sample()
            
            # 最近顶点
            nearest_idx = self._nearest_vertex(vertices, x_rand)
            x_nearest = vertices[nearest_idx]
            
            # 扩展
            x_new = self._steer(x_nearest, x_rand)
            
            # 碰撞检测
            if not self._is_collision_free(x_nearest, x_new):
                continue
            
            # 找附近顶点
            near_indices = []
            for i, vertex in enumerate(vertices):
                if self._distance(vertex, x_new) <= self.rewire_radius:
                    near_indices.append(i)
            
            # 选择最优父节点
            best_cost = float('inf')
            best_parent = None
            
            for idx in near_indices:
                if self._is_collision_free(vertices[idx], x_new):
                    cost = costs[idx] + self._distance(vertices[idx], x_new)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = idx
            
            if best_parent is None:
                continue
            
            # 添加新顶点
            new_idx = len(vertices)
            vertices.append(x_new)
            parent[new_idx] = best_parent
            costs[new_idx] = best_cost
            edges.append((best_parent, new_idx))
            
            # 重连
            for idx in near_indices:
                if idx == new_idx or idx == best_parent:
                    continue
                
                new_cost = costs[new_idx] + self._distance(x_new, vertices[idx])
                
                if (new_cost < costs[idx] and 
                    self._is_collision_free(x_new, vertices[idx])):
                    
                    # 重连
                    old_parent = parent[idx]
                    parent[idx] = new_idx
                    costs[idx] = new_cost
                    
                    # 更新边
                    if old_parent is not None:
                        old_edge = (old_parent, idx)
                        if old_edge in edges:
                            edges.remove(old_edge)
                    
                    edges.append((new_idx, idx))
            
            # 检查是否到达目标
            if self._distance(x_new, self.goal) < self.goal_tolerance:
                pass  # RRT*继续优化
        
        planning_time = time.time() - start_time
        path = self._extract_path(vertices, parent)
        
        return {
            'algorithm': 'RRT*',
            'vertices': vertices,
            'edges': edges,
            'path': path,
            'planning_time': planning_time,
            'path_length': self._calculate_path_length(path),
            'nodes_generated': len(vertices)
        }
    
    def run_comparison(self, num_trials=5):
        """运行算法比较"""
        print("开始算法比较...")
        
        rrt_results = []
        rrt_star_results = []
        
        for trial in range(num_trials):
            print(f"试验 {trial + 1}/{num_trials}")
            
            # 重置随机种子以确保公平比较
            random.seed(trial)
            
            # 运行RRT
            rrt_result = self.run_basic_rrt()
            rrt_results.append(rrt_result)
            
            # 重置随机种子
            random.seed(trial)
            
            # 运行RRT*
            rrt_star_result = self.run_rrt_star()
            rrt_star_results.append(rrt_star_result)
        
        self.results = {
            'RRT': rrt_results,
            'RRT*': rrt_star_results
        }
        
        print("比较完成!")
    
    def analyze_results(self):
        """分析结果"""
        print("\n" + "="*50)
        print("算法性能比较结果")
        print("="*50)
        
        for alg_name, results in self.results.items():
            print(f"\n{alg_name}:")
            
            # 计算统计数据
            successful_trials = [r for r in results if len(r['path']) > 0]
            success_rate = len(successful_trials) / len(results)
            
            if successful_trials:
                avg_path_length = np.mean([r['path_length'] for r in successful_trials])
                std_path_length = np.std([r['path_length'] for r in successful_trials])
            else:
                avg_path_length = 0
                std_path_length = 0
            
            avg_time = np.mean([r['planning_time'] for r in results])
            avg_nodes = np.mean([r['nodes_generated'] for r in results])
            
            print(f"  成功率: {success_rate:.1%}")
            print(f"  平均路径长度: {avg_path_length:.2f} ± {std_path_length:.2f} m")
            print(f"  平均规划时间: {avg_time:.3f} s")
            print(f"  平均节点数: {avg_nodes:.1f}")
    
    def visualize_comparison(self):
        """可视化比较结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 路径示例对比
        axes[0, 0].set_xlim(self.bounds[0], self.bounds[1])
        axes[0, 0].set_ylim(self.bounds[2], self.bounds[3])
        axes[0, 0].set_title('路径对比示例')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].set_aspect('equal')
        
        # 绘制障碍物
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.7)
                axes[0, 0].add_patch(circle)
            elif obs['type'] == 'rectangle':
                rect = Rectangle(obs['corner'], obs['width'], obs['height'], 
                               color='red', alpha=0.7)
                axes[0, 0].add_patch(rect)
        
        # 绘制起点和终点
        axes[0, 0].plot(self.start[0], self.start[1], 'go', markersize=10, label='起点')
        axes[0, 0].plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='目标')
        
        # 绘制路径
        colors = ['blue', 'green']
        for i, (alg_name, color) in enumerate(zip(['RRT', 'RRT*'], colors)):
            if alg_name in self.results:
                for result in self.results[alg_name]:
                    if result['path']:
                        path = result['path']
                        path_x = [p[0] for p in path]
                        path_y = [p[1] for p in path]
                        axes[0, 0].plot(path_x, path_y, color=color, 
                                       linewidth=2, alpha=0.6, label=alg_name)
                        break
        
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率对比
        alg_names = list(self.results.keys())
        success_rates = []
        
        for alg_name in alg_names:
            results = self.results[alg_name]
            successful = sum(1 for r in results if len(r['path']) > 0)
            success_rates.append(successful / len(results))
        
        axes[0, 1].bar(alg_names, success_rates, color=['blue', 'green'])
        axes[0, 1].set_title('成功率比较')
        axes[0, 1].set_ylabel('成功率')
        axes[0, 1].set_ylim(0, 1.1)
        
        for i, v in enumerate(success_rates):
            axes[0, 1].text(i, v + 0.05, f'{v:.1%}', ha='center')
        
        # 3. 路径长度对比
        path_lengths = {}
        for alg_name in alg_names:
            lengths = [r['path_length'] for r in self.results[alg_name] if r['path']]
            path_lengths[alg_name] = lengths
        
        axes[1, 0].boxplot(path_lengths.values(), labels=path_lengths.keys())
        axes[1, 0].set_title('路径长度分布')
        axes[1, 0].set_ylabel('路径长度 (m)')
        
        # 4. 规划时间对比
        planning_times = {}
        for alg_name in alg_names:
            times = [r['planning_time'] for r in self.results[alg_name]]
            planning_times[alg_name] = times
        
        axes[1, 1].boxplot(planning_times.values(), labels=planning_times.keys())
        axes[1, 1].set_title('规划时间分布')
        axes[1, 1].set_ylabel('规划时间 (s)')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("=== RRT算法对比分析 ===")
    
    # 创建比较实例
    comparison = RRTComparison()
    
    # 运行比较
    comparison.run_comparison(num_trials=3)
    
    # 分析结果
    comparison.analyze_results()
    
    # 可视化
    comparison.visualize_comparison()

if __name__ == "__main__":
    main() 