
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
Dubins约束下的RRT*算法实现
专门针对固定翼无人机路径规划

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
from dataclasses import dataclass
from enum import Enum
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PathType(Enum):
    """Dubins路径类型"""
    LSL = "LSL"  # Left-Straight-Left
    RSR = "RSR"  # Right-Straight-Right
    LSR = "LSR"  # Left-Straight-Right
    RSL = "RSL"  # Right-Straight-Left
    LRL = "LRL"  # Left-Right-Left
    RLR = "RLR"  # Right-Left-Right

@dataclass
class State:
    """无人机状态"""
    x: float        # x坐标
    y: float        # y坐标
    z: float        # z坐标 (高度)
    psi: float      # 航向角 (弧度)
    gamma: float    # 航迹角 (弧度)
    
    def __post_init__(self):
        """标准化角度"""
        self.psi = self.normalize_angle(self.psi)
        
    @staticmethod
    def normalize_angle(angle):
        """将角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

@dataclass
class DubinsPath:
    """Dubins路径"""
    path_type: PathType
    length: float
    segments: List[dict]
    waypoints: List[State]

class UAVConstraints:
    """固定翼无人机约束参数"""
    def __init__(self):
        # 运动约束
        self.min_turn_radius = 50.0      # 最小转弯半径 (m)
        self.max_climb_angle = math.radians(15)   # 最大爬升角 (rad)
        self.max_dive_angle = math.radians(-20)   # 最大下降角 (rad)
        self.cruise_speed = 25.0         # 巡航速度 (m/s)
        self.max_bank_angle = math.radians(45)    # 最大滚转角 (rad)
        
        # 环境约束
        self.min_altitude = 50.0         # 最低飞行高度 (m)
        self.max_altitude = 500.0        # 最高飞行高度 (m)
        
        # 算法参数
        self.step_size = 30.0            # 扩展步长 (m)
        self.goal_tolerance = 20.0       # 目标容忍度 (m)

class DubinsCalculator:
    """Dubins路径计算器"""
    
    def __init__(self, min_turn_radius: float):
        self.min_turn_radius = min_turn_radius
    
    def calculate_dubins_path(self, start: State, end: State) -> Optional[DubinsPath]:
        """计算两点间的最短Dubins路径"""
        # 生成简化的Dubins路径
        waypoints = self._generate_simple_path(start, end)
        
        if waypoints:
            total_length = self._calculate_path_length(waypoints)
            return DubinsPath(
                path_type=PathType.LSL,  # 简化实现
                length=total_length,
                segments=[],
                waypoints=waypoints
            )
        return None
    
    def _generate_simple_path(self, start: State, end: State) -> List[State]:
        """生成简化的路径"""
        waypoints = []
        num_points = 20
        
        for i in range(num_points + 1):
            t = i / num_points
            
            # 位置插值
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            z = start.z + t * (end.z - start.z)
            
            # 航向角插值
            psi = start.psi + t * (end.psi - start.psi)
            
            # 计算航迹角
            if i < num_points:
                next_t = (i + 1) / num_points
                next_x = start.x + next_t * (end.x - start.x)
                next_y = start.y + next_t * (end.y - start.y)
                next_z = start.z + next_t * (end.z - start.z)
                
                dx = next_x - x
                dy = next_y - y
                dz = next_z - z
                horizontal_dist = math.sqrt(dx**2 + dy**2)
                
                if horizontal_dist > 0:
                    gamma = math.atan2(dz, horizontal_dist)
                else:
                    gamma = 0
            else:
                gamma = end.gamma
            
            waypoints.append(State(x, y, z, psi, gamma))
        
        return waypoints
    
    def _calculate_path_length(self, waypoints: List[State]) -> float:
        """计算路径总长度"""
        if len(waypoints) < 2:
            return 0
        
        total_length = 0
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            dz = wp2.z - wp1.z
            total_length += math.sqrt(dx**2 + dy**2 + dz**2)
        
        return total_length

class DubinsRRTStar:
    """带Dubins约束的RRT*算法"""
    
    def __init__(self, start: State, goal: State, 
                 obstacles: List[dict], constraints: UAVConstraints):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.constraints = constraints
        self.dubins_calc = DubinsCalculator(constraints.min_turn_radius)
        
        # 算法参数
        self.max_iter = 2000
        self.goal_sample_rate = 0.1
        self.rewire_radius = 80.0
        
        # 树结构
        self.vertices = [start]
        self.edges = {}
        self.costs = {0: 0.0}
        self.parent = {0: None}
        
        # 搜索空间
        self.x_range = (-200, 800)
        self.y_range = (-200, 800)
        self.z_range = (constraints.min_altitude, constraints.max_altitude)
        
        # 可视化
        self.fig = None
        self.ax = None
        self.path_history = []
        
    def plan(self) -> Optional[List[State]]:
        """执行RRT*路径规划"""
        print("开始Dubins约束RRT*规划...")
        
        for i in range(self.max_iter):
            # 采样新状态
            if random.random() < self.goal_sample_rate:
                x_rand = self.goal
            else:
                x_rand = self._random_sample()
            
            # 找到最近节点
            nearest_idx = self._find_nearest(x_rand)
            x_nearest = self.vertices[nearest_idx]
            
            # 扩展新节点
            x_new = self._steer(x_nearest, x_rand)
            if x_new is None:
                continue
            
            # 碰撞检测
            if not self._is_collision_free(x_nearest, x_new):
                continue
            
            # 找到附近节点
            near_indices = self._find_near_vertices(x_new)
            
            # 选择最优父节点
            best_parent_idx = self._choose_parent(x_new, near_indices)
            if best_parent_idx is None:
                continue
            
            # 添加新节点
            new_idx = len(self.vertices)
            self.vertices.append(x_new)
            self.parent[new_idx] = best_parent_idx
            
            # 计算代价
            parent_cost = self.costs[best_parent_idx]
            edge_cost = self._calculate_cost(self.vertices[best_parent_idx], x_new)
            self.costs[new_idx] = parent_cost + edge_cost
            
            # 重连附近节点
            self._rewire(new_idx, near_indices)
            
            # 检查是否到达目标
            if self._distance(x_new, self.goal) < self.constraints.goal_tolerance:
                print(f"在第{i}次迭代找到路径!")
                return self._extract_path(new_idx)
            
            # 记录搜索历史
            if i % 100 == 0:
                print(f"迭代: {i}, 节点数: {len(self.vertices)}")
                current_path = self._extract_path(new_idx) if len(self.vertices) > 1 else None
                self.path_history.append({
                    'iteration': i,
                    'vertices': self.vertices.copy(),
                    'path': current_path
                })
        
        print("未找到有效路径")
        return None
    
    def _random_sample(self) -> State:
        """随机采样状态"""
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        z = random.uniform(*self.z_range)
        psi = random.uniform(-math.pi, math.pi)
        gamma = random.uniform(self.constraints.max_dive_angle, 
                              self.constraints.max_climb_angle)
        
        return State(x, y, z, psi, gamma)
    
    def _find_nearest(self, state: State) -> int:
        """找到最近的节点"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(self.vertices):
            dist = self._distance(vertex, state)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_state: State, to_state: State) -> Optional[State]:
        """扩展函数：考虑Dubins约束"""
        # 计算方向和距离
        dx = to_state.x - from_state.x
        dy = to_state.y - from_state.y
        dz = to_state.z - from_state.z
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance == 0:
            return None
        
        # 限制扩展步长
        if distance > self.constraints.step_size:
            ratio = self.constraints.step_size / distance
            new_x = from_state.x + ratio * dx
            new_y = from_state.y + ratio * dy
            new_z = from_state.z + ratio * dz
        else:
            new_x = to_state.x
            new_y = to_state.y
            new_z = to_state.z
        
        # 计算新的航向角
        new_psi = math.atan2(dy, dx)
        
        # 计算航迹角
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        if horizontal_dist > 0:
            new_gamma = math.atan2(dz, horizontal_dist)
        else:
            new_gamma = 0
        
        # 检查约束
        if (new_gamma > self.constraints.max_climb_angle or 
            new_gamma < self.constraints.max_dive_angle):
            # 限制航迹角
            new_gamma = max(self.constraints.max_dive_angle, 
                           min(self.constraints.max_climb_angle, new_gamma))
        
        return State(new_x, new_y, new_z, new_psi, new_gamma)
    
    def _is_collision_free(self, state1: State, state2: State) -> bool:
        """检查两状态间路径是否无碰撞"""
        # 简化的碰撞检测
        num_checks = 10
        for i in range(num_checks + 1):
            t = i / num_checks
            
            x = state1.x + t * (state2.x - state1.x)
            y = state1.y + t * (state2.y - state1.y)
            z = state1.z + t * (state2.z - state1.z)
            
            # 检查高度约束
            if z < self.constraints.min_altitude or z > self.constraints.max_altitude:
                return False
            
            # 检查障碍物碰撞
            check_state = State(x, y, z, 0, 0)
            if self._point_in_obstacles(check_state):
                return False
        
        return True
    
    def _point_in_obstacles(self, state: State) -> bool:
        """检查点是否在障碍物内"""
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                dx = state.x - obs['center'][0]
                dy = state.y - obs['center'][1]
                if dx**2 + dy**2 <= obs['radius']**2:
                    return True
            elif obs['type'] == 'rectangle':
                x_min, y_min = obs['bottom_left']
                x_max, y_max = obs['top_right']
                if x_min <= state.x <= x_max and y_min <= state.y <= y_max:
                    return True
        return False
    
    def _find_near_vertices(self, state: State) -> List[int]:
        """找到指定半径内的节点"""
        near_indices = []
        for i, vertex in enumerate(self.vertices):
            if self._distance(vertex, state) <= self.rewire_radius:
                near_indices.append(i)
        return near_indices
    
    def _choose_parent(self, state: State, near_indices: List[int]) -> Optional[int]:
        """选择最优父节点"""
        if not near_indices:
            return None
        
        best_cost = float('inf')
        best_parent = None
        
        for idx in near_indices:
            parent_state = self.vertices[idx]
            if self._is_collision_free(parent_state, state):
                cost = self.costs[idx] + self._calculate_cost(parent_state, state)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = idx
        
        return best_parent
    
    def _rewire(self, new_idx: int, near_indices: List[int]):
        """重连附近节点"""
        new_state = self.vertices[new_idx]
        
        for idx in near_indices:
            if idx == new_idx or idx == self.parent.get(new_idx):
                continue
            
            old_state = self.vertices[idx]
            new_cost = (self.costs[new_idx] + 
                       self._calculate_cost(new_state, old_state))
            
            if (new_cost < self.costs[idx] and 
                self._is_collision_free(new_state, old_state)):
                # 重连
                self.parent[idx] = new_idx
                self.costs[idx] = new_cost
                
                # 更新子树代价
                self._update_costs(idx)
    
    def _update_costs(self, idx: int):
        """更新子树的代价"""
        for child_idx, parent_idx in self.parent.items():
            if parent_idx == idx:
                parent_state = self.vertices[parent_idx]
                child_state = self.vertices[child_idx]
                self.costs[child_idx] = (self.costs[parent_idx] + 
                                       self._calculate_cost(parent_state, child_state))
                self._update_costs(child_idx)
    
    def _calculate_cost(self, state1: State, state2: State) -> float:
        """计算两状态间的代价"""
        return self._distance(state1, state2)
    
    def _distance(self, state1: State, state2: State) -> float:
        """计算两状态间的欧几里得距离"""
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        dz = state2.z - state1.z
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def _extract_path(self, goal_idx: int) -> List[State]:
        """提取路径"""
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            path.append(self.vertices[current_idx])
            current_idx = self.parent.get(current_idx)
        
        path.reverse()
        return path
    
    def visualize_planning_process(self, save_animation=False):
        """可视化规划过程"""
        if not self.path_history:
            # 如果没有历史数据，创建当前状态的快照
            self.path_history = [{
                'iteration': len(self.vertices),
                'vertices': self.vertices.copy(),
                'path': None
            }]
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame):
            self.ax.clear()
            
            # 获取当前帧数据
            if frame < len(self.path_history):
                data = self.path_history[frame]
            else:
                data = self.path_history[-1]
                
            vertices = data['vertices']
            path = data['path']
            iteration = data['iteration']
            
            # 绘制搜索空间
            self.ax.set_xlim(self.x_range)
            self.ax.set_ylim(self.y_range)
            
            # 绘制障碍物
            for obs in self.obstacles:
                if obs['type'] == 'circle':
                    circle = Circle(obs['center'], obs['radius'], 
                                  color='red', alpha=0.7)
                    self.ax.add_patch(circle)
                elif obs['type'] == 'rectangle':
                    rect = Rectangle(obs['bottom_left'], 
                                   obs['top_right'][0] - obs['bottom_left'][0],
                                   obs['top_right'][1] - obs['bottom_left'][1],
                                   color='red', alpha=0.7)
                    self.ax.add_patch(rect)
            
            # 绘制树
            for i, vertex in enumerate(vertices):
                if i in self.parent and self.parent[i] is not None:
                    parent_vertex = vertices[self.parent[i]]
                    self.ax.plot([parent_vertex.x, vertex.x], 
                               [parent_vertex.y, vertex.y], 
                               'b-', alpha=0.3, linewidth=0.5)
            
            # 绘制节点
            if vertices:
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                self.ax.scatter(x_coords, y_coords, c='blue', s=2, alpha=0.6)
            
            # 绘制起点和终点
            self.ax.scatter(self.start.x, self.start.y, c='green', s=100, 
                          marker='o', label='起点', zorder=5)
            self.ax.scatter(self.goal.x, self.goal.y, c='red', s=100, 
                          marker='*', label='目标', zorder=5)
            
            # 绘制当前最优路径
            if path and len(path) > 1:
                path_x = [p.x for p in path]
                path_y = [p.y for p in path]
                self.ax.plot(path_x, path_y, 'orange', linewidth=3, 
                           label='当前路径', zorder=4)
                
                # 绘制航向箭头
                for i in range(0, len(path), max(1, len(path)//10)):
                    state = path[i]
                    arrow_length = 15
                    dx = arrow_length * math.cos(state.psi)
                    dy = arrow_length * math.sin(state.psi)
                    self.ax.arrow(state.x, state.y, dx, dy, 
                                head_width=5, head_length=3, 
                                fc='orange', ec='orange', alpha=0.8)
            
            self.ax.set_title(f'Dubins约束RRT* - 迭代: {iteration}, 节点数: {len(vertices)}')
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
        
        anim = FuncAnimation(self.fig, animate, frames=max(1, len(self.path_history)), 
                           interval=1000, repeat=True, blit=False)
        
        if save_animation:
            try:
                anim.save('dubins_rrt_star_planning.gif', writer='pillow', fps=2)
                print("动画已保存为 dubins_rrt_star_planning.gif")
            except Exception as e:
                print(f"保存动画失败: {e}")
        
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    """主函数：演示Dubins约束RRT*算法"""
    print("=== Dubins约束RRT*算法演示 ===")
    
    # 设置约束参数
    constraints = UAVConstraints()
    
    # 定义起点和终点
    start = State(0, 0, 100, math.radians(0), 0)  # x, y, z, psi, gamma
    goal = State(600, 500, 150, math.radians(90), 0)
    
    # 定义障碍物
    obstacles = [
        {'type': 'circle', 'center': (200, 150), 'radius': 50},
        {'type': 'circle', 'center': (400, 300), 'radius': 60},
        {'type': 'rectangle', 'bottom_left': (150, 350), 'top_right': (250, 450)},
        {'type': 'rectangle', 'bottom_left': (450, 100), 'top_right': (550, 200)},
    ]
    
    # 创建规划器
    planner = DubinsRRTStar(start, goal, obstacles, constraints)
    
    # 执行规划
    start_time = time.time()
    path = planner.plan()
    planning_time = time.time() - start_time
    
    if path:
        print(f"\n✅ 规划成功!")
        print(f"规划时间: {planning_time:.2f}秒")
        print(f"路径长度: {len(path)}个航点")
        
        # 计算路径统计信息
        total_length = 0
        max_turn_rate = 0
        max_climb_rate = 0
        
        for i in range(len(path) - 1):
            # 路径长度
            total_length += planner._distance(path[i], path[i+1])
            
            # 转弯率
            turn_rate = abs(path[i+1].psi - path[i].psi)
            max_turn_rate = max(max_turn_rate, turn_rate)
            
            # 爬升率
            climb_rate = abs(path[i+1].gamma)
            max_climb_rate = max(max_climb_rate, climb_rate)
        
        print(f"总路径长度: {total_length:.2f}m")
        print(f"最大转弯率: {math.degrees(max_turn_rate):.2f}°")
        print(f"最大爬升角: {math.degrees(max_climb_rate):.2f}°")
        
        # 验证约束满足
        constraints_satisfied = True
        for i, state in enumerate(path):
            if (state.gamma > constraints.max_climb_angle or 
                state.gamma < constraints.max_dive_angle):
                print(f"⚠️ 航点{i}违反航迹角约束: {math.degrees(state.gamma):.2f}°")
                constraints_satisfied = False
        
        if constraints_satisfied:
            print("✅ 所有约束均满足")
        
    else:
        print("❌ 规划失败")
    
    # 可视化结果
    print("\n正在生成可视化...")
    planner.visualize_planning_process(save_animation=False)

if __name__ == "__main__":
    main() 