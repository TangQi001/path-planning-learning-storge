
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
3D地形环境下的RRT路径规划
适用于固定翼无人机在复杂地形中的航迹规划

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import math
import random
import time
from typing import List, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class State3D:
    """3D状态点"""
    def __init__(self, x: float, y: float, z: float, psi: float = 0, gamma: float = 0):
        self.x = x      # X坐标
        self.y = y      # Y坐标
        self.z = z      # Z坐标（高度）
        self.psi = psi  # 航向角（弧度）
        self.gamma = gamma  # 航迹角（弧度）
    
    def to_list(self):
        return [self.x, self.y, self.z]

class TerrainMap:
    """地形地图类"""
    
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        self.x_range = x_range
        self.y_range = y_range
        
        # 创建地形网格
        self.resolution = 20  # 每个单位的网格数
        self.x_grid = np.linspace(x_range[0], x_range[1], 
                                 int((x_range[1] - x_range[0]) * self.resolution))
        self.y_grid = np.linspace(y_range[0], y_range[1], 
                                 int((y_range[1] - y_range[0]) * self.resolution))
        
        # 生成地形高度
        self._generate_terrain()
        
        # 障碍物（山峰、建筑等）
        self.obstacles = [
            {'type': 'cylinder', 'center': [200, 150], 'radius': 40, 'height': 200},
            {'type': 'cylinder', 'center': [400, 280], 'radius': 50, 'height': 150},
            {'type': 'box', 'corner': [350, 80], 'size': [80, 70, 120]},
        ]
    
    def _generate_terrain(self):
        """生成地形高度图"""
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # 基础地形：多个正弦波叠加
        Z = (30 * np.sin(X/100) * np.cos(Y/80) + 
             20 * np.sin(X/150) * np.sin(Y/120) + 
             15 * np.cos(X/200) * np.cos(Y/100) + 
             50)  # 基础高度
        
        # 添加一些山峰
        peak1 = 100 * np.exp(-((X-250)**2 + (Y-200)**2) / 5000)
        peak2 = 80 * np.exp(-((X-450)**2 + (Y-300)**2) / 4000)
        peak3 = 60 * np.exp(-((X-150)**2 + (Y-350)**2) / 3000)
        
        Z += peak1 + peak2 + peak3
        
        self.terrain_height = Z
        self.X, self.Y = X, Y
    
    def get_terrain_height(self, x: float, y: float) -> float:
        """获取指定位置的地形高度"""
        # 插值获取高度
        if (self.x_range[0] <= x <= self.x_range[1] and 
            self.y_range[0] <= y <= self.y_range[1]):
            
            # 简单的双线性插值
            x_idx = int((x - self.x_range[0]) * self.resolution)
            y_idx = int((y - self.y_range[0]) * self.resolution)
            
            x_idx = max(0, min(x_idx, len(self.x_grid) - 1))
            y_idx = max(0, min(y_idx, len(self.y_grid) - 1))
            
            return self.terrain_height[y_idx, x_idx]
        
        return 0
    
    def is_collision(self, state: State3D, safety_margin: float = 20) -> bool:
        """检查是否与地形或障碍物碰撞"""
        # 检查地形碰撞
        terrain_height = self.get_terrain_height(state.x, state.y)
        if state.z < terrain_height + safety_margin:
            return True
        
        # 检查障碍物碰撞
        for obs in self.obstacles:
            if obs['type'] == 'cylinder':
                center = obs['center']
                radius = obs['radius']
                height = obs['height']
                
                dist_2d = math.sqrt((state.x - center[0])**2 + (state.y - center[1])**2)
                if (dist_2d <= radius + safety_margin and 
                    0 <= state.z <= height + safety_margin):
                    return True
            
            elif obs['type'] == 'box':
                corner = obs['corner']
                size = obs['size']
                
                if (corner[0] - safety_margin <= state.x <= corner[0] + size[0] + safety_margin and
                    corner[1] - safety_margin <= state.y <= corner[1] + size[1] + safety_margin and
                    0 <= state.z <= size[2] + safety_margin):
                    return True
        
        return False

class UAVConstraints3D:
    """3D固定翼无人机约束"""
    
    def __init__(self):
        self.min_turn_radius = 50.0      # 最小转弯半径 (m)
        self.max_climb_angle = 15.0      # 最大爬升角 (度)
        self.max_dive_angle = -20.0      # 最大下降角 (度)
        self.cruise_speed = 25.0         # 巡航速度 (m/s)
        self.step_size = 30.0            # 扩展步长 (m)
        self.min_altitude = 50.0         # 最小飞行高度 (m)
        self.max_altitude = 500.0        # 最大飞行高度 (m)
        
        # 转换角度为弧度
        self.max_climb_angle_rad = math.radians(self.max_climb_angle)
        self.max_dive_angle_rad = math.radians(self.max_dive_angle)

class TerrainRRT3D:
    """3D地形环境下的RRT路径规划器"""
    
    def __init__(self, terrain: TerrainMap, constraints: UAVConstraints3D):
        self.terrain = terrain
        self.constraints = constraints
        
        # 算法参数
        self.max_iter = 1000
        self.goal_tolerance = 25.0
        self.goal_sample_rate = 0.1
        
        # 搜索树
        self.vertices = []
        self.edges = []
        self.parent = {}
        
        # 统计信息
        self.stats = {
            'planning_time': 0,
            'path_length': 0,
            'nodes_generated': 0,
            'path_found': False
        }
    
    def set_start_goal(self, start: State3D, goal: State3D):
        """设置起点和终点"""
        self.start = start
        self.goal = goal
        self.vertices = [start]
        self.parent = {0: None}
    
    def _distance(self, state1: State3D, state2: State3D) -> float:
        """计算3D距离"""
        return math.sqrt((state1.x - state2.x)**2 + 
                        (state1.y - state2.y)**2 + 
                        (state1.z - state2.z)**2)
    
    def _random_sample(self) -> State3D:
        """3D随机采样"""
        if random.random() < self.goal_sample_rate:
            return self.goal
        
        x = random.uniform(self.terrain.x_range[0], self.terrain.x_range[1])
        y = random.uniform(self.terrain.y_range[0], self.terrain.y_range[1])
        
        # 考虑地形高度的采样
        terrain_height = self.terrain.get_terrain_height(x, y)
        min_z = max(self.constraints.min_altitude, terrain_height + 30)
        max_z = self.constraints.max_altitude
        
        if min_z >= max_z:
            min_z = terrain_height + 10
            max_z = min_z + 100
        
        z = random.uniform(min_z, max_z)
        
        # 随机航向角
        psi = random.uniform(0, 2 * math.pi)
        
        return State3D(x, y, z, psi)
    
    def _nearest_vertex(self, state: State3D) -> int:
        """找到最近的顶点"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, vertex in enumerate(self.vertices):
            dist = self._distance(vertex, state)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_state: State3D, to_state: State3D) -> Optional[State3D]:
        """考虑约束的扩展函数"""
        # 计算方向和距离
        dx = to_state.x - from_state.x
        dy = to_state.y - from_state.y
        dz = to_state.z - from_state.z
        
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        total_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 限制扩展步长
        if total_dist > self.constraints.step_size:
            ratio = self.constraints.step_size / total_dist
            new_x = from_state.x + ratio * dx
            new_y = from_state.y + ratio * dy
            new_z = from_state.z + ratio * dz
        else:
            new_x = to_state.x
            new_y = to_state.y
            new_z = to_state.z
        
        # 计算新的航向角
        if horizontal_dist > 0:
            new_psi = math.atan2(dy, dx)
        else:
            new_psi = from_state.psi
        
        # 检查航向角变化约束（转弯半径）
        psi_diff = abs(new_psi - from_state.psi)
        if psi_diff > math.pi:
            psi_diff = 2 * math.pi - psi_diff
        
        max_psi_change = self.constraints.step_size / self.constraints.min_turn_radius
        if psi_diff > max_psi_change:
            # 限制航向角变化
            if (new_psi - from_state.psi + 2*math.pi) % (2*math.pi) < math.pi:
                new_psi = from_state.psi + max_psi_change
            else:
                new_psi = from_state.psi - max_psi_change
            
            new_psi = new_psi % (2 * math.pi)
        
        # 检查航迹角约束
        if horizontal_dist > 0:
            new_gamma = math.atan2(new_z - from_state.z, horizontal_dist)
        else:
            new_gamma = 0
        
        new_gamma = max(self.constraints.max_dive_angle_rad, 
                       min(self.constraints.max_climb_angle_rad, new_gamma))
        
        # 根据航迹角调整高度
        new_z = from_state.z + horizontal_dist * math.tan(new_gamma)
        
        # 检查高度限制
        terrain_height = self.terrain.get_terrain_height(new_x, new_y)
        min_z = max(self.constraints.min_altitude, terrain_height + 30)
        max_z = self.constraints.max_altitude
        
        new_z = max(min_z, min(max_z, new_z))
        
        return State3D(new_x, new_y, new_z, new_psi, new_gamma)
    
    def _is_path_collision_free(self, state1: State3D, state2: State3D) -> bool:
        """检查路径是否无碰撞"""
        # 离散化路径检查
        dist = self._distance(state1, state2)
        num_checks = int(dist / 5) + 1
        
        for i in range(num_checks + 1):
            t = i / num_checks if num_checks > 0 else 0
            
            check_state = State3D(
                state1.x + t * (state2.x - state1.x),
                state1.y + t * (state2.y - state1.y),
                state1.z + t * (state2.z - state1.z)
            )
            
            if self.terrain.is_collision(check_state):
                return False
        
        return True
    
    def _extract_path(self) -> List[State3D]:
        """提取路径"""
        # 找到最接近目标的节点
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
    
    def plan(self) -> List[State3D]:
        """执行路径规划"""
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # 随机采样
            x_rand = self._random_sample()
            
            # 找最近顶点
            nearest_idx = self._nearest_vertex(x_rand)
            x_nearest = self.vertices[nearest_idx]
            
            # 扩展
            x_new = self._steer(x_nearest, x_rand)
            if x_new is None:
                continue
            
            # 碰撞检测
            if not self._is_path_collision_free(x_nearest, x_new):
                continue
            
            # 添加新顶点
            new_idx = len(self.vertices)
            self.vertices.append(x_new)
            self.parent[new_idx] = nearest_idx
            self.edges.append((nearest_idx, new_idx))
            
            # 检查是否到达目标
            if self._distance(x_new, self.goal) < self.goal_tolerance:
                self.stats['path_found'] = True
                break
        
        # 计算统计信息
        self.stats['planning_time'] = time.time() - start_time
        self.stats['nodes_generated'] = len(self.vertices)
        
        path = self._extract_path()
        if path:
            path_length = 0
            for i in range(len(path) - 1):
                path_length += self._distance(path[i], path[i+1])
            self.stats['path_length'] = path_length
        
        return path
    
    def visualize_3d(self, path: List[State3D] = None):
        """3D可视化"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地形
        ax.plot_surface(self.terrain.X, self.terrain.Y, self.terrain.terrain_height,
                       alpha=0.3, cmap='terrain')
        
        # 绘制障碍物
        for obs in self.terrain.obstacles:
            if obs['type'] == 'cylinder':
                center = obs['center']
                radius = obs['radius']
                height = obs['height']
                
                # 创建圆柱体
                theta = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(0, height, 10)
                theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
                
                x_cyl = center[0] + radius * np.cos(theta_mesh)
                y_cyl = center[1] + radius * np.sin(theta_mesh)
                
                ax.plot_surface(x_cyl, y_cyl, z_mesh, alpha=0.6, color='red')
        
        # 绘制搜索树
        for edge in self.edges:
            p1 = self.vertices[edge[0]]
            p2 = self.vertices[edge[1]]
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # 绘制顶点
        for vertex in self.vertices:
            ax.scatter(vertex.x, vertex.y, vertex.z, c='blue', s=10, alpha=0.6)
        
        # 绘制起点和终点
        ax.scatter(self.start.x, self.start.y, self.start.z, 
                  c='green', s=100, marker='o', label='起点')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, 
                  c='red', s=100, marker='*', label='目标')
        
        # 绘制路径
        if path:
            path_x = [state.x for state in path]
            path_y = [state.y for state in path]
            path_z = [state.z for state in path]
            ax.plot(path_x, path_y, path_z, 'orange', linewidth=3, label='规划路径')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('高度 (m)')
        ax.set_title('3D地形环境下的RRT路径规划')
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_stats(self):
        """打印统计信息"""
        print("\n=== 3D RRT规划统计 ===")
        print(f"规划时间: {self.stats['planning_time']:.3f} 秒")
        print(f"生成节点数: {self.stats['nodes_generated']}")
        print(f"路径找到: {'是' if self.stats['path_found'] else '否'}")
        if self.stats['path_found']:
            print(f"路径长度: {self.stats['path_length']:.2f} 米")

def main():
    """主函数"""
    print("=== 3D地形环境RRT路径规划演示 ===")
    
    # 创建地形
    terrain = TerrainMap((0, 600), (0, 400))
    
    # 创建约束
    constraints = UAVConstraints3D()
    
    # 创建规划器
    planner = TerrainRRT3D(terrain, constraints)
    
    # 设置起点和终点
    start = State3D(50, 50, 100, 0)  # x, y, z, 航向角
    goal = State3D(550, 350, 150, math.pi/4)
    
    planner.set_start_goal(start, goal)
    
    print("开始路径规划...")
    print(f"起点: ({start.x}, {start.y}, {start.z})")
    print(f"终点: ({goal.x}, {goal.y}, {goal.z})")
    
    # 执行规划
    path = planner.plan()
    
    # 显示结果
    planner.print_stats()
    
    # 可视化
    print("\n正在生成3D可视化...")
    planner.visualize_3d(path)

if __name__ == "__main__":
    main() 