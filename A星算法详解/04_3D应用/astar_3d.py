"""
三维A*算法实现
Author: AI Assistant
Description: 支持高度约束和DEM数据的三维路径规划
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

@dataclass
class Node3D:
    """三维节点类"""
    x: int
    y: int
    z: float = 0.0  # 高度值
    g: float = 0.0  # 从起点的实际代价
    h: float = 0.0  # 启发函数值
    f: float = field(init=False)  # 总评估代价
    parent: Optional['Node3D'] = None
    
    def __post_init__(self):
        self.f = self.g + self.h
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class AStar3D:
    """三维A*算法实现"""
    
    def __init__(self, grid: np.ndarray, height_map: Optional[np.ndarray] = None):
        """
        初始化三维A*算法
        
        Args:
            grid: 二维占用网格 (0: 可通行, 1: 障碍物)
            height_map: 高度图 (DEM数据)，如果为None则使用平面
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        
        # 高度图处理
        if height_map is not None:
            self.height_map = height_map
        else:
            self.height_map = np.zeros_like(grid, dtype=float)
        
        # 8方向移动（二维）
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 移动代价权重
        self.horizontal_cost = 1.0
        self.diagonal_cost = math.sqrt(2)
        self.height_penalty = 2.0  # 高度变化惩罚系数
        self.max_slope = 0.3  # 最大可通行坡度 (高度变化/水平距离)
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效"""
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x, y] == 0)
    
    def get_height(self, x: int, y: int) -> float:
        """获取指定位置的高度"""
        if self.is_valid_position(x, y):
            return self.height_map[x, y]
        return float('inf')
    
    def calculate_move_cost(self, current: Node3D, neighbor_x: int, neighbor_y: int) -> float:
        """计算移动代价，考虑高度变化"""
        # 基础水平移动代价
        dx = abs(neighbor_x - current.x)
        dy = abs(neighbor_y - current.y)
        
        if dx == 1 and dy == 1:
            base_cost = self.diagonal_cost
            horizontal_distance = self.diagonal_cost
        else:
            base_cost = self.horizontal_cost
            horizontal_distance = self.horizontal_cost
        
        # 高度变化代价
        current_height = current.z
        neighbor_height = self.get_height(neighbor_x, neighbor_y)
        height_diff = abs(neighbor_height - current_height)
        
        # 检查坡度限制
        if height_diff / horizontal_distance > self.max_slope:
            return float('inf')  # 坡度太陡，不可通行
        
        # 计算高度惩罚
        height_cost = height_diff * self.height_penalty
        
        return base_cost + height_cost
    
    def heuristic_3d(self, node: Node3D, goal: Node3D, method: str = 'euclidean_3d') -> float:
        """
        三维启发函数
        
        Args:
            node: 当前节点
            goal: 目标节点
            method: 启发函数类型
        """
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        dz = abs(node.z - goal.z)
        
        if method == 'euclidean_3d':
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        elif method == 'euclidean_2d':
            return math.sqrt(dx*dx + dy*dy)
        elif method == 'manhattan_3d':
            return dx + dy + dz
        elif method == 'manhattan_2d':
            return dx + dy
        else:
            return math.sqrt(dx*dx + dy*dy)
    
    def get_neighbors_3d(self, node: Node3D) -> List[Tuple[Node3D, float]]:
        """获取三维节点的邻居及移动代价"""
        neighbors = []
        
        for dx, dy in self.directions:
            new_x, new_y = node.x + dx, node.y + dy
            
            if self.is_valid_position(new_x, new_y):
                new_z = self.get_height(new_x, new_y)
                move_cost = self.calculate_move_cost(node, new_x, new_y)
                
                if move_cost != float('inf'):
                    neighbor = Node3D(new_x, new_y, new_z)
                    neighbors.append((neighbor, move_cost))
        
        return neighbors
    
    def find_path_3d(self, start: Tuple[int, int], goal: Tuple[int, int], 
                     heuristic_method: str = 'euclidean_3d') -> Optional[List[Tuple[int, int, float]]]:
        """
        三维A*路径搜索
        
        Args:
            start: 起始位置 (x, y)
            goal: 目标位置 (x, y)
            heuristic_method: 启发函数类型
        
        Returns:
            路径列表 [(x, y, z), ...] 或 None
        """
        start_x, start_y = start
        goal_x, goal_y = goal
        
        if not self.is_valid_position(start_x, start_y):
            raise ValueError(f"Invalid start position: {start}")
        if not self.is_valid_position(goal_x, goal_y):
            raise ValueError(f"Invalid goal position: {goal}")
        
        # 创建起始和目标节点
        start_node = Node3D(start_x, start_y, self.get_height(start_x, start_y))
        goal_node = Node3D(goal_x, goal_y, self.get_height(goal_x, goal_y))
        
        # 初始化搜索结构
        open_list = []
        closed_set: Set[Node3D] = set()
        open_dict = {}
        
        # 设置起始节点
        start_node.h = self.heuristic_3d(start_node, goal_node, heuristic_method)
        start_node.f = start_node.g + start_node.h
        heapq.heappush(open_list, start_node)
        open_dict[(start_x, start_y)] = start_node
        
        nodes_explored = 0
        
        while open_list:
            current_node = heapq.heappop(open_list)
            del open_dict[(current_node.x, current_node.y)]
            nodes_explored += 1
            
            # 检查是否到达目标
            if current_node == goal_node:
                path = self.reconstruct_path_3d(current_node)
                print(f"3D路径搜索完成！探索了 {nodes_explored} 个节点")
                print(f"路径长度: {len(path)} 步")
                return path
            
            closed_set.add(current_node)
            
            # 检查所有邻居
            for neighbor, move_cost in self.get_neighbors_3d(current_node):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_node.g + move_cost
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in open_dict:
                    existing_neighbor = open_dict[neighbor_pos]
                    if tentative_g < existing_neighbor.g:
                        existing_neighbor.g = tentative_g
                        existing_neighbor.f = existing_neighbor.g + existing_neighbor.h
                        existing_neighbor.parent = current_node
                        existing_neighbor.z = neighbor.z  # 更新高度
                else:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic_3d(neighbor, goal_node, heuristic_method)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node
                    
                    heapq.heappush(open_list, neighbor)
                    open_dict[neighbor_pos] = neighbor
        
        print(f"未找到3D路径！探索了 {nodes_explored} 个节点")
        return None
    
    def reconstruct_path_3d(self, goal_node: Node3D) -> List[Tuple[int, int, float]]:
        """重构三维路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y, current.z))
            current = current.parent
        
        return path[::-1]
    
    def get_path_cost_3d(self, path: List[Tuple[int, int, float]]) -> float:
        """计算三维路径总代价"""
        if not path or len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            x1, y1, z1 = path[i]
            x2, y2, z2 = path[i + 1]
            
            # 水平距离
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 1 and dy == 1:
                horizontal_dist = self.diagonal_cost
            else:
                horizontal_dist = self.horizontal_cost
            
            # 高度差
            height_diff = abs(z2 - z1)
            height_cost = height_diff * self.height_penalty
            
            total_cost += horizontal_dist + height_cost
        
        return total_cost

class TerrainGenerator:
    """地形生成器"""
    
    @staticmethod
    def create_hill_terrain(rows: int, cols: int, num_hills: int = 3, 
                           max_height: float = 10.0, noise_level: float = 0.5) -> np.ndarray:
        """创建丘陵地形"""
        terrain = np.zeros((rows, cols))
        
        # 生成随机山丘
        for _ in range(num_hills):
            center_x = np.random.randint(0, rows)
            center_y = np.random.randint(0, cols)
            height = np.random.uniform(max_height * 0.5, max_height)
            radius = np.random.uniform(min(rows, cols) * 0.1, min(rows, cols) * 0.3)
            
            for i in range(rows):
                for j in range(cols):
                    dist = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        # 高斯分布的山丘
                        hill_height = height * math.exp(-(dist**2) / (2 * (radius/3)**2))
                        terrain[i, j] += hill_height
        
        # 添加噪声
        noise = np.random.normal(0, noise_level, (rows, cols))
        terrain += noise
        terrain = np.maximum(terrain, 0)  # 确保非负
        
        return terrain
    
    @staticmethod
    def create_valley_terrain(rows: int, cols: int) -> np.ndarray:
        """创建山谷地形"""
        terrain = np.zeros((rows, cols))
        
        # 创建从边缘到中心的高度递减
        center_x, center_y = rows // 2, cols // 2
        
        for i in range(rows):
            for j in range(cols):
                # 距离边缘的最小距离
                edge_dist = min(i, j, rows - 1 - i, cols - 1 - j)
                terrain[i, j] = edge_dist * 2.0
        
        return terrain
    
    @staticmethod
    def create_canyon_terrain(rows: int, cols: int) -> np.ndarray:
        """创建峡谷地形"""
        terrain = np.ones((rows, cols)) * 10.0
        
        # 创建峡谷通道
        canyon_center = cols // 2
        canyon_width = cols // 6
        
        for i in range(rows):
            for j in range(cols):
                dist_from_center = abs(j - canyon_center)
                if dist_from_center < canyon_width:
                    # 峡谷内部，高度较低
                    terrain[i, j] = 1.0 + dist_from_center * 0.5
        
        return terrain

class Terrain3DVisualizer:
    """三维地形可视化器"""
    
    def __init__(self, grid: np.ndarray, height_map: np.ndarray):
        self.grid = grid
        self.height_map = height_map
        self.astar_3d = AStar3D(grid, height_map)
    
    def visualize_terrain_and_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                                  path: Optional[List[Tuple[int, int, float]]] = None):
        """可视化地形和路径"""
        fig = plt.figure(figsize=(15, 10))
        
        # 3D地形视图
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_3d_terrain(ax1, path, start, goal)
        
        # 2D俯视图
        ax2 = fig.add_subplot(122)
        self.plot_2d_overview(ax2, path, start, goal)
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_terrain(self, ax, path, start, goal):
        """绘制3D地形"""
        rows, cols = self.height_map.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        
        # 绘制地形表面
        surf = ax.plot_surface(X, Y, self.height_map, cmap='terrain', alpha=0.7)
        
        # 标记障碍物
        for i in range(rows):
            for j in range(cols):
                if self.grid[i, j] == 1:
                    ax.scatter([j], [i], [self.height_map[i, j] + 1], 
                             c='red', s=100, marker='s')
        
        # 绘制路径
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            path_z = [p[2] + 0.5 for p in path]  # 稍微抬高路径线
            ax.plot(path_x, path_y, path_z, 'yellow', linewidth=4, marker='o', markersize=4)
        
        # 标记起点和终点
        start_z = self.height_map[start[0], start[1]] + 1
        goal_z = self.height_map[goal[0], goal[1]] + 1
        ax.scatter([start[1]], [start[0]], [start_z], c='green', s=200, marker='^')
        ax.scatter([goal[1]], [goal[0]], [goal_z], c='red', s=200, marker='v')
        
        ax.set_title('三维地形与路径')
        ax.set_xlabel('Y轴')
        ax.set_ylabel('X轴')
        ax.set_zlabel('高度')
    
    def plot_2d_overview(self, ax, path, start, goal):
        """绘制2D俯视图"""
        # 显示高度图
        im = ax.imshow(self.height_map, cmap='terrain')
        plt.colorbar(im, ax=ax, label='高度')
        
        # 显示障碍物
        obstacle_mask = self.grid == 1
        ax.imshow(np.where(obstacle_mask, 1, np.nan), cmap='Reds', alpha=0.8)
        
        # 绘制路径
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'yellow', linewidth=3, marker='o', markersize=3)
        
        # 标记起点和终点
        ax.plot(start[1], start[0], 'go', markersize=15, label='起点')
        ax.plot(goal[1], goal[0], 'ro', markersize=15, label='终点')
        
        ax.set_title('俯视图')
        ax.set_xlabel('Y轴')
        ax.set_ylabel('X轴')
        ax.legend()

def create_test_environment():
    """创建测试环境"""
    # 创建网格
    rows, cols = 20, 20
    grid = np.zeros((rows, cols))
    
    # 添加障碍物
    grid[5:8, 5:8] = 1    # 方形障碍物
    grid[12:15, 10:13] = 1 # 另一个障碍物
    grid[8:12, 15:18] = 1  # 第三个障碍物
    
    # 生成地形
    terrain_gen = TerrainGenerator()
    height_map = terrain_gen.create_hill_terrain(rows, cols, num_hills=4, max_height=8.0)
    
    return grid, height_map

def main():
    """主演示函数"""
    print("=== 三维A*算法演示 ===")
    
    # 创建测试环境
    grid, height_map = create_test_environment()
    
    # 设置起点和终点
    start = (1, 1)
    goal = (18, 18)
    
    print(f"网格大小: {grid.shape}")
    print(f"起点: {start}, 高度: {height_map[start]:.2f}")
    print(f"终点: {goal}, 高度: {height_map[goal]:.2f}")
    
    # 创建3D A*算法实例
    astar_3d = AStar3D(grid, height_map)
    
    # 测试不同的启发函数
    heuristic_methods = ['euclidean_3d', 'euclidean_2d', 'manhattan_2d']
    
    results = {}
    for method in heuristic_methods:
        print(f"\n--- 使用启发函数: {method} ---")
        path = astar_3d.find_path_3d(start, goal, method)
        
        if path:
            cost = astar_3d.get_path_cost_3d(path)
            results[method] = {'path': path, 'cost': cost}
            print(f"路径代价: {cost:.2f}")
            
            # 计算高度变化统计
            heights = [p[2] for p in path]
            height_change = max(heights) - min(heights)
            print(f"高度变化: {height_change:.2f}")
        else:
            results[method] = None
    
    # 可视化最佳结果
    best_method = 'euclidean_3d'
    if results[best_method]:
        print(f"\n可视化 {best_method} 方法的结果...")
        visualizer = Terrain3DVisualizer(grid, height_map)
        visualizer.visualize_terrain_and_path(start, goal, results[best_method]['path'])
    
    # 打印结果比较
    print("\n=== 结果比较 ===")
    for method, result in results.items():
        if result:
            print(f"{method}: 路径长度={len(result['path'])}, 代价={result['cost']:.2f}")
        else:
            print(f"{method}: 未找到路径")

if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main() 