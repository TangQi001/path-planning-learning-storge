"""
Voronoi图法路径规划 - 核心实现
============================

本模块实现基于Voronoi图的路径规划算法核心功能：
1. Voronoi图构造
2. 有效边提取
3. 路径图构建
4. 最短路径搜索

作者：AI教程生成器
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
import networkx as nx
from collections import namedtuple
import math
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据结构定义
Point = namedtuple('Point', ['x', 'y'])
Obstacle = namedtuple('Obstacle', ['center', 'radius'])

class VoronoiPathPlanner:
    """基于Voronoi图的路径规划器"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], 
                 safety_margin: float = 0.5):
        """
        初始化路径规划器
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) 环境边界
            safety_margin: 安全边距
        """
        self.bounds = bounds
        self.safety_margin = safety_margin
        self.obstacles = []
        self.voronoi = None
        self.path_graph = None
        self.valid_edges = []
        
        # 数值精度常量
        self.EPSILON = 1e-9
        
    def add_obstacle(self, center: Point, radius: float):
        """添加圆形障碍物"""
        obstacle = Obstacle(center, radius + self.safety_margin)
        self.obstacles.append(obstacle)
        
    def add_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """批量添加障碍物
        
        Args:
            obstacles: [(x, y, radius), ...] 障碍物列表
        """
        for x, y, radius in obstacles:
            self.add_obstacle(Point(x, y), radius)
    
    def _generate_boundary_points(self, margin: float = 2.0) -> List[Point]:
        """生成边界点确保Voronoi图有界"""
        x_min, y_min, x_max, y_max = self.bounds
        
        # 扩展边界
        x_min -= margin
        y_min -= margin
        x_max += margin
        y_max += margin
        
        boundary_points = [
            Point(x_min, y_min), Point(x_max, y_min),
            Point(x_max, y_max), Point(x_min, y_max),
            Point((x_min + x_max) / 2, y_min),
            Point((x_min + x_max) / 2, y_max),
            Point(x_min, (y_min + y_max) / 2),
            Point(x_max, (y_min + y_max) / 2)
        ]
        
        return boundary_points
    
    def construct_voronoi(self):
        """构造Voronoi图"""
        if not self.obstacles:
            raise ValueError("至少需要添加一个障碍物")
            
        # 提取障碍物中心点作为种子点
        seed_points = [obs.center for obs in self.obstacles]
        
        # 添加边界点
        boundary_points = self._generate_boundary_points()
        all_points = seed_points + boundary_points
        
        # 转换为numpy数组
        points_array = np.array([(p.x, p.y) for p in all_points])
        
        # 构造Voronoi图
        self.voronoi = Voronoi(points_array)
        
        # 提取有效边
        self._extract_valid_edges()
        
        # 构建路径图
        self._build_path_graph()
        
        print(f"Voronoi图构造完成: {len(self.valid_edges)}条有效边")
        
    def _point_in_bounds(self, point: np.ndarray) -> bool:
        """检查点是否在边界内"""
        x, y = point
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _line_intersects_obstacle(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """检查线段是否与任何障碍物相交"""
        for obstacle in self.obstacles:
            center = np.array([obstacle.center.x, obstacle.center.y])
            radius = obstacle.radius
            
            # 计算点到线段的距离
            dist = self._point_to_line_distance(center, p1, p2)
            
            if dist < radius:
                return True
                
        return False
    
    def _point_to_line_distance(self, point: np.ndarray, 
                               line_start: np.ndarray, 
                               line_end: np.ndarray) -> float:
        """计算点到线段的最短距离"""
        # 向量计算
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len < self.EPSILON:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        # 投影点在线段外
        if proj_length < 0:
            return np.linalg.norm(point_vec)
        elif proj_length > line_len:
            return np.linalg.norm(point - line_end)
        else:
            # 投影点在线段上
            proj_point = line_start + proj_length * line_unitvec
            return np.linalg.norm(point - proj_point)
    
    def _extract_valid_edges(self):
        """提取有效的Voronoi边"""
        self.valid_edges = []
        
        for edge_indices in self.voronoi.ridge_vertices:
            # 跳过无界边
            if -1 in edge_indices:
                continue
                
            v1 = self.voronoi.vertices[edge_indices[0]]
            v2 = self.voronoi.vertices[edge_indices[1]]
            
            # 检查边是否有效
            if (self._point_in_bounds(v1) and 
                self._point_in_bounds(v2) and
                not self._line_intersects_obstacle(v1, v2)):
                
                self.valid_edges.append((
                    Point(v1[0], v1[1]),
                    Point(v2[0], v2[1])
                ))
    
    def _build_path_graph(self):
        """构建路径图"""
        self.path_graph = nx.Graph()
        
        for edge in self.valid_edges:
            p1, p2 = edge
            # 使用欧氏距离作为边权重
            weight = euclidean([p1.x, p1.y], [p2.x, p2.y])
            self.path_graph.add_edge(p1, p2, weight=weight)
        
        print(f"路径图构建完成: {self.path_graph.number_of_nodes()}个节点, "
              f"{self.path_graph.number_of_edges()}条边")
    
    def _find_nearest_graph_node(self, target_point: Point) -> Optional[Point]:
        """寻找最近的图节点"""
        if not self.path_graph.nodes():
            return None
            
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.path_graph.nodes():
            dist = euclidean([target_point.x, target_point.y], 
                           [node.x, node.y])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    def _can_connect_directly(self, p1: Point, p2: Point) -> bool:
        """检查两点是否可以直接连接（不与障碍物相交）"""
        start = np.array([p1.x, p1.y])
        end = np.array([p2.x, p2.y])
        return not self._line_intersects_obstacle(start, end)
    
    def plan_path(self, start: Point, goal: Point) -> Tuple[Optional[List[Point]], float]:
        """规划从起点到终点的路径
        
        Args:
            start: 起点
            goal: 终点
            
        Returns:
            (path, distance): 路径点列表和总距离，无路径时返回(None, inf)
        """
        if self.voronoi is None:
            self.construct_voronoi()
        
        # 寻找起点和终点最近的图节点
        start_node = self._find_nearest_graph_node(start)
        goal_node = self._find_nearest_graph_node(goal)
        
        if start_node is None or goal_node is None:
            print("无法找到合适的图节点")
            return None, float('inf')
        
        # 创建临时图，包含起点和终点连接
        temp_graph = self.path_graph.copy()
        
        # 连接起点到图
        if self._can_connect_directly(start, start_node):
            weight = euclidean([start.x, start.y], [start_node.x, start_node.y])
            temp_graph.add_edge(start, start_node, weight=weight)
        else:
            # 寻找多个可连接的节点
            candidates = []
            for node in self.path_graph.nodes():
                if self._can_connect_directly(start, node):
                    dist = euclidean([start.x, start.y], [node.x, node.y])
                    candidates.append((node, dist))
            
            if not candidates:
                print("起点无法连接到Voronoi图")
                return None, float('inf')
            
            # 连接到最近的可达节点
            candidates.sort(key=lambda x: x[1])
            best_node, weight = candidates[0]
            temp_graph.add_edge(start, best_node, weight=weight)
        
        # 连接终点到图
        if self._can_connect_directly(goal, goal_node):
            weight = euclidean([goal.x, goal.y], [goal_node.x, goal_node.y])
            temp_graph.add_edge(goal, goal_node, weight=weight)
        else:
            # 寻找多个可连接的节点
            candidates = []
            for node in self.path_graph.nodes():
                if self._can_connect_directly(goal, node):
                    dist = euclidean([goal.x, goal.y], [node.x, node.y])
                    candidates.append((node, dist))
            
            if not candidates:
                print("终点无法连接到Voronoi图")
                return None, float('inf')
            
            # 连接到最近的可达节点
            candidates.sort(key=lambda x: x[1])
            best_node, weight = candidates[0]
            temp_graph.add_edge(goal, best_node, weight=weight)
        
        # 搜索最短路径
        try:
            path = nx.shortest_path(temp_graph, start, goal, weight='weight')
            distance = nx.shortest_path_length(temp_graph, start, goal, weight='weight')
            
            print(f"路径规划成功: 长度={distance:.2f}, 路径点数={len(path)}")
            return path, distance
            
        except nx.NetworkXNoPath:
            print("无可行路径")
            return None, float('inf')
    
    def visualize(self, path: Optional[List[Point]] = None, 
                  title: str = "Voronoi图路径规划"):
        """可视化Voronoi图和路径"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 绘制边界
        x_min, y_min, x_max, y_max = self.bounds
        boundary = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(boundary)
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle.center.x, obstacle.center.y),
                              obstacle.radius, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # 绘制Voronoi图
        if self.voronoi is not None:
            # 绘制Voronoi顶点
            vertices = self.voronoi.vertices
            valid_vertices = []
            for vertex in vertices:
                if self._point_in_bounds(vertex):
                    valid_vertices.append(vertex)
            
            if valid_vertices:
                valid_vertices = np.array(valid_vertices)
                ax.scatter(valid_vertices[:, 0], valid_vertices[:, 1], 
                          c='blue', s=20, alpha=0.6, label='Voronoi顶点')
            
            # 绘制有效边
            for edge in self.valid_edges:
                p1, p2 = edge
                ax.plot([p1.x, p2.x], [p1.y, p2.y], 
                       'b-', alpha=0.5, linewidth=1, label='Voronoi边' if edge == self.valid_edges[0] else "")
        
        # 绘制路径
        if path is not None and len(path) > 1:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=3, label='规划路径')
            ax.scatter(path_x[0], path_y[0], c='green', s=100, marker='o', label='起点')
            ax.scatter(path_x[-1], path_y[-1], c='red', s=100, marker='s', label='终点')
        
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        plt.tight_layout()
        plt.show()

def demo_basic_planning():
    """基本路径规划演示"""
    print("=== Voronoi图路径规划基本演示 ===")
    
    # 创建规划器
    planner = VoronoiPathPlanner(bounds=(0, 0, 20, 15), safety_margin=0.3)
    
    # 添加障碍物
    obstacles = [
        (5, 5, 1.5),    # (x, y, radius)
        (12, 8, 2.0),
        (8, 12, 1.0),
        (15, 3, 1.8),
        (3, 10, 1.2),
        (18, 10, 1.5)
    ]
    
    planner.add_obstacles(obstacles)
    
    # 构造Voronoi图
    planner.construct_voronoi()
    
    # 规划路径
    start = Point(1, 1)
    goal = Point(19, 14)
    
    path, distance = planner.plan_path(start, goal)
    
    if path:
        print(f"路径规划成功！")
        print(f"路径长度: {distance:.2f}")
        print(f"路径点数: {len(path)}")
        
        # 可视化结果
        planner.visualize(path, "Voronoi图路径规划演示")
    else:
        print("路径规划失败！")
        planner.visualize(title="Voronoi图（无可行路径）")

def demo_comparison():
    """对比直线路径和Voronoi路径"""
    print("\n=== 路径对比演示 ===")
    
    planner = VoronoiPathPlanner(bounds=(0, 0, 15, 10), safety_margin=0.2)
    
    # 添加障碍物 - 创建一个需要绕行的场景
    obstacles = [
        (7, 5, 2.5),    # 中央大障碍物
        (4, 7, 1.0),    # 辅助障碍物
        (10, 3, 1.0),   # 辅助障碍物
    ]
    
    planner.add_obstacles(obstacles)
    planner.construct_voronoi()
    
    start = Point(1, 2)
    goal = Point(14, 8)
    
    # Voronoi路径
    voronoi_path, voronoi_dist = planner.plan_path(start, goal)
    
    # 直线距离
    direct_dist = euclidean([start.x, start.y], [goal.x, goal.y])
    
    print(f"直线距离: {direct_dist:.2f}")
    if voronoi_path:
        print(f"Voronoi路径长度: {voronoi_dist:.2f}")
        print(f"路径增长率: {(voronoi_dist/direct_dist - 1)*100:.1f}%")
    
    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：直线路径
    ax1.set_aspect('equal')
    for obstacle in planner.obstacles:
        circle = plt.Circle((obstacle.center.x, obstacle.center.y),
                          obstacle.radius, color='red', alpha=0.7)
        ax1.add_patch(circle)
    
    ax1.plot([start.x, goal.x], [start.y, goal.y], 'r--', linewidth=2, label='直线路径')
    ax1.scatter([start.x], [start.y], c='green', s=100, marker='o', label='起点')
    ax1.scatter([goal.x], [goal.y], c='red', s=100, marker='s', label='终点')
    ax1.set_xlim(-1, 16)
    ax1.set_ylim(-1, 11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('直线路径（不安全）')
    
    # 右图：Voronoi路径
    ax2.set_aspect('equal')
    for obstacle in planner.obstacles:
        circle = plt.Circle((obstacle.center.x, obstacle.center.y),
                          obstacle.radius, color='red', alpha=0.7)
        ax2.add_patch(circle)
    
    # 绘制Voronoi边
    for edge in planner.valid_edges:
        p1, p2 = edge
        ax2.plot([p1.x, p2.x], [p1.y, p2.y], 'b-', alpha=0.3, linewidth=1)
    
    if voronoi_path:
        path_x = [p.x for p in voronoi_path]
        path_y = [p.y for p in voronoi_path]
        ax2.plot(path_x, path_y, 'g-', linewidth=3, label='Voronoi路径')
    
    ax2.scatter([start.x], [start.y], c='green', s=100, marker='o', label='起点')
    ax2.scatter([goal.x], [goal.y], c='red', s=100, marker='s', label='终点')
    ax2.set_xlim(-1, 16)
    ax2.set_ylim(-1, 11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Voronoi路径（安全）')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行演示
    demo_basic_planning()
    demo_comparison() 