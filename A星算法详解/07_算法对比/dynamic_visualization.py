#!/usr/bin/env python3
"""
路径搜索算法动态可视化演示
实时显示A*、Dijkstra、贪心最佳优先算法的搜索过程
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import math
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Node:
    """节点类"""
    x: int
    y: int
    g: float = float('inf')
    h: float = 0.0
    f: float = float('inf')
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class DynamicPathfinding:
    """动态路径搜索演示类"""
    
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 8方向移动
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 移动代价
        self.move_costs = [
            math.sqrt(2), 1, math.sqrt(2),
            1,               1,
            math.sqrt(2), 1, math.sqrt(2)
        ]
        
        # 颜色配置
        self.colors = {
            'obstacle': '#2C3E50',     # 障碍物 - 深蓝灰
            'empty': '#ECF0F1',        # 空地 - 浅灰
            'start': '#27AE60',        # 起点 - 绿色
            'goal': '#E74C3C',         # 终点 - 红色
            'open': '#F39C12',         # 开放列表 - 橙色
            'closed': '#3498DB',       # 关闭列表 - 蓝色
            'current': '#9B59B6',      # 当前节点 - 紫色
            'path': '#E67E22',         # 最终路径 - 深橙色
        }
        
        # 搜索统计
        self.search_data = {}
    
    def is_valid(self, x: int, y: int) -> bool:
        """检查坐标是否有效"""
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x][y] == 0)
    
    def get_neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        """获取邻居节点"""
        neighbors = []
        for i, (dx, dy) in enumerate(self.directions):
            new_x, new_y = node.x + dx, node.y + dy
            if self.is_valid(new_x, new_y):
                neighbor = Node(new_x, new_y)
                cost = self.move_costs[i]
                neighbors.append((neighbor, cost))
        return neighbors
    
    def heuristic(self, node: Node, goal: Node, method: str = 'euclidean') -> float:
        """计算启发函数值"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        
        if method == 'euclidean':
            return math.sqrt(dx * dx + dy * dy)
        elif method == 'manhattan':
            return dx + dy
        elif method == 'diagonal':
            return max(dx, dy)
        else:
            return math.sqrt(dx * dx + dy * dy)
    
    def reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """重构路径"""
        path = []
        current = goal_node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]
    
    def dijkstra_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Dijkstra算法逐步执行"""
        start_node = Node(start[0], start[1], g=0, f=0)
        goal_node = Node(goal[0], goal[1])
        
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        steps = []
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            if current_pos in closed_set:
                continue
            
            # 记录当前步骤状态
            step_data = {
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy(),
                'g_values': {pos: node.g for pos, node in nodes_dict.items()},
                'f_values': {pos: node.f for pos, node in nodes_dict.items()}
            }
            steps.append(step_data)
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            if current_pos == goal:
                path = self.reconstruct_path(current)
                # 添加最终路径步骤
                final_step = step_data.copy()
                final_step['path'] = path
                final_step['completed'] = True
                steps.append(final_step)
                break
            
            # 扩展邻居
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g = current.g + move_cost
                
                if neighbor_pos in nodes_dict:
                    existing_node = nodes_dict[neighbor_pos]
                    if tentative_g < existing_node.g:
                        existing_node.g = tentative_g
                        existing_node.f = tentative_g
                        existing_node.parent = current
                else:
                    neighbor.g = tentative_g
                    neighbor.f = tentative_g
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'Dijkstra',
            'steps': steps,
            'nodes_explored': nodes_explored,
            'description': 'Dijkstra算法：f(n) = g(n)，保证最优解'
        }
    
    def greedy_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """贪心最佳优先算法逐步执行"""
        start_node = Node(start[0], start[1], g=0)
        goal_node = Node(goal[0], goal[1])
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.h  # 贪心算法：f = h
        
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        steps = []
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            if current_pos in closed_set:
                continue
            
            # 记录当前步骤状态
            step_data = {
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy(),
                'h_values': {pos: node.h for pos, node in nodes_dict.items()},
                'f_values': {pos: node.f for pos, node in nodes_dict.items()}
            }
            steps.append(step_data)
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            if current_pos == goal:
                path = self.reconstruct_path(current)
                final_step = step_data.copy()
                final_step['path'] = path
                final_step['completed'] = True
                steps.append(final_step)
                break
            
            # 扩展邻居
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                if neighbor_pos not in nodes_dict:
                    neighbor.g = current.g + move_cost
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    neighbor.f = neighbor.h  # 贪心算法：只考虑启发值
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'Greedy Best-First',
            'steps': steps,
            'nodes_explored': nodes_explored,
            'description': '贪心最佳优先：f(n) = h(n)，快速但可能非最优'
        }
    
    def astar_step_by_step(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """A*算法逐步执行"""
        start_node = Node(start[0], start[1], g=0)
        goal_node = Node(goal[0], goal[1])
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        steps = []
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            if current_pos in closed_set:
                continue
            
            # 记录当前步骤状态
            step_data = {
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy(),
                'g_values': {pos: node.g for pos, node in nodes_dict.items()},
                'h_values': {pos: node.h for pos, node in nodes_dict.items()},
                'f_values': {pos: node.f for pos, node in nodes_dict.items()}
            }
            steps.append(step_data)
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            if current_pos == goal:
                path = self.reconstruct_path(current)
                final_step = step_data.copy()
                final_step['path'] = path
                final_step['completed'] = True
                steps.append(final_step)
                break
            
            # 扩展邻居
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g = current.g + move_cost
                
                if neighbor_pos in nodes_dict:
                    existing_node = nodes_dict[neighbor_pos]
                    if tentative_g < existing_node.g:
                        existing_node.g = tentative_g
                        existing_node.f = existing_node.g + existing_node.h
                        existing_node.parent = current
                else:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'A*',
            'steps': steps,
            'nodes_explored': nodes_explored,
            'description': 'A*算法：f(n) = g(n) + h(n)，平衡最优性和效率'
        }

class DynamicVisualizer:
    """动态可视化控制器"""
    
    def __init__(self, grid: List[List[int]]):
        self.pathfinder = DynamicPathfinding(grid)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 动画参数
        self.current_algorithm = 0
        self.current_step = 0
        self.algorithms_data = []
        self.speed = 500  # 动画间隔（毫秒）
        
    def prepare_algorithms(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """准备三个算法的数据"""
        print("正在计算算法步骤...")
        
        # 计算三个算法的步骤
        dijkstra_data = self.pathfinder.dijkstra_step_by_step(start, goal)
        greedy_data = self.pathfinder.greedy_step_by_step(start, goal)
        astar_data = self.pathfinder.astar_step_by_step(start, goal)
        
        self.algorithms_data = [dijkstra_data, greedy_data, astar_data]
        self.start = start
        self.goal = goal
        
        print("算法计算完成！")
        print(f"Dijkstra: {len(dijkstra_data['steps'])} 步")
        print(f"贪心最佳优先: {len(greedy_data['steps'])} 步")
        print(f"A*: {len(astar_data['steps'])} 步")
    
    def create_dynamic_demo(self):
        """创建动态演示"""
        if not self.algorithms_data:
            print("请先调用 prepare_algorithms 方法")
            return
        
        # 创建图形窗口
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('路径搜索算法动态对比演示', fontsize=16, fontweight='bold')
        
        # 初始化子图
        for i, ax in enumerate(self.axes):
            algorithm_data = self.algorithms_data[i]
            ax.set_title(f"{algorithm_data['algorithm']}\n{algorithm_data['description']}", 
                        fontsize=12)
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-0.5, self.rows - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # 隐藏坐标轴
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 创建动画
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, frames=self.get_max_steps(),
            interval=self.speed, repeat=True, blit=False
        )
        
        # 添加控制说明
        self.fig.text(0.5, 0.02, 
                     '🟩起点 🟥终点 🟦已探索 🟨开放列表 🟪当前节点 🟧最终路径',
                     ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def get_max_steps(self) -> int:
        """获取最大步数（用于动画帧数）"""
        max_steps = max(len(data['steps']) for data in self.algorithms_data)
        return max_steps + 10  # 额外帧用于显示最终结果
    
    def animate(self, frame):
        """动画更新函数"""
        # 清除所有子图
        for ax in self.axes:
            ax.clear()
        
        # 更新每个算法的可视化
        for i, (ax, algorithm_data) in enumerate(zip(self.axes, self.algorithms_data)):
            self.update_algorithm_plot(ax, algorithm_data, frame)
    
    def update_algorithm_plot(self, ax, algorithm_data, frame):
        """更新单个算法的可视化"""
        steps = algorithm_data['steps']
        algorithm_name = algorithm_data['algorithm']
        description = algorithm_data['description']
        
        # 设置标题和坐标轴
        ax.set_title(f"{algorithm_name}\n{description}", fontsize=10)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 绘制基础网格
        self.draw_base_grid(ax)
        
        # 确定当前步骤
        current_step = min(frame, len(steps) - 1)
        
        if current_step >= 0 and current_step < len(steps):
            step_data = steps[current_step]
            
            # 绘制关闭列表（已探索的节点）
            for pos in step_data['closed_set']:
                rect = Rectangle((pos[1] - 0.4, pos[0] - 0.4), 0.8, 0.8,
                               facecolor=self.pathfinder.colors['closed'],
                               edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
            
            # 绘制开放列表（待探索的节点）
            for pos in step_data['open_set']:
                rect = Rectangle((pos[1] - 0.4, pos[0] - 0.4), 0.8, 0.8,
                               facecolor=self.pathfinder.colors['open'],
                               edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
            
            # 绘制当前节点
            if 'current' in step_data:
                pos = step_data['current']
                rect = Rectangle((pos[1] - 0.4, pos[0] - 0.4), 0.8, 0.8,
                               facecolor=self.pathfinder.colors['current'],
                               edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            
            # 绘制最终路径（如果存在）
            if 'path' in step_data:
                for pos in step_data['path']:
                    if pos != self.start and pos != self.goal:
                        rect = Rectangle((pos[1] - 0.3, pos[0] - 0.3), 0.6, 0.6,
                                       facecolor=self.pathfinder.colors['path'],
                                       edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
        
        # 重新绘制起点和终点（确保在最上层）
        self.draw_start_goal(ax)
        
        # 添加统计信息
        if current_step >= 0 and current_step < len(steps):
            step_data = steps[current_step]
            stats_text = f"步骤: {current_step + 1}/{len(steps)}\n"
            stats_text += f"已探索: {len(step_data['closed_set'])}\n"
            stats_text += f"待探索: {len(step_data['open_set'])}"
            
            if 'path' in step_data:
                stats_text += f"\n路径长度: {len(step_data['path'])}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def draw_base_grid(self, ax):
        """绘制基础网格"""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 1:  # 障碍物
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=self.pathfinder.colors['obstacle'],
                                   edgecolor='black', linewidth=0.5)
                else:  # 空地
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=self.pathfinder.colors['empty'],
                                   edgecolor='gray', linewidth=0.2)
                ax.add_patch(rect)
    
    def draw_start_goal(self, ax):
        """绘制起点和终点"""
        # 起点
        start_rect = Rectangle((self.start[1] - 0.4, self.start[0] - 0.4), 0.8, 0.8,
                             facecolor=self.pathfinder.colors['start'],
                             edgecolor='black', linewidth=2)
        ax.add_patch(start_rect)
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        
        # 终点
        goal_rect = Rectangle((self.goal[1] - 0.4, self.goal[0] - 0.4), 0.8, 0.8,
                            facecolor=self.pathfinder.colors['goal'],
                            edgecolor='black', linewidth=2)
        ax.add_patch(goal_rect)
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

def create_demo_grid() -> List[List[int]]:
    """创建演示网格"""
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    return grid

def main():
    """主函数"""
    print("=== 路径搜索算法动态可视化演示 ===")
    
    # 创建测试网格
    grid = create_demo_grid()
    
    # 设置起点和终点
    start = (0, 0)
    goal = (8, 11)
    
    print(f"网格大小: {len(grid)}x{len(grid[0])}")
    print(f"起点: {start}")
    print(f"终点: {goal}")
    
    # 创建可视化器
    visualizer = DynamicVisualizer(grid)
    
    # 准备算法数据
    visualizer.prepare_algorithms(start, goal)
    
    # 显示动态演示
    print("\n正在启动动态演示...")
    print("将显示三个算法的逐步搜索过程：")
    print("- 左图：Dijkstra算法")
    print("- 中图：贪心最佳优先算法") 
    print("- 右图：A*算法")
    print("\n颜色说明：")
    print("🟩 起点  🟥 终点  🟦 已探索  🟨 开放列表  🟪 当前节点  🟧 最终路径")
    
    visualizer.create_dynamic_demo()

if __name__ == "__main__":
    main() 