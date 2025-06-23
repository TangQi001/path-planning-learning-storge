#!/usr/bin/env python3
"""
路径搜索算法对比演示
包含A*、Dijkstra、贪心最佳优先算法的实现和对比
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import time
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Node:
    """节点类，表示网格中的一个位置"""
    x: int
    y: int
    g: float = float('inf')  # 从起点的实际代价
    h: float = 0.0           # 启发函数值
    f: float = float('inf')  # 总评估代价
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.f < other.f
    
    def __eq__(self, other):
        """节点相等比较"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """使节点可哈希，用于集合操作"""
        return hash((self.x, self.y))

class PathfindingAlgorithms:
    """路径搜索算法集合"""
    
    def __init__(self, grid: List[List[int]]):
        """
        初始化算法
        
        Args:
            grid: 网格地图，0表示可通行，1表示障碍物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # 8方向移动
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 移动代价（直线移动代价为1，对角移动代价为√2）
        self.move_costs = [
            math.sqrt(2), 1, math.sqrt(2),
            1,               1,
            math.sqrt(2), 1, math.sqrt(2)
        ]
    
    def is_valid(self, x: int, y: int) -> bool:
        """检查坐标是否有效且可通行"""
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x][y] == 0)
    
    def get_neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        """获取节点的邻居和移动代价"""
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.directions):
            new_x, new_y = node.x + dx, node.y + dy
            
            if self.is_valid(new_x, new_y):
                neighbor = Node(new_x, new_y)
                cost = self.move_costs[i]
                neighbors.append((neighbor, cost))
        
        return neighbors
    
    def heuristic(self, node: Node, goal: Node, method: str = 'euclidean') -> float:
        """
        计算启发函数值
        
        Args:
            node: 当前节点
            goal: 目标节点
            method: 启发函数类型
        """
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
    
    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict:
        """
        Dijkstra算法实现
        特点：保证最优解，但不使用启发信息，搜索范围大
        """
        start_node = Node(start[0], start[1], g=0, f=0)
        goal_node = Node(goal[0], goal[1])
        
        # 优先队列和已访问集合
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        # 记录搜索过程
        search_steps = []
        nodes_explored = 0
        
        while open_set:
            # 选择g值最小的节点
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            # 记录搜索步骤
            search_steps.append({
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy()
            })
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            # 检查是否到达目标
            if current_pos == goal:
                path = self.reconstruct_path(current)
                return {
                    'algorithm': 'Dijkstra',
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'search_steps': search_steps,
                    'path_cost': current.g,
                    'success': True
                }
            
            # 扩展邻居节点
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g = current.g + move_cost
                
                if neighbor_pos in nodes_dict:
                    existing_node = nodes_dict[neighbor_pos]
                    if tentative_g < existing_node.g:
                        existing_node.g = tentative_g
                        existing_node.f = tentative_g  # Dijkstra中f=g
                        existing_node.parent = current
                else:
                    neighbor.g = tentative_g
                    neighbor.f = tentative_g  # Dijkstra中f=g
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'Dijkstra',
            'path': None,
            'nodes_explored': nodes_explored,
            'search_steps': search_steps,
            'path_cost': float('inf'),
            'success': False
        }
    
    def greedy_best_first(self, start: Tuple[int, int], goal: Tuple[int, int], 
                         heuristic_method: str = 'euclidean') -> Dict:
        """
        贪心最佳优先算法实现
        特点：只使用启发函数h，快速但不保证最优解
        """
        start_node = Node(start[0], start[1], g=0)
        goal_node = Node(goal[0], goal[1])
        
        # 计算启发函数值
        start_node.h = self.heuristic(start_node, goal_node, heuristic_method)
        start_node.f = start_node.h  # 贪心算法中f=h
        
        # 优先队列和已访问集合
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        # 记录搜索过程
        search_steps = []
        nodes_explored = 0
        
        while open_set:
            # 选择h值最小的节点
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            # 记录搜索步骤
            search_steps.append({
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy()
            })
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            # 检查是否到达目标
            if current_pos == goal:
                path = self.reconstruct_path(current)
                return {
                    'algorithm': 'Greedy Best-First',
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'search_steps': search_steps,
                    'path_cost': current.g,
                    'success': True
                }
            
            # 扩展邻居节点
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                if neighbor_pos not in nodes_dict:
                    neighbor.g = current.g + move_cost
                    neighbor.h = self.heuristic(neighbor, goal_node, heuristic_method)
                    neighbor.f = neighbor.h  # 贪心算法中f=h
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'Greedy Best-First',
            'path': None,
            'nodes_explored': nodes_explored,
            'search_steps': search_steps,
            'path_cost': float('inf'),
            'success': False
        }
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], 
               heuristic_method: str = 'euclidean') -> Dict:
        """
        A*算法实现
        特点：结合实际代价g和启发函数h，平衡最优性和效率
        """
        start_node = Node(start[0], start[1], g=0)
        goal_node = Node(goal[0], goal[1])
        
        # 计算启发函数值
        start_node.h = self.heuristic(start_node, goal_node, heuristic_method)
        start_node.f = start_node.g + start_node.h
        
        # 优先队列和已访问集合
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        nodes_dict: Dict[Tuple[int, int], Node] = {(start[0], start[1]): start_node}
        
        # 记录搜索过程
        search_steps = []
        nodes_explored = 0
        
        while open_set:
            # 选择f值最小的节点
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            # 记录搜索步骤
            search_steps.append({
                'current': current_pos,
                'open_set': [(n.x, n.y) for n in open_set],
                'closed_set': closed_set.copy()
            })
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            # 检查是否到达目标
            if current_pos == goal:
                path = self.reconstruct_path(current)
                return {
                    'algorithm': 'A*',
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'search_steps': search_steps,
                    'path_cost': current.g,
                    'success': True
                }
            
            # 扩展邻居节点
            for neighbor, move_cost in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g = current.g + move_cost
                
                if neighbor_pos in nodes_dict:
                    existing_node = nodes_dict[neighbor_pos]
                    if tentative_g < existing_node.g:
                        existing_node.g = tentative_g
                        existing_node.h = self.heuristic(existing_node, goal_node, heuristic_method)
                        existing_node.f = existing_node.g + existing_node.h
                        existing_node.parent = current
                else:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node, heuristic_method)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current
                    nodes_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        return {
            'algorithm': 'A*',
            'path': None,
            'nodes_explored': nodes_explored,
            'search_steps': search_steps,
            'path_cost': float('inf'),
            'success': False
        }

class AlgorithmComparator:
    """算法对比器"""
    
    def __init__(self, grid: List[List[int]]):
        self.algorithms = PathfindingAlgorithms(grid)
        self.grid = grid
        
    def compare_algorithms(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict:
        """
        对比三种算法的性能
        
        Returns:
            包含所有算法结果的字典
        """
        print(f"开始算法对比: {start} → {goal}")
        
        results = {}
        
        # 运行Dijkstra算法
        print("运行Dijkstra算法...")
        start_time = time.time()
        dijkstra_result = self.algorithms.dijkstra(start, goal)
        dijkstra_result['execution_time'] = time.time() - start_time
        results['dijkstra'] = dijkstra_result
        
        # 运行贪心最佳优先算法
        print("运行贪心最佳优先算法...")
        start_time = time.time()
        greedy_result = self.algorithms.greedy_best_first(start, goal)
        greedy_result['execution_time'] = time.time() - start_time
        results['greedy'] = greedy_result
        
        # 运行A*算法
        print("运行A*算法...")
        start_time = time.time()
        astar_result = self.algorithms.a_star(start, goal)
        astar_result['execution_time'] = time.time() - start_time
        results['astar'] = astar_result
        
        return results
    
    def visualize_comparison(self, start: Tuple[int, int], goal: Tuple[int, int], 
                           results: Dict):
        """可视化算法对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('路径搜索算法对比', fontsize=16, fontweight='bold')
        
        # 算法顺序
        algorithms = ['dijkstra', 'greedy', 'astar']
        algorithm_names = ['Dijkstra算法', '贪心最佳优先算法', 'A*算法']
        
        # 为每个算法创建可视化
        for i, (alg_key, alg_name) in enumerate(zip(algorithms, algorithm_names)):
            if i < 3:  # 前三个子图
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                result = results[alg_key]
                self._plot_algorithm_result(ax, result, start, goal, alg_name)
        
        # 第四个子图：性能对比表
        ax = axes[1, 1]
        self._plot_performance_comparison(ax, results)
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细结果
        self._print_comparison_results(results)
    
    def _plot_algorithm_result(self, ax, result: Dict, start: Tuple[int, int], 
                              goal: Tuple[int, int], algorithm_name: str):
        """绘制单个算法的结果"""
        rows, cols = len(self.grid), len(self.grid[0])
        
        # 创建颜色图
        color_map = np.zeros((rows, cols, 3))
        
        # 基础颜色：白色(空地)，黑色(障碍物)
        for i in range(rows):
            for j in range(cols):
                if self.grid[i][j] == 1:
                    color_map[i, j] = [0, 0, 0]  # 黑色障碍物
                else:
                    color_map[i, j] = [1, 1, 1]  # 白色空地
        
        # 标记探索的节点
        if result['search_steps']:
            final_step = result['search_steps'][-1]
            for pos in final_step['closed_set']:
                color_map[pos[0], pos[1]] = [0.8, 0.8, 1.0]  # 浅蓝色已探索
        
        # 标记路径
        if result['path']:
            for pos in result['path']:
                color_map[pos[0], pos[1]] = [1.0, 1.0, 0.0]  # 黄色路径
        
        # 标记起点和终点
        color_map[start[0], start[1]] = [0.0, 1.0, 0.0]  # 绿色起点
        color_map[goal[0], goal[1]] = [1.0, 0.0, 0.0]    # 红色终点
        
        # 显示图像
        ax.imshow(color_map)
        ax.set_title(f'{algorithm_name}\n'
                    f'探索节点: {result["nodes_explored"]}, '
                    f'路径长度: {len(result["path"]) if result["path"] else "无"}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加网格线
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    def _plot_performance_comparison(self, ax, results: Dict):
        """绘制性能对比图表"""
        algorithms = ['dijkstra', 'greedy', 'astar']
        algorithm_names = ['Dijkstra', '贪心BFS', 'A*']
        
        # 提取性能数据
        nodes_explored = [results[alg]['nodes_explored'] for alg in algorithms]
        execution_times = [results[alg]['execution_time'] * 1000 for alg in algorithms]  # 转换为毫秒
        path_costs = [results[alg]['path_cost'] if results[alg]['success'] else 0 
                     for alg in algorithms]
        
        # 创建柱状图
        x = np.arange(len(algorithm_names))
        width = 0.25
        
        ax.bar(x - width, nodes_explored, width, label='探索节点数', alpha=0.8)
        ax.bar(x, execution_times, width, label='执行时间(ms)', alpha=0.8)
        ax.bar(x + width, path_costs, width, label='路径代价', alpha=0.8)
        
        ax.set_xlabel('算法')
        ax.set_ylabel('数值')
        ax.set_title('性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithm_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _print_comparison_results(self, results: Dict):
        """打印详细的对比结果"""
        print("\n" + "="*80)
        print("算法性能对比结果")
        print("="*80)
        
        algorithms = ['dijkstra', 'greedy', 'astar']
        algorithm_names = ['Dijkstra算法', '贪心最佳优先算法', 'A*算法']
        
        for alg_key, alg_name in zip(algorithms, algorithm_names):
            result = results[alg_key]
            print(f"\n【{alg_name}】")
            print(f"  成功找到路径: {'是' if result['success'] else '否'}")
            print(f"  探索节点数: {result['nodes_explored']}")
            print(f"  执行时间: {result['execution_time']*1000:.2f} ms")
            
            if result['success']:
                print(f"  路径长度: {len(result['path'])} 步")
                print(f"  路径代价: {result['path_cost']:.2f}")
            else:
                print("  路径长度: 无")
                print("  路径代价: 无穷大")
        
        # 算法特点总结
        print(f"\n{'='*80}")
        print("算法特点总结:")
        print("【Dijkstra算法】")
        print("  ✓ 保证找到最优路径")
        print("  ✓ 适用于所有非负权重图")
        print("  ✗ 不使用启发信息，搜索范围大")
        print("  ✗ 计算开销较大")
        
        print("\n【贪心最佳优先算法】")
        print("  ✓ 搜索速度快，启发式导向明确")
        print("  ✓ 内存使用相对较少")
        print("  ✗ 不保证找到最优路径")
        print("  ✗ 可能陷入局部最优")
        
        print("\n【A*算法】")
        print("  ✓ 平衡最优性和效率")
        print("  ✓ 在启发函数可接受时保证最优解")
        print("  ✓ 使用启发信息引导搜索")
        print("  ✗ 需要设计好的启发函数")

def create_test_grids() -> Dict[str, List[List[int]]]:
    """创建测试网格"""
    grids = {
        'simple': [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        
        'complex': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        
        'maze': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ]
    }
    
    return grids

def main():
    """主函数：算法对比演示"""
    print("="*60)
    print("  路径搜索算法对比演示")
    print("  A* vs Dijkstra vs 贪心最佳优先")
    print("="*60)
    
    # 演示模式选择
    print("\n请选择演示模式：")
    print("1. 静态对比分析（数据统计+最终结果）")
    print("2. 动态可视化演示（实时搜索过程）")
    
    while True:
        try:
            mode_choice = input("\n请选择模式 (1-2): ").strip()
            if mode_choice in ['1', '2']:
                break
            else:
                print("请输入1或2。")
        except (ValueError, KeyboardInterrupt):
            print("输入无效，请重试。")
    
    if mode_choice == '2':
        # 动态演示模式
        try:
            from dynamic_visualization import DynamicVisualizer, create_demo_grid
            
            print("\n=== 动态可视化演示模式 ===")
            grid = create_demo_grid()
            start = (0, 0)
            goal = (8, 11)
            
            print(f"网格大小: {len(grid)}x{len(grid[0])}")
            print(f"起点: {start}")
            print(f"终点: {goal}")
            
            visualizer = DynamicVisualizer(grid)
            visualizer.prepare_algorithms(start, goal)
            
            print("\n正在启动动态演示...")
            print("将显示三个算法的逐步搜索过程：")
            print("- 左图：Dijkstra算法")
            print("- 中图：贪心最佳优先算法") 
            print("- 右图：A*算法")
            print("\n颜色说明：")
            print("🟩 起点  🟥 终点  🟦 已探索  🟨 开放列表  🟪 当前节点  🟧 最终路径")
            
            visualizer.create_dynamic_demo()
            return
            
        except ImportError:
            print("动态可视化模块未找到，将使用静态模式")
        except Exception as e:
            print(f"启动动态演示失败: {e}")
            print("将使用静态模式")
    
    # 静态对比分析模式
    print("\n=== 静态对比分析模式 ===")
    
    # 获取测试网格
    grids = create_test_grids()
    
    print("\n可用的测试场景:")
    for i, (name, grid) in enumerate(grids.items(), 1):
        size = f"{len(grid)}x{len(grid[0])}"
        print(f"{i}. {name} ({size})")
    
    # 用户选择场景
    while True:
        try:
            choice = input("\n请选择测试场景 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                grid_names = list(grids.keys())
                selected_grid = grids[grid_names[int(choice) - 1]]
                scenario_name = grid_names[int(choice) - 1]
                break
            else:
                print("请输入1-3之间的数字。")
        except (ValueError, KeyboardInterrupt):
            print("输入无效，请重试。")
    
    print(f"\n选择的场景: {scenario_name}")
    
    # 设置起点和终点
    if scenario_name == 'simple':
        start, goal = (0, 0), (4, 4)
    elif scenario_name == 'complex':
        start, goal = (0, 0), (9, 9)
    else:  # maze
        start, goal = (0, 0), (9, 9)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    
    # 创建算法对比器
    comparator = AlgorithmComparator(selected_grid)
    
    # 运行算法对比
    print(f"\n开始运行三种算法...")
    results = comparator.compare_algorithms(start, goal)
    
    # 可视化结果
    print(f"\n生成可视化对比图...")
    comparator.visualize_comparison(start, goal, results)

if __name__ == "__main__":
    main() 