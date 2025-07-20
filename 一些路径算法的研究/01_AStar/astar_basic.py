
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
A*算法基础实现
Author: AI Assistant
Description: 基于二维网格的A*路径搜索算法实现
"""

import heapq
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

@dataclass
class Node:
    """节点类，表示网格中的一个位置"""
    x: int
    y: int
    g: float = 0.0  # 从起点到当前节点的实际代价
    h: float = 0.0  # 启发函数值（到终点的估计代价）
    f: float = field(init=False)  # 总评估代价 f = g + h
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f = self.g + self.h
    
    def __lt__(self, other):
        """定义节点比较规则，用于优先队列排序"""
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

# 为集成测试提供简化接口
class AStarPlanner:
    """A*算法简化接口"""
    def plan(self, grid, start, goal):
        """执行路径规划"""
        astar = AStar(grid)
        return astar.find_path(start, goal)

class AStar:
    """A*算法实现类"""
    
    def __init__(self, grid: List[List[int]]):
        """
        初始化A*算法
        
        Args:
            grid: 二维网格，0表示可通行，1表示障碍物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
        # 8方向移动（包括对角线）
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),  # 上方三个方向
            (0, -1),           (0, 1),   # 左右
            (1, -1),  (1, 0),  (1, 1)    # 下方三个方向
        ]
        
        # 移动代价：直线移动代价为1，对角线移动代价为√2
        self.move_costs = [
            math.sqrt(2), 1, math.sqrt(2),
            1,               1,
            math.sqrt(2), 1, math.sqrt(2)
        ]
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效（在网格内且不是障碍物）"""
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x][y] == 0)
    
    def get_neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        """获取节点的所有有效邻居及移动代价"""
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.directions):
            new_x, new_y = node.x + dx, node.y + dy
            
            if self.is_valid_position(new_x, new_y):
                neighbor = Node(new_x, new_y)
                move_cost = self.move_costs[i]
                neighbors.append((neighbor, move_cost))
        
        return neighbors
    
    def heuristic(self, node: Node, goal: Node, method: str = 'euclidean') -> float:
        """
        计算启发函数值
        
        Args:
            node: 当前节点
            goal: 目标节点
            method: 启发函数类型 ('euclidean', 'manhattan', 'diagonal')
        
        Returns:
            启发函数值
        """
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        
        if method == 'euclidean':
            return math.sqrt(dx*dx + dy*dy)
        elif method == 'manhattan':
            return dx + dy
        elif method == 'diagonal':
            return max(dx, dy)
        else:
            raise ValueError(f"Unknown heuristic method: {method}")
    
    def reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """重构路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # 反转路径，从起点到终点
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  heuristic_method: str = 'euclidean', record_steps: bool = False) -> Optional[List[Tuple[int, int]]]:
        """
        使用A*算法寻找路径
        
        Args:
            start: 起始位置 (x, y)
            goal: 目标位置 (x, y)
            heuristic_method: 启发函数类型
            record_steps: 是否记录搜索步骤（用于可视化）
        
        Returns:
            路径列表，如果无路径则返回None
        """
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # 检查起点和终点是否有效
        if not self.is_valid_position(start_x, start_y):
            raise ValueError(f"Invalid start position: {start}")
        if not self.is_valid_position(goal_x, goal_y):
            raise ValueError(f"Invalid goal position: {goal}")
        
        # 初始化起始节点和目标节点
        start_node = Node(start_x, start_y)
        goal_node = Node(goal_x, goal_y)
        
        # 开放列表（优先队列）和关闭列表（集合）
        open_list = []
        closed_set: Set[Node] = set()
        
        # 用于快速查找开放列表中的节点
        open_dict = {}
        
        # 将起始节点加入开放列表
        start_node.h = self.heuristic(start_node, goal_node, heuristic_method)
        start_node.f = start_node.g + start_node.h
        heapq.heappush(open_list, start_node)
        open_dict[(start_x, start_y)] = start_node
        
        # 搜索统计
        nodes_explored = 0
        
        while open_list:
            # 取出f值最小的节点
            current_node = heapq.heappop(open_list)
            del open_dict[(current_node.x, current_node.y)]
            nodes_explored += 1
            
            # 如果到达目标，重构路径
            if current_node == goal_node:
                path = self.reconstruct_path(current_node)
                print(f"路径搜索完成！探索了 {nodes_explored} 个节点")
                print(f"路径长度: {len(path)} 步")
                return path
            
            # 将当前节点加入关闭列表
            closed_set.add(current_node)
            
            # 检查所有邻居
            for neighbor, move_cost in self.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue
                
                # 计算新的g值
                tentative_g = current_node.g + move_cost
                
                # 检查是否找到更好的路径
                neighbor_pos = (neighbor.x, neighbor.y)
                if neighbor_pos in open_dict:
                    existing_neighbor = open_dict[neighbor_pos]
                    if tentative_g < existing_neighbor.g:
                        # 找到更好的路径，更新节点
                        existing_neighbor.g = tentative_g
                        existing_neighbor.f = existing_neighbor.g + existing_neighbor.h
                        existing_neighbor.parent = current_node
                else:
                    # 新节点，添加到开放列表
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node, heuristic_method)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node
                    
                    heapq.heappush(open_list, neighbor)
                    open_dict[neighbor_pos] = neighbor
        
        print(f"未找到路径！探索了 {nodes_explored} 个节点")
        return None
    
    def get_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """计算路径总代价"""
        if not path or len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 计算移动代价
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 1 and dy == 1:
                total_cost += math.sqrt(2)  # 对角线移动
            else:
                total_cost += 1.0  # 直线移动
        
        return total_cost

def create_test_grid() -> List[List[int]]:
    """创建测试网格"""
    # 0: 可通行, 1: 障碍物
    grid = [
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
    ]
    return grid

def print_grid_with_path(grid: List[List[int]], path: List[Tuple[int, int]]):
    """打印带路径的网格"""
    # 复制网格
    display_grid = [row[:] for row in grid]
    
    # 标记路径
    for i, (x, y) in enumerate(path):
        if i == 0:
            display_grid[x][y] = 'S'  # 起点
        elif i == len(path) - 1:
            display_grid[x][y] = 'G'  # 终点
        else:
            display_grid[x][y] = '*'  # 路径
    
    # 打印网格
    print("\n网格地图 (S:起点, G:终点, *:路径, 1:障碍物, 0:空地):")
    for row in display_grid:
        print(' '.join(str(cell) for cell in row))
    print()

def create_dynamic_demo():
    """创建动态演示"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Rectangle
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("=== A*算法动态可视化演示 ===")
        print("提示：关闭图形窗口以结束演示")
        
        # 创建测试网格
        grid = create_test_grid()
        rows, cols = len(grid), len(grid[0])
        
        # 定义颜色
        colors = {
            'obstacle': '#2C3E50',    # 障碍物
            'empty': '#ECF0F1',       # 空地  
            'start': '#27AE60',       # 起点
            'goal': '#E74C3C',        # 终点
            'path': '#F39C12'         # 路径
        }
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('A*算法路径搜索演示', fontsize=16, fontweight='bold')
        
        # 绘制基础网格
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:  # 障碍物
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=colors['obstacle'],
                                   edgecolor='black', linewidth=0.5)
                else:  # 空地
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=colors['empty'],
                                   edgecolor='gray', linewidth=0.2)
                ax.add_patch(rect)
        
        # 设置起点和终点
        start = (0, 0)
        goal = (9, 9)
        
        # 绘制起点
        start_rect = Rectangle((start[1] - 0.4, start[0] - 0.4), 0.8, 0.8,
                             facecolor=colors['start'], edgecolor='black', linewidth=2)
        ax.add_patch(start_rect)
        ax.text(start[1], start[0], 'S', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        
        # 绘制终点
        goal_rect = Rectangle((goal[1] - 0.4, goal[0] - 0.4), 0.8, 0.8,
                            facecolor=colors['goal'], edgecolor='black', linewidth=2)
        ax.add_patch(goal_rect)
        ax.text(goal[1], goal[0], 'G', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        
        # 运行A*算法
        astar = AStar(grid)
        path = astar.find_path(start, goal)
        
        if path:
            # 绘制路径
            for i, (x, y) in enumerate(path):
                if (x, y) != start and (x, y) != goal:
                    path_rect = Rectangle((y - 0.3, x - 0.3), 0.6, 0.6,
                                        facecolor=colors['path'],
                                        edgecolor='black', linewidth=1)
                    ax.add_patch(path_rect)
                    # 添加步骤编号
                    ax.text(y, x, str(i), ha='center', va='center',
                           fontsize=10, fontweight='bold')
            
            # 添加统计信息
            cost = astar.get_path_cost(path)
            info_text = f"路径找到！\n路径长度: {len(path)} 步\n路径代价: {cost:.2f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            # 未找到路径
            ax.text(0.5, 0.5, '未找到路径！', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加图例
        legend_text = "🟩 起点   🟥 终点   🟧 路径   ⬛ 障碍物"
        fig.text(0.5, 0.02, legend_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib未安装，无法显示可视化演示")
        print("请运行: pip install matplotlib")

if __name__ == "__main__":
    print("=== A*算法演示程序 ===")
    print("请选择演示模式：")
    print("1. 基础测试（文本输出）")
    print("2. 可视化演示（图形界面）")
    
    while True:
        try:
            choice = input("\n请选择模式 (1-2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("请输入1或2")
        except (ValueError, KeyboardInterrupt):
            print("输入无效，请重试")
    
    if choice == '2':
        # 可视化演示
        create_dynamic_demo()
    else:
        # 基础测试
        print("\n=== A*算法基础测试 ===")
        
        # 创建测试网格
        grid = create_test_grid()
        astar = AStar(grid)
        
        # 设置起点和终点
        start = (0, 0)
        goal = (9, 9)
        
        print(f"起点: {start}")
        print(f"终点: {goal}")
        
        # 测试不同的启发函数
        heuristics = ['euclidean', 'manhattan', 'diagonal']
        
        for heuristic in heuristics:
            print(f"\n--- 使用 {heuristic} 启发函数 ---")
            path = astar.find_path(start, goal, heuristic)
            
            if path:
                cost = astar.get_path_cost(path)
                print(f"路径代价: {cost:.2f}")
                print_grid_with_path(grid, path)
            else:
                print("未找到路径！")
    
    print("\n测试完成！") 