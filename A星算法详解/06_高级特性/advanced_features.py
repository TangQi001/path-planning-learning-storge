"""
A*算法高级特性实现
包含路径平滑、动态权重、双向搜索等高级功能
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from typing import List, Tuple, Optional

class AdvancedAStar:
    """高级A*算法实现"""
    
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """路径平滑处理"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 寻找最远的可直达点
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j]):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """检查两点间是否有直接视线（无障碍）"""
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham直线算法检查路径上是否有障碍
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        error = dx - dy
        
        for _ in range(dx + dy):
            if not (0 <= x < self.rows and 0 <= y < self.cols):
                return False
            if self.grid[x, y] == 1:
                return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        return True
    
    def dynamic_weight_astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
                           initial_weight: float = 2.0) -> Optional[List[Tuple[int, int]]]:
        """动态权重A*算法"""
        open_list = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        
        # 动态权重会随着搜索进行而调整
        current_weight = initial_weight
        
        start_h = self._heuristic(start, goal)
        f_score = {start: g_score[start] + current_weight * start_h}
        heapq.heappush(open_list, (f_score[start], start))
        
        nodes_expanded = 0
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            nodes_expanded += 1
            
            # 动态调整权重（随着搜索深度增加而减小）
            current_weight = max(1.0, initial_weight * (1 - nodes_expanded / 1000))
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid(neighbor) or neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self._heuristic(neighbor, goal)
                    f_score[neighbor] = tentative_g + current_weight * h_score
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return None
    
    def bidirectional_astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """双向A*搜索"""
        # 前向搜索
        open_forward = []
        closed_forward = set()
        came_from_forward = {}
        g_forward = {start: 0}
        f_forward = {start: self._heuristic(start, goal)}
        heapq.heappush(open_forward, (f_forward[start], start))
        
        # 后向搜索
        open_backward = []
        closed_backward = set()
        came_from_backward = {}
        g_backward = {goal: 0}
        f_backward = {goal: self._heuristic(goal, start)}
        heapq.heappush(open_backward, (f_backward[goal], goal))
        
        # 相遇点
        meeting_point = None
        best_cost = float('inf')
        
        while open_forward and open_backward:
            # 前向搜索步骤
            if open_forward:
                current_f, current = heapq.heappop(open_forward)
                if current not in closed_forward:
                    closed_forward.add(current)
                    
                    # 检查是否与后向搜索相遇
                    if current in closed_backward:
                        total_cost = g_forward[current] + g_backward[current]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_point = current
                    
                    # 扩展前向搜索
                    for dx, dy in self.directions:
                        neighbor = (current[0] + dx, current[1] + dy)
                        if self._is_valid(neighbor) and neighbor not in closed_forward:
                            tentative_g = g_forward[current] + self._distance(current, neighbor)
                            if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                                came_from_forward[neighbor] = current
                                g_forward[neighbor] = tentative_g
                                f_forward[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                                heapq.heappush(open_forward, (f_forward[neighbor], neighbor))
            
            # 后向搜索步骤
            if open_backward:
                current_f, current = heapq.heappop(open_backward)
                if current not in closed_backward:
                    closed_backward.add(current)
                    
                    # 检查是否与前向搜索相遇
                    if current in closed_forward:
                        total_cost = g_forward[current] + g_backward[current]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_point = current
                    
                    # 扩展后向搜索
                    for dx, dy in self.directions:
                        neighbor = (current[0] + dx, current[1] + dy)
                        if self._is_valid(neighbor) and neighbor not in closed_backward:
                            tentative_g = g_backward[current] + self._distance(current, neighbor)
                            if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                                came_from_backward[neighbor] = current
                                g_backward[neighbor] = tentative_g
                                f_backward[neighbor] = tentative_g + self._heuristic(neighbor, start)
                                heapq.heappush(open_backward, (f_backward[neighbor], neighbor))
        
        # 重构路径
        if meeting_point:
            # 前半段路径
            forward_path = self._reconstruct_path(came_from_forward, meeting_point)
            # 后半段路径（需要反转）
            backward_path = self._reconstruct_path(came_from_backward, meeting_point)
            backward_path.reverse()
            
            # 合并路径
            full_path = forward_path + backward_path[1:]  # 避免重复中间点
            return full_path
        
        return None
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] == 0
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点间距离"""
        dx, dy = abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)
        return 1.0
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发函数"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重构路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class PathVisualizer:
    """路径可视化器"""
    
    def __init__(self, grid):
        self.grid = grid
    
    def compare_paths(self, paths_dict: dict, start: Tuple[int, int], goal: Tuple[int, int]):
        """比较不同算法的路径"""
        fig, axes = plt.subplots(1, len(paths_dict), figsize=(5*len(paths_dict), 5))
        
        if len(paths_dict) == 1:
            axes = [axes]
        
        for i, (method_name, path) in enumerate(paths_dict.items()):
            ax = axes[i]
            
            # 显示网格
            ax.imshow(self.grid, cmap='binary')
            
            # 绘制路径
            if path:
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                ax.plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=4)
                
                # 标记起点和终点
                ax.plot(start[1], start[0], 'go', markersize=10, label='起点')
                ax.plot(goal[1], goal[0], 'ro', markersize=10, label='终点')
                
                ax.set_title(f'{method_name}\n路径长度: {len(path)}')
            else:
                ax.set_title(f'{method_name}\n无路径')
            
            ax.set_xlabel('Y轴')
            ax.set_ylabel('X轴')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_complex_test_grid():
    """创建复杂测试网格"""
    grid = np.zeros((20, 20))
    
    # 添加各种障碍物
    grid[5:8, 5:15] = 1    # 长条障碍
    grid[10:15, 2:5] = 1   # 方形障碍
    grid[2:10, 18:20] = 1  # 边缘障碍
    grid[15:18, 8:12] = 1  # 另一个障碍
    
    return grid.tolist()

def main():
    """主演示函数"""
    print("=== A*算法高级特性演示 ===")
    
    # 创建测试环境
    grid = create_complex_test_grid()
    start = (1, 1)
    goal = (18, 18)
    
    # 创建高级A*算法实例
    advanced_astar = AdvancedAStar(grid)
    
    # 测试不同算法
    print("测试不同的A*算法变种...")
    
    paths = {}
    
    # 1. 动态权重A*
    print("1. 动态权重A*...")
    path1 = advanced_astar.dynamic_weight_astar(start, goal)
    if path1:
        smoothed_path1 = advanced_astar.smooth_path(path1)
        paths['动态权重A*'] = path1
        paths['动态权重A*(平滑)'] = smoothed_path1
        print(f"   原始路径长度: {len(path1)}")
        print(f"   平滑后长度: {len(smoothed_path1)}")
    
    # 2. 双向A*
    print("2. 双向A*...")
    path2 = advanced_astar.bidirectional_astar(start, goal)
    if path2:
        paths['双向A*'] = path2
        print(f"   路径长度: {len(path2)}")
    
    # 可视化比较
    if paths:
        print("\n生成路径比较图...")
        visualizer = PathVisualizer(grid)
        visualizer.compare_paths(paths, start, goal)
    
    # 性能比较
    print("\n=== 性能比较 ===")
    import time
    
    # 测试多次以获得平均性能
    num_tests = 10
    times = {'动态权重': [], '双向': []}
    
    for _ in range(num_tests):
        # 动态权重A*
        start_time = time.time()
        advanced_astar.dynamic_weight_astar(start, goal)
        times['动态权重'].append(time.time() - start_time)
        
        # 双向A*
        start_time = time.time()
        advanced_astar.bidirectional_astar(start, goal)
        times['双向'].append(time.time() - start_time)
    
    for method, time_list in times.items():
        avg_time = sum(time_list) / len(time_list)
        print(f"{method}A*平均用时: {avg_time:.4f}s")

if __name__ == "__main__":
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main() 