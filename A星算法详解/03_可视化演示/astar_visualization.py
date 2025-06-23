"""
A*算法可视化演示
使用matplotlib实现A*算法搜索过程的可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# 简化的A*算法实现用于可视化
class SimpleAStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
    def heuristic(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def find_path_visual(self, start, goal):
        """带可视化的路径搜索"""
        open_list = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        # 存储搜索过程
        search_steps = []
        
        while open_list:
            # 找到f值最小的节点
            current = min(open_list, key=lambda x: f_score.get(x, float('inf')))
            
            # 记录当前搜索状态
            search_steps.append({
                'current': current,
                'open_list': open_list.copy(),
                'closed_set': closed_set.copy(),
                'path': self.reconstruct_path(came_from, current) if current in came_from else [current]
            })
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path, search_steps
            
            open_list.remove(current)
            closed_set.add(current)
            
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] == 0):
                    
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in open_list:
                        open_list.append(neighbor)
                    elif tentative_g >= g_score.get(neighbor, float('inf')):
                        continue
                    
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
        
        return None, search_steps
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class AStarVisualizer:
    def __init__(self, grid, start, goal):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.astar = SimpleAStar(grid)
        
    def visualize_search(self):
        """可视化搜索过程"""
        path, search_steps = self.astar.find_path_visual(self.start, self.goal)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 显示最终结果
        self.show_result(ax1, path, search_steps[-1] if search_steps else None)
        
        # 显示搜索统计
        self.show_statistics(ax2, search_steps, path)
        
        plt.tight_layout()
        plt.show()
        
        return path, search_steps
    
    def show_result(self, ax, path, final_step):
        """显示最终搜索结果"""
        # 创建可视化网格
        vis_grid = np.ones((*self.grid.shape, 3))
        
        # 设置基础颜色
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 1:
                    vis_grid[i, j] = [0, 0, 0]  # 障碍物 - 黑色
                else:
                    vis_grid[i, j] = [1, 1, 1]  # 空地 - 白色
        
        # 显示搜索过程
        if final_step:
            # 关闭列表 - 浅红色
            for pos in final_step['closed_set']:
                if self.grid[pos] == 0:
                    vis_grid[pos] = [1.0, 0.7, 0.7]
            
            # 开放列表 - 浅蓝色
            for pos in final_step['open_list']:
                if self.grid[pos] == 0:
                    vis_grid[pos] = [0.7, 0.9, 1.0]
        
        # 最终路径 - 黄色
        if path:
            for pos in path:
                if pos != self.start and pos != self.goal:
                    vis_grid[pos] = [1, 1, 0]
        
        # 起点和终点
        vis_grid[self.start] = [0, 1, 0]  # 绿色
        vis_grid[self.goal] = [1, 0, 0]   # 红色
        
        ax.imshow(vis_grid)
        ax.set_title("A*算法搜索结果", fontsize=14, fontweight='bold')
        ax.set_xlabel("列")
        ax.set_ylabel("行")
        
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    def show_statistics(self, ax, search_steps, path):
        """显示搜索统计信息"""
        ax.axis('off')
        
        info_text = "A*算法搜索统计\n\n"
        info_text += "颜色说明:\n"
        info_text += "🟢 起点\n"
        info_text += "🔴 终点\n"
        info_text += "⬜ 空地\n"
        info_text += "⬛ 障碍物\n"
        info_text += "🔷 开放列表\n"
        info_text += "🔸 关闭列表\n"
        info_text += "🟡 最终路径\n\n"
        
        if search_steps:
            info_text += f"搜索步数: {len(search_steps)}\n"
            final_step = search_steps[-1]
            info_text += f"探索节点数: {len(final_step['closed_set'])}\n"
            info_text += f"待探索节点数: {len(final_step['open_list'])}\n"
        
        if path:
            info_text += f"路径长度: {len(path)} 步\n"
            info_text += f"起点: {self.start}\n"
            info_text += f"终点: {self.goal}\n"
            info_text += "\n✅ 成功找到路径!"
        else:
            info_text += "\n❌ 未找到路径"
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')

def create_test_grids():
    """创建多个测试网格"""
    
    # 简单网格
    simple_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    # 复杂网格
    complex_grid = [
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
    
    # 迷宫网格
    maze_grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    ]
    
    return {
        'simple': simple_grid,
        'complex': complex_grid,
        'maze': maze_grid
    }

def main():
    """主演示函数"""
    print("=== A*算法可视化演示 ===")
    
    # 获取测试网格
    grids = create_test_grids()
    
    print("可用的测试场景:")
    for i, (name, _) in enumerate(grids.items(), 1):
        print(f"{i}. {name}")
    
    # 用户选择
    try:
        choice = int(input("请选择测试场景 (1-3): ")) - 1
        grid_names = list(grids.keys())
        if 0 <= choice < len(grid_names):
            selected_name = grid_names[choice]
            selected_grid = grids[selected_name]
        else:
            print("无效选择，使用默认复杂场景")
            selected_name = 'complex'
            selected_grid = grids['complex']
    except ValueError:
        print("无效输入，使用默认复杂场景")
        selected_name = 'complex'
        selected_grid = grids['complex']
    
    # 设置起点和终点
    if selected_name == 'simple':
        start, goal = (0, 0), (4, 4)
    else:
        start, goal = (0, 0), (9, 9)
    
    print(f"\n场景: {selected_name}")
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"网格大小: {len(selected_grid)}x{len(selected_grid[0])}")
    
    # 创建可视化器
    visualizer = AStarVisualizer(selected_grid, start, goal)
    
    # 执行可视化
    print("\n开始搜索...")
    path, search_steps = visualizer.visualize_search()
    
    # 输出结果
    if path:
        print(f"✅ 找到路径! 长度: {len(path)} 步")
        print(f"搜索步数: {len(search_steps)}")
        print(f"路径: {' -> '.join(map(str, path))}")
    else:
        print("❌ 未找到路径")

if __name__ == "__main__":
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main() 