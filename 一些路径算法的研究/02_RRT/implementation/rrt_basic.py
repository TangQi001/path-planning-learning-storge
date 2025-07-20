"""
RRT (Rapidly-exploring Random Tree) 基础实现

作者: AICP-7协议实现
功能: 2D环境下的路径规划
特点: 快速随机采样、避障、路径生成
"""

import random
import math

# 可选的可视化依赖
HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
    
    # 中文字体配置
    try:
        from font_config import setup_chinese_font
        setup_chinese_font()
    except ImportError:
        # 如果无法导入字体配置，使用基本配置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("注意: matplotlib未安装，可视化功能将不可用")
    print("要启用可视化功能，请运行: pip install matplotlib")

# 数值计算库（可选）
HAS_NUMPY = False
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("注意: numpy未安装，将使用Python内置数学函数")

class Node:
    """RRT树的节点类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # 从起始点到该节点的代价

    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f})"

# 为了兼容集成测试，提供简化接口
class RRT:
    """RRT算法简化接口"""
    def __init__(self, start, goal, obstacles, map_bounds):
        # 转换为内部格式
        obstacle_list = []
        for obs in obstacles:
            if isinstance(obs, dict) and 'center' in obs and 'radius' in obs:
                obstacle_list.append((obs['center'][0], obs['center'][1], obs['radius']))
            elif len(obs) == 3:
                obstacle_list.append(obs)
        
        boundary = (map_bounds[0], map_bounds[1], map_bounds[2], map_bounds[3])
        
        self.rrt = RRTBasic(start, goal, obstacle_list, boundary)
    
    def plan(self, max_iterations=1000):
        """执行路径规划"""
        self.rrt.max_iter = max_iterations
        return self.rrt.plan()

class RRTBasic:
    """基础RRT算法实现"""
    
    def __init__(self, start, goal, obstacle_list, boundary, 
                 step_size=1.0, goal_sample_rate=0.1, max_iter=1000):
        """
        初始化RRT规划器
        
        Args:
            start: (x, y) 起始点
            goal: (x, y) 目标点  
            obstacle_list: [(x, y, radius)] 圆形障碍物列表
            boundary: (x_min, x_max, y_min, y_max) 边界
            step_size: 扩展步长
            goal_sample_rate: 目标偏向采样概率
            max_iter: 最大迭代次数
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.boundary = boundary
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        # 树结构存储
        self.node_list = [self.start]
        
    def plan(self):
        """
        执行RRT路径规划
        
        Returns:
            path: 路径点列表 [(x, y), ...] 或 None
        """
        for i in range(self.max_iter):
            # 1. 随机采样
            rand_node = self.sample()
            
            # 2. 找到最近邻节点
            nearest_node = self.get_nearest_node(rand_node)
            
            # 3. 扩展新节点
            new_node = self.steer(nearest_node, rand_node)
            
            # 4. 碰撞检测
            if self.check_collision(nearest_node, new_node):
                continue
                
            # 5. 添加到树中
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + self.distance(nearest_node, new_node)
            self.node_list.append(new_node)
            
            # 6. 检查是否到达目标
            if self.distance(new_node, self.goal) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                if not self.check_collision(new_node, final_node):
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + self.distance(new_node, final_node)
                    return self.generate_final_course(final_node)
                    
        return None  # 未找到路径
        
    def sample(self):
        """随机采样策略"""
        if random.random() > self.goal_sample_rate:
            # 在边界内均匀随机采样
            x = random.uniform(self.boundary[0], self.boundary[1])
            y = random.uniform(self.boundary[2], self.boundary[3])
            return Node(x, y)
        else:
            # 目标偏向采样
            return Node(self.goal.x, self.goal.y)
    
    def get_nearest_node(self, target_node):
        """找到树中距离目标最近的节点"""
        distances = [self.distance(node, target_node) for node in self.node_list]
        min_index = distances.index(min(distances))
        return self.node_list[min_index]
    
    def steer(self, from_node, to_node):
        """从from_node向to_node方向扩展step_size距离"""
        dist = self.distance(from_node, to_node)
        
        if dist <= self.step_size:
            return Node(to_node.x, to_node.y)
        
        # 计算扩展方向
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * math.cos(theta)
        new_y = from_node.y + self.step_size * math.sin(theta)
        
        return Node(new_x, new_y)
    
    def check_collision(self, from_node, to_node):
        """检查路径是否与障碍物碰撞"""
        # 边界检查
        if (to_node.x < self.boundary[0] or to_node.x > self.boundary[1] or
            to_node.y < self.boundary[2] or to_node.y > self.boundary[3]):
            return True
            
        # 障碍物碰撞检测
        for (ox, oy, radius) in self.obstacle_list:
            # 点到直线距离公式进行路径碰撞检测
            if self.line_circle_intersection(from_node, to_node, ox, oy, radius):
                return True
                
        return False
    
    def line_circle_intersection(self, node1, node2, cx, cy, radius):
        """检查线段与圆是否相交"""
        # 线段端点
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        
        # 线段长度
        dx = x2 - x1
        dy = y2 - y1
        
        # 参数方程: P(t) = (x1, y1) + t * (dx, dy), t ∈ [0, 1]
        # 点到直线最短距离的参数t
        t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
        
        # 最近点坐标
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # 距离检查
        distance = math.sqrt((closest_x - cx)**2 + (closest_y - cy)**2)
        return distance <= radius
    
    def distance(self, node1, node2):
        """计算两点间欧几里得距离"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def generate_final_course(self, goal_node):
        """从目标节点回溯生成最终路径"""
        path = []
        node = goal_node
        
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
            
        return path[::-1]  # 反转得到从起点到终点的路径
    
    def draw(self, path=None, show_tree=True, save_path=None):
        """可视化RRT树和路径"""
        if not HAS_MATPLOTLIB:
            print("matplotlib未安装，无法显示可视化")
            print("路径规划结果（文本形式）:")
            if path:
                print(f"  - 路径长度: {len(path)}个点")
                print(f"  - 起点: ({path[0][0]:.2f}, {path[0][1]:.2f})")
                print(f"  - 终点: ({path[-1][0]:.2f}, {path[-1][1]:.2f})")
                print(f"  - 路径总长度: {calculate_path_length(path):.2f}")
            print(f"  - RRT树节点数: {len(self.node_list)}")
            return
            
        plt.figure(figsize=(12, 10))
        
        # 绘制边界
        boundary_rect = patches.Rectangle(
            (self.boundary[0], self.boundary[2]), 
            self.boundary[1] - self.boundary[0], 
            self.boundary[3] - self.boundary[2],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        plt.gca().add_patch(boundary_rect)
        
        # 绘制障碍物
        for (ox, oy, radius) in self.obstacle_list:
            circle = patches.Circle((ox, oy), radius, 
                                  facecolor='red', alpha=0.6, edgecolor='darkred')
            plt.gca().add_patch(circle)
        
        # 绘制RRT树
        if show_tree:
            for node in self.node_list:
                if node.parent:
                    plt.plot([node.x, node.parent.x], 
                           [node.y, node.parent.y], 'g-', alpha=0.3, linewidth=1)
                    
        # 绘制节点
        node_x = [node.x for node in self.node_list]
        node_y = [node.y for node in self.node_list]
        plt.scatter(node_x, node_y, c='lightgreen', s=20, alpha=0.6, zorder=3)
        
        # 绘制起点和终点
        plt.scatter(self.start.x, self.start.y, c='blue', s=100, 
                   marker='o', label='起点', zorder=5)
        plt.scatter(self.goal.x, self.goal.y, c='red', s=100, 
                   marker='*', label='目标', zorder=5)
        
        # 绘制找到的路径
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'b-', linewidth=3, label='RRT路径')
            
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('RRT算法路径规划演示')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def demo_basic_rrt():
    """RRT基础演示"""
    print("🚀 开始RRT算法演示...")
    
    # 设置问题参数
    start = (2, 2)
    goal = (18, 18)
    
    # 定义障碍物 (x, y, radius)
    obstacles = [
        (5, 5, 1.5),
        (8, 8, 1.0),
        (12, 3, 1.5),
        (15, 12, 2.0),
        (7, 15, 1.2),
        (13, 8, 1.8)
    ]
    
    # 定义边界
    boundary = (0, 20, 0, 20)
    
    # 创建RRT规划器
    rrt = RRTBasic(
        start=start,
        goal=goal, 
        obstacle_list=obstacles,
        boundary=boundary,
        step_size=1.0,
        goal_sample_rate=0.1,
        max_iter=3000
    )
    
    # 执行路径规划
    print("📊 正在搜索路径...")
    path = rrt.plan()
    
    if path:
        print(f"✅ 找到路径! 长度: {len(path)}个点")
        print(f"📏 路径总长度: {calculate_path_length(path):.2f}")
        print(f"🌳 生成节点数: {len(rrt.node_list)}")
        
        # 可视化结果
        rrt.draw(path=path, show_tree=True)
        
        return path
    else:
        print("❌ 未找到可行路径")
        rrt.draw(show_tree=True)
        return None

def calculate_path_length(path):
    """计算路径总长度"""
    if len(path) < 2:
        return 0
    
    length = 0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += math.sqrt(dx*dx + dy*dy)
    
    return length

def comparison_demo():
    """不同参数对比演示"""
    print("\n🔄 参数对比演示...")
    
    start = (1, 1)
    goal = (19, 19)
    obstacles = [(8, 8, 2), (15, 5, 1.5), (5, 15, 1.8)]
    boundary = (0, 20, 0, 20)
    
    configs = [
        {"step_size": 0.5, "goal_rate": 0.05, "name": "小步长+低目标偏向"},
        {"step_size": 1.5, "goal_rate": 0.15, "name": "大步长+高目标偏向"},
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, config in enumerate(configs):
        rrt = RRTBasic(
            start=start, goal=goal, obstacle_list=obstacles, boundary=boundary,
            step_size=config["step_size"], goal_sample_rate=config["goal_rate"]
        )
        
        path = rrt.plan()
        
        plt.sca(axes[i])
        rrt.draw(path=path, show_tree=True)
        plt.title(config["name"])
        
        if path:
            length = calculate_path_length(path)
            print(f"{config['name']}: 路径长度 {length:.2f}, 节点数 {len(rrt.node_list)}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== RRT算法演示程序 ===")
    
    if HAS_MATPLOTLIB:
        print("检测到matplotlib，启用完整可视化功能")
        print("请选择演示模式：")
        print("1. 基础演示（带可视化）")
        print("2. 参数对比演示")
        print("3. 仅核心算法测试（无可视化）")
        
        while True:
            try:
                choice = input("\n请选择模式 (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                else:
                    print("请输入1、2或3")
            except (ValueError, KeyboardInterrupt):
                print("输入无效，请重试")
        
        if choice == '1':
            demo_basic_rrt()
        elif choice == '2':
            comparison_demo()
        else:
            # 核心算法测试
            test_core_algorithm()
    else:
        print("matplotlib未安装，运行核心算法测试")
        test_core_algorithm()

def test_core_algorithm():
    """测试核心算法（无可视化）"""
    print("\n=== RRT核心算法测试 ===")
    
    # 设置问题参数
    start = (2, 2)
    goal = (18, 18)
    
    # 定义障碍物 (x, y, radius)
    obstacles = [
        (5, 5, 1.5),
        (8, 8, 1.0),
        (12, 3, 1.5),
        (15, 12, 2.0),
        (7, 15, 1.2),
        (13, 8, 1.8)
    ]
    
    # 定义边界
    boundary = (0, 20, 0, 20)
    
    print(f"起点: {start}")
    print(f"目标: {goal}")
    print(f"障碍物数量: {len(obstacles)}")
    print(f"搜索边界: {boundary}")
    
    # 创建RRT规划器
    rrt = RRTBasic(
        start=start,
        goal=goal, 
        obstacle_list=obstacles,
        boundary=boundary,
        step_size=1.0,
        goal_sample_rate=0.1,
        max_iter=3000
    )
    
    # 执行路径规划
    print("\n正在搜索路径...")
    path = rrt.plan()
    
    if path:
        print(f"✅ 找到路径!")
        print(f"路径点数: {len(path)}")
        print(f"路径总长度: {calculate_path_length(path):.2f}")
        print(f"生成节点数: {len(rrt.node_list)}")
        print(f"起点: ({path[0][0]:.2f}, {path[0][1]:.2f})")
        print(f"终点: ({path[-1][0]:.2f}, {path[-1][1]:.2f})")
        
        # 显示路径的前几个和后几个点
        print("\n路径预览:")
        for i, point in enumerate(path[:3]):
            print(f"  点{i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        if len(path) > 6:
            print("  ...")
            for i, point in enumerate(path[-3:], len(path)-2):
                print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        return path
    else:
        print("❌ 未找到可行路径")
        print(f"生成节点数: {len(rrt.node_list)}")
        return None 