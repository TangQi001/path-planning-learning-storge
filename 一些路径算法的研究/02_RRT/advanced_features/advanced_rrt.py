
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
RRT算法高级特性实现

作者: AICP-7协议实现
功能: 高级RRT变种算法
特点: Informed RRT*、双向RRT、动态环境适应
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import sys
import os

# 添加代码实现目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_代码实现'))
from rrt_basic import Node, RRTBasic
from rrt_star import RRTStar

class InformedRRTStar(RRTStar):
    """Informed RRT* - 使用椭圆采样提高效率"""
    
    def __init__(self, start, goal, obstacle_list, boundary,
                 step_size=1.0, goal_sample_rate=0.1, max_iter=2000,
                 search_radius=3.0, gamma=50.0):
        super().__init__(start, goal, obstacle_list, boundary,
                        step_size, goal_sample_rate, max_iter,
                        search_radius, gamma)
        
        self.best_path_length = float('inf')
        self.ellipse_center = None
        self.ellipse_axes = None
        self.ellipse_rotation = None
        self.use_informed_sampling = False
    
    def plan(self):
        """执行Informed RRT*路径规划"""
        for i in range(self.max_iter):
            # 使用椭圆采样（如果找到初始解）
            if self.use_informed_sampling:
                rand_node = self.ellipse_sample()
            else:
                rand_node = self.sample()
            
            # 标准RRT*步骤
            nearest_node = self.get_nearest_node(rand_node)
            new_node = self.steer(nearest_node, rand_node)
            
            if self.check_collision(nearest_node, new_node):
                continue
            
            # 找到邻居节点
            near_nodes = self.find_near_nodes(new_node)
            
            # 选择最优父节点
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + self.distance(nearest_node, new_node)
            
            for near_node in near_nodes:
                if (not self.check_collision(near_node, new_node) and
                    near_node.cost + self.distance(near_node, new_node) < min_cost):
                    min_cost_node = near_node
                    min_cost = near_node.cost + self.distance(near_node, new_node)
            
            new_node.parent = min_cost_node
            new_node.cost = min_cost
            self.node_list.append(new_node)
            
            # 重连
            self.rewire(new_node, near_nodes)
            
            # 检查是否到达目标并更新椭圆
            if self.distance(new_node, self.goal) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                if not self.check_collision(new_node, final_node):
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + self.distance(new_node, final_node)
                    
                    # 更新最优路径长度和椭圆参数
                    if final_node.cost < self.best_path_length:
                        self.best_path_length = final_node.cost
                        self.update_ellipse_parameters()
                        self.use_informed_sampling = True
                        
                        # 如果是第一次找到解，返回路径
                        if not hasattr(self, 'found_solution'):
                            self.found_solution = True
                            return self.generate_final_course(final_node)
        
        # 返回最优解
        return self.get_best_path_to_goal()
    
    def update_ellipse_parameters(self):
        """更新椭圆采样参数"""
        # 椭圆中心是起点和终点的中点
        self.ellipse_center = np.array([
            (self.start.x + self.goal.x) / 2,
            (self.start.y + self.goal.y) / 2
        ])
        
        # 焦点距离
        c = self.distance(self.start, self.goal) / 2
        
        # 长轴半径 = 最优路径长度 / 2
        a = self.best_path_length / 2
        
        # 短轴半径
        if a > c:
            b = math.sqrt(a**2 - c**2)
        else:
            b = 0.1  # 防止退化
        
        self.ellipse_axes = np.array([a, b])
        
        # 椭圆旋转角度
        self.ellipse_rotation = math.atan2(
            self.goal.y - self.start.y,
            self.goal.x - self.start.x
        )
    
    def ellipse_sample(self):
        """在椭圆内采样"""
        # 在单位圆内随机采样
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                break
        
        # 缩放到椭圆
        ellipse_point = np.array([
            x * self.ellipse_axes[0],
            y * self.ellipse_axes[1]
        ])
        
        # 旋转椭圆
        rotation_matrix = np.array([
            [math.cos(self.ellipse_rotation), -math.sin(self.ellipse_rotation)],
            [math.sin(self.ellipse_rotation), math.cos(self.ellipse_rotation)]
        ])
        
        rotated_point = rotation_matrix @ ellipse_point
        
        # 平移到椭圆中心
        final_point = self.ellipse_center + rotated_point
        
        return Node(final_point[0], final_point[1])

def demo_informed_rrt_star():
    """Informed RRT*演示"""
    print("🎯 Informed RRT*算法演示")
    
    start = (2, 2)
    goal = (18, 18)
    obstacles = [
        (6, 6, 1.5), (10, 4, 1.2), (14, 8, 1.8),
        (8, 12, 1.0), (15, 15, 1.5)
    ]
    boundary = (0, 20, 0, 20)
    
    # 创建Informed RRT*规划器
    informed_rrt = InformedRRTStar(
        start=start, goal=goal, obstacle_list=obstacles, boundary=boundary,
        max_iter=2000, search_radius=3.0
    )
    
    print("📊 正在搜索优化路径...")
    path = informed_rrt.plan()
    
    if path:
        print(f"✅ 找到路径! 长度: {len(path)}个点")
        
        path_length = sum(math.sqrt((path[i][0] - path[i-1][0])**2 + 
                                   (path[i][1] - path[i-1][1])**2)
                         for i in range(1, len(path)))
        print(f"📏 路径总长度: {path_length:.2f}")
        print(f"🎯 使用椭圆采样: {informed_rrt.use_informed_sampling}")
        
        informed_rrt.draw(path=path, show_tree=True)
        return path
    else:
        print("❌ 未找到可行路径")
        return None

if __name__ == "__main__":
    demo_informed_rrt_star() 