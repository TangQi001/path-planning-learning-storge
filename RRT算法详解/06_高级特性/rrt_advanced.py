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
from collections import defaultdict

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

class BiRRT(RRTBasic):
    """双向RRT算法"""
    
    def __init__(self, start, goal, obstacle_list, boundary,
                 step_size=1.0, max_iter=2000):
        super().__init__(start, goal, obstacle_list, boundary,
                        step_size, 0.0, max_iter)  # 关闭目标偏向采样
        
        # 从目标开始的树
        self.goal_tree = [Node(goal[0], goal[1])]
        self.connect_distance = step_size * 2
    
    def plan(self):
        """执行双向RRT路径规划"""
        for i in range(self.max_iter):
            # 随机采样
            rand_node = self.sample()
            
            # 扩展start树
            start_extended = self.extend_tree(self.node_list, rand_node)
            
            if start_extended:
                # 尝试连接到goal树
                goal_connection = self.connect_trees(
                    start_extended, self.goal_tree, reverse=True)
                
                if goal_connection:
                    return self.construct_path(start_extended, goal_connection)
            
            # 扩展goal树
            goal_extended = self.extend_tree(self.goal_tree, rand_node)
            
            if goal_extended:
                # 尝试连接到start树
                start_connection = self.connect_trees(
                    goal_extended, self.node_list, reverse=False)
                
                if start_connection:
                    return self.construct_path(start_connection, goal_extended)
        
        return None
    
    def extend_tree(self, tree, target_node):
        """扩展树结构"""
        nearest_node = self.get_nearest_node_from_tree(tree, target_node)
        new_node = self.steer(nearest_node, target_node)
        
        if not self.check_collision(nearest_node, new_node):
            new_node.parent = nearest_node
            tree.append(new_node)
            return new_node
        
        return None
    
    def connect_trees(self, new_node, target_tree, reverse=False):
        """尝试连接两棵树"""
        nearest_node = self.get_nearest_node_from_tree(target_tree, new_node)
        
        if self.distance(new_node, nearest_node) <= self.connect_distance:
            if not self.check_collision(new_node, nearest_node):
                return nearest_node
        
        return None
    
    def get_nearest_node_from_tree(self, tree, target_node):
        """从指定树中找到最近节点"""
        distances = [self.distance(node, target_node) for node in tree]
        min_index = distances.index(min(distances))
        return tree[min_index]
    
    def construct_path(self, start_connection, goal_connection):
        """构建连接两棵树的路径"""
        # 从start到连接点的路径
        start_path = []
        node = start_connection
        while node is not None:
            start_path.append((node.x, node.y))
            node = node.parent
        start_path.reverse()
        
        # 从goal到连接点的路径
        goal_path = []
        node = goal_connection
        while node is not None:
            goal_path.append((node.x, node.y))
            node = node.parent
        
        # 合并路径
        return start_path + goal_path
    
    def draw(self, path=None, show_tree=True, save_path=None):
        """可视化双向RRT"""
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
        
        # 绘制start树
        if show_tree:
            for node in self.node_list:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y],
                            'g-', alpha=0.5, linewidth=1)
            
            # 绘制goal树  
            for node in self.goal_tree:
                if node.parent:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y],
                            'b-', alpha=0.5, linewidth=1)
        
        # 绘制节点
        start_coords = np.array([(node.x, node.y) for node in self.node_list])
        goal_coords = np.array([(node.x, node.y) for node in self.goal_tree])
        
        if len(start_coords) > 0:
            plt.scatter(start_coords[:, 0], start_coords[:, 1],
                       c='lightgreen', s=20, alpha=0.6, label='Start树')
        
        if len(goal_coords) > 0:
            plt.scatter(goal_coords[:, 0], goal_coords[:, 1],
                       c='lightblue', s=20, alpha=0.6, label='Goal树')
        
        # 绘制起点和终点
        plt.scatter(self.start.x, self.start.y, c='blue', s=100,
                   marker='o', label='起点', zorder=5)
        plt.scatter(self.goal.x, self.goal.y, c='red', s=100,
                   marker='*', label='目标', zorder=5)
        
        # 绘制路径
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'purple', linewidth=4, label='路径')
        
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('双向RRT (Bi-RRT) 算法')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class DynamicRRT(RRTBasic):
    """动态环境RRT - 处理移动障碍物"""
    
    def __init__(self, start, goal, static_obstacles, moving_obstacles,
                 boundary, step_size=1.0, max_iter=2000, time_horizon=10):
        super().__init__(start, goal, static_obstacles, boundary,
                        step_size, 0.1, max_iter)
        
        self.moving_obstacles = moving_obstacles  # [(x, y, r, vx, vy), ...]
        self.time_horizon = time_horizon
        self.current_time = 0
        self.time_step = 0.1
    
    def get_obstacle_position(self, obstacle, time):
        """获取移动障碍物在指定时间的位置"""
        x, y, radius, vx, vy = obstacle
        future_x = x + vx * time
        future_y = y + vy * time
        return (future_x, future_y, radius)
    
    def check_dynamic_collision(self, from_node, to_node, time):
        """检查动态碰撞"""
        # 检查静态障碍物
        if self.check_collision(from_node, to_node):
            return True
        
        # 检查移动障碍物
        for obstacle in self.moving_obstacles:
            future_obstacle = self.get_obstacle_position(obstacle, time)
            
            # 简化的点-圆碰撞检测
            if self.point_circle_collision(to_node, future_obstacle):
                return True
        
        return False
    
    def point_circle_collision(self, node, obstacle):
        """检查点与圆的碰撞"""
        ox, oy, radius = obstacle
        distance = math.sqrt((node.x - ox)**2 + (node.y - oy)**2)
        return distance <= radius
    
    def plan_with_time(self):
        """考虑时间的动态规划"""
        for i in range(self.max_iter):
            # 更新时间
            self.current_time += self.time_step
            
            # 随机采样
            rand_node = self.sample()
            
            # 找到最近节点
            nearest_node = self.get_nearest_node(rand_node)
            
            # 扩展新节点
            new_node = self.steer(nearest_node, rand_node)
            
            # 动态碰撞检测
            if self.check_dynamic_collision(nearest_node, new_node, self.current_time):
                continue
            
            # 添加节点
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + self.distance(nearest_node, new_node)
            self.node_list.append(new_node)
            
            # 检查目标
            if self.distance(new_node, self.goal) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                if not self.check_dynamic_collision(new_node, final_node, self.current_time):
                    final_node.parent = new_node
                    return self.generate_final_course(final_node)
        
        return None
    
    def visualize_dynamic(self, path=None, time_snapshot=0):
        """可视化动态环境"""
        plt.figure(figsize=(12, 10))
        
        # 绘制边界
        boundary_rect = patches.Rectangle(
            (self.boundary[0], self.boundary[2]),
            self.boundary[1] - self.boundary[0],
            self.boundary[3] - self.boundary[2],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        plt.gca().add_patch(boundary_rect)
        
        # 绘制静态障碍物
        for (ox, oy, radius) in self.obstacle_list:
            circle = patches.Circle((ox, oy), radius,
                                  facecolor='red', alpha=0.6, edgecolor='darkred',
                                  label='静态障碍物' if (ox, oy, radius) == self.obstacle_list[0] else "")
            plt.gca().add_patch(circle)
        
        # 绘制动态障碍物
        for i, obstacle in enumerate(self.moving_obstacles):
            future_pos = self.get_obstacle_position(obstacle, time_snapshot)
            circle = patches.Circle((future_pos[0], future_pos[1]), future_pos[2],
                                  facecolor='orange', alpha=0.6, edgecolor='darkorange',
                                  label='动态障碍物' if i == 0 else "")
            plt.gca().add_patch(circle)
            
            # 绘制运动轨迹
            x, y, _, vx, vy = obstacle
            trail_x = [x + vx * t for t in np.linspace(0, self.time_horizon, 20)]
            trail_y = [y + vy * t for t in np.linspace(0, self.time_horizon, 20)]
            plt.plot(trail_x, trail_y, '--', color='orange', alpha=0.5)
        
        # 绘制RRT树
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y],
                        'g-', alpha=0.3, linewidth=1)
        
        # 绘制节点
        node_coords = np.array([(node.x, node.y) for node in self.node_list])
        if len(node_coords) > 0:
            plt.scatter(node_coords[:, 0], node_coords[:, 1],
                       c='lightgreen', s=20, alpha=0.6)
        
        # 绘制起点和终点
        plt.scatter(self.start.x, self.start.y, c='blue', s=100,
                   marker='o', label='起点', zorder=5)
        plt.scatter(self.goal.x, self.goal.y, c='red', s=100,
                   marker='*', label='目标', zorder=5)
        
        # 绘制路径
        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'purple', linewidth=4, label='规划路径')
        
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title(f'动态环境RRT (时间: {time_snapshot:.1f}s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

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

def demo_birrt():
    """双向RRT演示"""
    print("↔️ 双向RRT算法演示")
    
    start = (2, 10)
    goal = (18, 10)
    obstacles = [(10, 10, 3), (6, 6, 2), (14, 14, 2)]
    boundary = (0, 20, 0, 20)
    
    # 创建双向RRT规划器
    birrt = BiRRT(start=start, goal=goal, obstacle_list=obstacles, 
                  boundary=boundary, max_iter=1500)
    
    print("📊 正在从两端搜索路径...")
    path = birrt.plan()
    
    if path:
        print(f"✅ 找到路径! 长度: {len(path)}个点")
        print(f"🌳 Start树节点数: {len(birrt.node_list)}")
        print(f"🌳 Goal树节点数: {len(birrt.goal_tree)}")
        
        birrt.draw(path=path, show_tree=True)
        return path
    else:
        print("❌ 未找到可行路径")
        birrt.draw(show_tree=True)
        return None

def demo_dynamic_rrt():
    """动态环境RRT演示"""
    print("🏃 动态环境RRT算法演示")
    
    start = (2, 2)
    goal = (18, 18)
    static_obstacles = [(10, 10, 2)]
    
    # 移动障碍物: (x, y, radius, vx, vy)
    moving_obstacles = [
        (5, 15, 1.5, 1.0, -0.5),  # 向右下移动
        (15, 5, 1.2, -0.8, 1.2),  # 向左上移动
    ]
    
    boundary = (0, 20, 0, 20)
    
    # 创建动态RRT规划器
    dynamic_rrt = DynamicRRT(
        start=start, goal=goal,
        static_obstacles=static_obstacles,
        moving_obstacles=moving_obstacles,
        boundary=boundary,
        max_iter=2000, time_horizon=10
    )
    
    print("📊 正在规划动态环境路径...")
    path = dynamic_rrt.plan_with_time()
    
    if path:
        print(f"✅ 找到动态路径! 长度: {len(path)}个点")
        
        # 可视化不同时间快照
        for t in [0, 2, 4]:
            dynamic_rrt.visualize_dynamic(path=path, time_snapshot=t)
        
        return path
    else:
        print("❌ 未找到可行动态路径")
        dynamic_rrt.visualize_dynamic(time_snapshot=0)
        return None

if __name__ == "__main__":
    print("🚀 RRT高级特性演示")
    print("=" * 50)
    
    # 演示所有高级特性
    demo_informed_rrt_star()
    demo_birrt()
    demo_dynamic_rrt() 