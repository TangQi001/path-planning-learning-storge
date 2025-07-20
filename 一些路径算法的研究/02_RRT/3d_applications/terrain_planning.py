
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
3D地形路径规划 - RRT在三维环境中的应用

作者: AICP-7协议实现
功能: 地形感知的无人机路径规划
特点: 3D碰撞检测、地形约束、高度优化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import sys
import os

# 添加代码实现目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_代码实现'))

class Node3D:
    """3D节点类"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

    def __repr__(self):
        return f"Node3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class Terrain3D:
    """3D地形生成器"""
    
    def __init__(self, x_range, y_range, resolution=1.0):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # 生成地形网格
        self.x_grid = np.arange(x_range[0], x_range[1], resolution)
        self.y_grid = np.arange(y_range[0], y_range[1], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # 生成地形高度
        self.Z = self.generate_terrain()
        
        # 生成建筑物/障碍物
        self.buildings = self.generate_buildings()
    
    def generate_terrain(self):
        """生成基础地形高度"""
        # 使用多个正弦波创建起伏地形
        terrain = (
            3 * np.sin(0.1 * self.X) * np.cos(0.1 * self.Y) +
            2 * np.sin(0.2 * self.X + 1) * np.cos(0.15 * self.Y + 1) +
            1.5 * np.sin(0.05 * self.X) +
            np.random.normal(0, 0.3, self.X.shape)  # 添加噪声
        )
        
        # 确保地形高度为正
        terrain = np.maximum(terrain, 0)
        
        return terrain
    
    def generate_buildings(self):
        """生成建筑物障碍"""
        buildings = []
        num_buildings = 8
        
        for _ in range(num_buildings):
            # 随机位置
            x = random.uniform(self.x_range[0] + 5, self.x_range[1] - 5)
            y = random.uniform(self.y_range[0] + 5, self.y_range[1] - 5)
            
            # 建筑物尺寸
            width = random.uniform(2, 5)
            length = random.uniform(2, 5) 
            height = random.uniform(5, 15)
            
            # 基础高度（地形高度）
            base_height = self.get_terrain_height(x, y)
            
            buildings.append({
                'center': (x, y, base_height + height/2),
                'size': (width, length, height),
                'base_height': base_height
            })
        
        return buildings
    
    def get_terrain_height(self, x, y):
        """获取指定位置的地形高度"""
        if (x < self.x_range[0] or x > self.x_range[1] or 
            y < self.y_range[0] or y > self.y_range[1]):
            return 0
        
        # 双线性插值
        x_idx = np.clip(int((x - self.x_range[0]) / self.resolution), 0, len(self.x_grid) - 2)
        y_idx = np.clip(int((y - self.y_range[0]) / self.resolution), 0, len(self.y_grid) - 2)
        
        # 插值权重
        x_weight = (x - self.x_grid[x_idx]) / self.resolution
        y_weight = (y - self.y_grid[y_idx]) / self.resolution
        
        # 双线性插值
        height = (
            (1 - x_weight) * (1 - y_weight) * self.Z[y_idx, x_idx] +
            x_weight * (1 - y_weight) * self.Z[y_idx, x_idx + 1] +
            (1 - x_weight) * y_weight * self.Z[y_idx + 1, x_idx] +
            x_weight * y_weight * self.Z[y_idx + 1, x_idx + 1]
        )
        
        return height

class RRT3D:
    """3D环境RRT算法"""
    
    def __init__(self, start, goal, terrain, boundary, 
                 step_size=2.0, goal_sample_rate=0.1, max_iter=3000,
                 min_altitude=2.0, max_altitude=25.0):
        """
        初始化3D RRT规划器
        
        Args:
            start: (x, y, z) 起始点
            goal: (x, y, z) 目标点
            terrain: Terrain3D对象
            boundary: (x_min, x_max, y_min, y_max) 边界
            step_size: 扩展步长
            goal_sample_rate: 目标偏向采样概率
            max_iter: 最大迭代次数
            min_altitude: 最小飞行高度
            max_altitude: 最大飞行高度
        """
        self.start = Node3D(start[0], start[1], start[2])
        self.goal = Node3D(goal[0], goal[1], goal[2])
        self.terrain = terrain
        self.boundary = boundary
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        
        # 树结构存储
        self.node_list = [self.start]
    
    def plan(self):
        """执行3D RRT路径规划"""
        for i in range(self.max_iter):
            # 1. 随机采样
            rand_node = self.sample()
            
            # 2. 找到最近邻节点
            nearest_node = self.get_nearest_node(rand_node)
            
            # 3. 扩展新节点
            new_node = self.steer(nearest_node, rand_node)
            
            # 4. 3D碰撞检测
            if self.check_collision_3d(nearest_node, new_node):
                continue
            
            # 5. 添加到树中
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + self.distance_3d(nearest_node, new_node)
            self.node_list.append(new_node)
            
            # 6. 检查是否到达目标
            if self.distance_3d(new_node, self.goal) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                if not self.check_collision_3d(new_node, final_node):
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + self.distance_3d(new_node, final_node)
                    return self.generate_final_course(final_node)
        
        return None
    
    def sample(self):
        """3D随机采样"""
        if random.random() > self.goal_sample_rate:
            # 在边界内随机采样
            x = random.uniform(self.boundary[0], self.boundary[1])
            y = random.uniform(self.boundary[2], self.boundary[3])
            z = random.uniform(self.min_altitude, self.max_altitude)
            return Node3D(x, y, z)
        else:
            # 目标偏向采样
            return Node3D(self.goal.x, self.goal.y, self.goal.z)
    
    def get_nearest_node(self, target_node):
        """找到3D空间中距离目标最近的节点"""
        distances = [self.distance_3d(node, target_node) for node in self.node_list]
        min_index = distances.index(min(distances))
        return self.node_list[min_index]
    
    def steer(self, from_node, to_node):
        """3D扩展操作"""
        dist = self.distance_3d(from_node, to_node)
        
        if dist <= self.step_size:
            return Node3D(to_node.x, to_node.y, to_node.z)
        
        # 计算扩展方向
        direction = np.array([
            to_node.x - from_node.x,
            to_node.y - from_node.y, 
            to_node.z - from_node.z
        ])
        direction = direction / np.linalg.norm(direction)
        
        new_pos = np.array([from_node.x, from_node.y, from_node.z]) + self.step_size * direction
        
        return Node3D(new_pos[0], new_pos[1], new_pos[2])
    
    def check_collision_3d(self, from_node, to_node):
        """3D碰撞检测"""
        # 检查边界
        if (to_node.x < self.boundary[0] or to_node.x > self.boundary[1] or
            to_node.y < self.boundary[2] or to_node.y > self.boundary[3] or
            to_node.z < self.min_altitude or to_node.z > self.max_altitude):
            return True
        
        # 检查地面碰撞
        terrain_height = self.terrain.get_terrain_height(to_node.x, to_node.y)
        if to_node.z <= terrain_height + self.min_altitude:
            return True
        
        # 检查建筑物碰撞
        for building in self.terrain.buildings:
            if self.point_in_building(to_node, building):
                return True
        
        # 检查路径与建筑物碰撞
        return self.path_building_collision(from_node, to_node)
    
    def point_in_building(self, point, building):
        """检查点是否在建筑物内"""
        center = building['center']
        size = building['size']
        
        return (abs(point.x - center[0]) <= size[0]/2 and
                abs(point.y - center[1]) <= size[1]/2 and
                point.z >= building['base_height'] and
                point.z <= center[2] + size[2]/2)
    
    def path_building_collision(self, from_node, to_node):
        """检查路径是否与建筑物碰撞"""
        # 路径离散化检查
        num_checks = int(self.distance_3d(from_node, to_node) / 0.5) + 1
        
        for i in range(num_checks):
            t = i / (num_checks - 1) if num_checks > 1 else 0
            
            check_point = Node3D(
                from_node.x + t * (to_node.x - from_node.x),
                from_node.y + t * (to_node.y - from_node.y),
                from_node.z + t * (to_node.z - from_node.z)
            )
            
            # 检查地面
            terrain_height = self.terrain.get_terrain_height(check_point.x, check_point.y)
            if check_point.z <= terrain_height + self.min_altitude:
                return True
            
            # 检查建筑物
            for building in self.terrain.buildings:
                if self.point_in_building(check_point, building):
                    return True
        
        return False
    
    def distance_3d(self, node1, node2):
        """计算3D欧几里得距离"""
        return math.sqrt((node1.x - node2.x)**2 + 
                        (node1.y - node2.y)**2 + 
                        (node1.z - node2.z)**2)
    
    def generate_final_course(self, goal_node):
        """生成最终3D路径"""
        path = []
        node = goal_node
        
        while node is not None:
            path.append((node.x, node.y, node.z))
            node = node.parent
            
        return path[::-1]
    
    def visualize_3d(self, path=None, show_tree=True, save_path=None):
        """3D可视化"""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地形
        ax.plot_surface(self.terrain.X, self.terrain.Y, self.terrain.Z, 
                       alpha=0.3, cmap='terrain', linewidth=0)
        
        # 绘制建筑物
        for building in self.terrain.buildings:
            center = building['center']
            size = building['size']
            base = building['base_height']
            
            # 建筑物的8个顶点
            x_coords = [center[0] - size[0]/2, center[0] + size[0]/2]
            y_coords = [center[1] - size[1]/2, center[1] + size[1]/2]
            z_coords = [base, center[2] + size[2]/2]
            
            # 绘制建筑物边框
            for x in x_coords:
                for y in y_coords:
                    ax.plot([x, x], [y, y], z_coords, 'k-', alpha=0.6)
            
            for x in x_coords:
                for z in z_coords:
                    ax.plot([x, x], y_coords, [z, z], 'k-', alpha=0.6)
            
            for y in y_coords:
                for z in z_coords:
                    ax.plot(x_coords, [y, y], [z, z], 'k-', alpha=0.6)
        
        # 绘制RRT树
        if show_tree:
            for node in self.node_list:
                if node.parent:
                    ax.plot([node.x, node.parent.x], 
                           [node.y, node.parent.y],
                           [node.z, node.parent.z], 
                           'g-', alpha=0.3, linewidth=1)
        
        # 绘制节点
        node_coords = np.array([(node.x, node.y, node.z) for node in self.node_list])
        if len(node_coords) > 0:
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                      c='lightgreen', s=20, alpha=0.6)
        
        # 绘制起点和终点
        ax.scatter(self.start.x, self.start.y, self.start.z, 
                  c='blue', s=100, marker='o', label='起点')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, 
                  c='red', s=100, marker='*', label='目标')
        
        # 绘制路径
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                   'b-', linewidth=4, label='规划路径')
        
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_zlabel('Z坐标 (m)')
        ax.set_title('3D地形环境RRT路径规划')
        ax.legend()
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def demo_3d_terrain_planning():
    """3D地形路径规划演示"""
    print("🏔️ 3D地形路径规划演示")
    
    # 创建地形
    terrain = Terrain3D(x_range=(0, 50), y_range=(0, 50), resolution=1.0)
    
    # 设置起点和终点
    start = (5, 5, 15)
    goal = (45, 45, 12)
    boundary = (0, 50, 0, 50)
    
    # 创建3D RRT规划器
    rrt_3d = RRT3D(
        start=start,
        goal=goal,
        terrain=terrain,
        boundary=boundary,
        step_size=3.0,
        goal_sample_rate=0.1,
        max_iter=3000,
        min_altitude=2.0,
        max_altitude=25.0
    )
    
    print("📊 正在搜索3D路径...")
    path = rrt_3d.plan()
    
    if path:
        print(f"✅ 找到3D路径! 包含 {len(path)} 个航点")
        
        # 计算路径长度
        path_length = sum(math.sqrt((path[i][0] - path[i-1][0])**2 + 
                                   (path[i][1] - path[i-1][1])**2 + 
                                   (path[i][2] - path[i-1][2])**2)
                         for i in range(1, len(path)))
        print(f"📏 3D路径总长度: {path_length:.2f} m")
        print(f"🌳 生成节点数: {len(rrt_3d.node_list)}")
        
        # 高度统计
        heights = [point[2] for point in path]
        print(f"🚁 飞行高度: {min(heights):.1f} - {max(heights):.1f} m")
        
        # 可视化结果
        rrt_3d.visualize_3d(path=path, show_tree=True)
        
        return path
    else:
        print("❌ 未找到可行3D路径")
        rrt_3d.visualize_3d(show_tree=True)
        return None

if __name__ == "__main__":
    demo_3d_terrain_planning() 