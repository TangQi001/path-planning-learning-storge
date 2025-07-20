"""
RRT* (RRT Star) 算法实现

作者: AICP-7协议实现  
功能: 渐进最优路径规划
特点: 重连机制、代价优化、收敛最优解
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
from collections import defaultdict
import sys
import os

# 中文字体配置
try:
    from font_config import setup_chinese_font
    setup_chinese_font()
except ImportError:
    # 如果无法导入字体配置，使用基本配置
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到路径，以便导入rrt_basic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rrt_basic import Node, RRTBasic

class RRTStar(RRTBasic):
    """RRT*算法 - 继承自基础RRT并添加重连机制"""
    
    def __init__(self, start, goal, obstacle_list, boundary,
                 step_size=1.0, goal_sample_rate=0.1, max_iter=2000,
                 search_radius=3.0, gamma=50.0):
        """
        初始化RRT*规划器
        
        Args:
            search_radius: 重连搜索半径
            gamma: 动态半径计算参数
            其他参数同RRTBasic
        """
        super().__init__(start, goal, obstacle_list, boundary, 
                        step_size, goal_sample_rate, max_iter)
        
        self.search_radius = search_radius
        self.gamma = gamma
        
    def plan(self):
        """
        执行RRT*路径规划 (带重连机制)
        
        Returns:
            path: 优化后的路径点列表 [(x, y), ...] 或 None
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
            
            # 5. 在搜索半径内找到邻居节点
            near_nodes = self.find_near_nodes(new_node)
            
            # 6. 选择最优父节点 (Choose Parent)
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + self.distance(nearest_node, new_node)
            
            for near_node in near_nodes:
                if (not self.check_collision(near_node, new_node) and
                    near_node.cost + self.distance(near_node, new_node) < min_cost):
                    min_cost_node = near_node
                    min_cost = near_node.cost + self.distance(near_node, new_node)
            
            # 7. 连接最优父节点
            new_node.parent = min_cost_node
            new_node.cost = min_cost
            self.node_list.append(new_node)
            
            # 8. 重连操作 (Rewire)
            self.rewire(new_node, near_nodes)
            
            # 9. 检查是否到达目标
            if self.distance(new_node, self.goal) <= self.step_size:
                final_node = self.steer(new_node, self.goal)
                if not self.check_collision(new_node, final_node):
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + self.distance(new_node, final_node)
                    return self.generate_final_course(final_node)
                    
        return None
    
    def find_near_nodes(self, target_node):
        """
        找到搜索半径内的邻居节点
        
        Args:
            target_node: 目标节点
            
        Returns:
            near_nodes: 邻居节点列表
        """
        # 动态调整搜索半径 (RRT*理论)
        n = len(self.node_list)
        radius = min(self.search_radius, 
                    self.gamma * math.sqrt(math.log(n) / n))
        
        near_nodes = []
        for node in self.node_list:
            if self.distance(node, target_node) <= radius:
                near_nodes.append(node)
                
        return near_nodes
    
    def rewire(self, new_node, near_nodes):
        """
        重连操作 - RRT*的核心优化机制
        
        Args:
            new_node: 新添加的节点
            near_nodes: 邻居节点列表
        """
        for near_node in near_nodes:
            # 计算通过new_node到达near_node的代价
            potential_cost = new_node.cost + self.distance(new_node, near_node)
            
            # 如果新路径更优且无碰撞，则重连
            if (potential_cost < near_node.cost and 
                not self.check_collision(new_node, near_node)):
                
                # 重连near_node到new_node
                near_node.parent = new_node
                near_node.cost = potential_cost
                
                # 递归更新受影响的子节点代价
                self.update_costs(near_node)
    
    def update_costs(self, node):
        """
        递归更新节点及其子节点的代价
        
        Args:
            node: 需要更新的节点
        """
        for child_node in self.node_list:
            if child_node.parent == node:
                child_node.cost = node.cost + self.distance(node, child_node)
                self.update_costs(child_node)
    
    def get_best_path_to_goal(self):
        """
        获取到目标的最优路径 (RRT*可能有多条到目标的路径)
        
        Returns:
            best_path: 最优路径或None
        """
        goal_candidates = []
        
        # 找到所有能到达目标区域的节点
        for node in self.node_list:
            if self.distance(node, self.goal) <= self.step_size:
                final_node = self.steer(node, self.goal)
                if not self.check_collision(node, final_node):
                    final_node.parent = node
                    final_node.cost = node.cost + self.distance(node, final_node)
                    goal_candidates.append(final_node)
        
        if not goal_candidates:
            return None
            
        # 选择代价最小的路径
        best_goal = min(goal_candidates, key=lambda x: x.cost)
        return self.generate_final_course(best_goal)

def demo_rrt_star():
    """RRT*算法演示"""
    print("🌟 开始RRT*算法演示...")
    
    # 问题设置
    start = (2, 2)
    goal = (18, 18)
    
    obstacles = [
        (5, 5, 1.5),
        (8, 8, 1.0),
        (12, 3, 1.5),
        (15, 12, 2.0),
        (7, 15, 1.2),
        (13, 8, 1.8),
        (3, 12, 1.0),
        (16, 6, 1.3)
    ]
    
    boundary = (0, 20, 0, 20)
    
    # 创建RRT*规划器
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacles,
        boundary=boundary,
        step_size=1.0,
        goal_sample_rate=0.1,
        max_iter=3000,
        search_radius=3.0,
        gamma=50.0
    )
    
    print("📊 正在搜索最优路径...")
    path = rrt_star.plan()
    
    if path:
        print(f"✅ 找到路径! 长度: {len(path)}个点")
        
        # 计算路径代价
        path_length = sum(math.sqrt((path[i][0] - path[i-1][0])**2 + 
                                   (path[i][1] - path[i-1][1])**2)
                         for i in range(1, len(path)))
        print(f"📏 路径总长度: {path_length:.2f}")
        print(f"🌳 生成节点数: {len(rrt_star.node_list)}")
        
        # 可视化结果
        rrt_star.draw(path=path, show_tree=True)
        
        return path, rrt_star
    else:
        print("❌ 未找到可行路径")
        rrt_star.draw(show_tree=True)
        return None, rrt_star

def rrt_vs_rrt_star_comparison():
    """RRT vs RRT* 性能对比"""
    print("\n⚔️ RRT vs RRT* 性能对比...")
    
    start = (1, 1)
    goal = (19, 19)
    obstacles = [
        (6, 6, 1.5),
        (10, 4, 1.2), 
        (14, 8, 1.8),
        (8, 12, 1.0),
        (15, 15, 1.5)
    ]
    boundary = (0, 20, 0, 20)
    
    # 多次运行统计
    num_runs = 5
    rrt_results = []
    rrt_star_results = []
    
    print(f"📊 进行 {num_runs} 次独立测试...")
    
    for i in range(num_runs):
        print(f"  运行 {i+1}/{num_runs}...")
        
        # 基础RRT
        rrt_basic = RRTBasic(start, goal, obstacles, boundary, max_iter=2000)
        rrt_path = rrt_basic.plan()
        
        if rrt_path:
            length = sum(math.sqrt((rrt_path[j][0] - rrt_path[j-1][0])**2 + 
                                 (rrt_path[j][1] - rrt_path[j-1][1])**2)
                        for j in range(1, len(rrt_path)))
            rrt_results.append({
                'length': length,
                'nodes': len(rrt_basic.node_list),
                'success': True
            })
        else:
            rrt_results.append({'success': False})
        
        # RRT*
        rrt_star = RRTStar(start, goal, obstacles, boundary, max_iter=2000)
        rrt_star_path = rrt_star.plan()
        
        if rrt_star_path:
            length = sum(math.sqrt((rrt_star_path[j][0] - rrt_star_path[j-1][0])**2 + 
                                 (rrt_star_path[j][1] - rrt_star_path[j-1][1])**2)
                        for j in range(1, len(rrt_star_path)))
            rrt_star_results.append({
                'length': length,
                'nodes': len(rrt_star.node_list),
                'success': True
            })
        else:
            rrt_star_results.append({'success': False})
    
    # 统计分析
    print("\n📈 性能对比结果:")
    print("=" * 50)
    
    # 成功率
    rrt_success = sum(1 for r in rrt_results if r['success']) / num_runs
    rrt_star_success = sum(1 for r in rrt_star_results if r['success']) / num_runs
    
    print(f"成功率:")
    print(f"  RRT:      {rrt_success:.1%}")
    print(f"  RRT*:     {rrt_star_success:.1%}")
    
    # 路径长度统计
    rrt_lengths = [r['length'] for r in rrt_results if r['success']]
    rrt_star_lengths = [r['length'] for r in rrt_star_results if r['success']]
    
    if rrt_lengths and rrt_star_lengths:
        print(f"\n路径长度:")
        print(f"  RRT  平均:  {np.mean(rrt_lengths):.2f} ± {np.std(rrt_lengths):.2f}")
        print(f"  RRT* 平均:  {np.mean(rrt_star_lengths):.2f} ± {np.std(rrt_star_lengths):.2f}")
        print(f"  改善程度:   {(np.mean(rrt_lengths) - np.mean(rrt_star_lengths)) / np.mean(rrt_lengths):.1%}")
    
    # 可视化一个对比例子
    print("\n🎨 生成对比可视化...")
    rrt_basic = RRTBasic(start, goal, obstacles, boundary, max_iter=2000)
    rrt_path = rrt_basic.plan()
    
    rrt_star = RRTStar(start, goal, obstacles, boundary, max_iter=2000)
    rrt_star_path = rrt_star.plan()
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制RRT结果
    plt.sca(ax1)
    rrt_basic.draw(path=rrt_path, show_tree=True)
    ax1.set_title('基础RRT算法')
    
    # 绘制RRT*结果
    plt.sca(ax2)
    rrt_star.draw(path=rrt_star_path, show_tree=True)
    ax2.set_title('RRT*算法')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # RRT*演示
    demo_rrt_star()
    
    # 性能对比
    rrt_vs_rrt_star_comparison() 