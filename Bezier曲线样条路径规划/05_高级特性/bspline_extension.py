#!/usr/bin/env python3
"""
B样条曲线扩展实现

这个模块展示了B样条曲线作为Bezier曲线的高级扩展
主要优势：局部控制性、更好的连续性、适合长路径
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import sys
import os

# 配置matplotlib支持中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加核心算法模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02_代码实现'))
from core_algorithm import Point, Constraints

class BSplineCurve:
    """B样条曲线类"""
    
    def __init__(self, control_points: List[Point], degree: int = 3):
        """
        初始化B样条曲线
        
        Args:
            control_points: 控制点列表
            degree: B样条次数（通常为3）
        """
        if len(control_points) < degree + 1:
            raise ValueError(f"控制点数量({len(control_points)})必须大于等于次数+1({degree + 1})")
        
        self.control_points = control_points
        self.degree = degree
        self.n = len(control_points) - 1
        
        # 生成均匀节点向量
        self.knot_vector = self._generate_uniform_knots()
    
    def _generate_uniform_knots(self) -> List[float]:
        """生成均匀节点向量"""
        # B样条节点向量长度 = n + p + 2 = 控制点数 + 次数 + 1
        m = len(self.control_points) + self.degree + 1
        knots = []
        
        # 开始的重复节点
        for i in range(self.degree + 1):
            knots.append(0.0)
        
        # 中间的内部节点
        num_internal = m - 2 * (self.degree + 1)
        if num_internal > 0:
            for i in range(num_internal):
                knots.append((i + 1) / (num_internal + 1))
        
        # 结束的重复节点
        for i in range(self.degree + 1):
            knots.append(1.0)
            
        return knots
    
    def _basis_function(self, i: int, p: int, t: float) -> float:
        """计算B样条基函数"""
        # 检查索引边界
        if i < 0 or i >= len(self.control_points):
            return 0.0
        
        if p == 0:
            # 检查节点向量边界
            if i + 1 >= len(self.knot_vector):
                return 0.0
            if self.knot_vector[i] <= t < self.knot_vector[i + 1]:
                return 1.0
            else:
                return 0.0
        
        result = 0.0
        
        # 第一项
        if i + p < len(self.knot_vector):
            denom1 = self.knot_vector[i + p] - self.knot_vector[i]
            if abs(denom1) > 1e-10:
                result += (t - self.knot_vector[i]) / denom1 * self._basis_function(i, p - 1, t)
        
        # 第二项  
        if i + p + 1 < len(self.knot_vector) and i + 1 < len(self.knot_vector):
            denom2 = self.knot_vector[i + p + 1] - self.knot_vector[i + 1]
            if abs(denom2) > 1e-10:
                result += (self.knot_vector[i + p + 1] - t) / denom2 * self._basis_function(i + 1, p - 1, t)
        
        return result
    
    def evaluate(self, t: float) -> Point:
        """计算B样条曲线上的点"""
        t = max(0.0, min(1.0, t))
        if t == 1.0:
            t = 1.0 - 1e-10
        
        result = Point(0, 0, 0)
        
        for i in range(len(self.control_points)):
            basis_value = self._basis_function(i, self.degree, t)
            cp = self.control_points[i]
            result = result + cp * basis_value
        
        return result
    
    def sample_points(self, num_points: int) -> List[Point]:
        """采样曲线上的点"""
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(self.evaluate(t))
        return points

def demo_bspline_vs_bezier():
    """演示B样条与Bezier曲线的对比"""
    print("=== B样条与Bezier曲线对比演示 ===")
    
    # 创建测试航点  
    waypoints = [
        Point(0, 0),
        Point(2, 3),
        Point(4, 1), 
        Point(6, 4),
        Point(8, 2),
        Point(10, 3)
    ]
    
    # 创建B样条曲线
    bspline = BSplineCurve(waypoints, degree=3)
    
    # 创建对比的分段Bezier曲线
    from core_algorithm import PathPlanner
    bezier_planner = PathPlanner()
    bezier_path = bezier_planner.generate_smooth_path(waypoints)
    
    # 可视化
    plt.figure(figsize=(14, 10))
    
    # 子图1：路径对比
    plt.subplot(2, 2, 1)
    
    # 航点
    wp_x = [wp.x for wp in waypoints]
    wp_y = [wp.y for wp in waypoints]
    plt.plot(wp_x, wp_y, 'ro-', markersize=8, linewidth=2, 
             label='航点', alpha=0.7)
    
    # B样条曲线
    bspline_points = bspline.sample_points(100)
    bs_x = [p.x for p in bspline_points]
    bs_y = [p.y for p in bspline_points]
    plt.plot(bs_x, bs_y, 'b-', linewidth=3, label='B样条曲线')
    
    # Bezier路径
    bezier_points = []
    for segment in bezier_path.segments:
        points = segment.sample_points(20)
        bezier_points.extend(points)
    
    bz_x = [p.x for p in bezier_points]
    bz_y = [p.y for p in bezier_points] 
    plt.plot(bz_x, bz_y, 'g--', linewidth=2, label='分段Bezier')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('路径对比')
    plt.axis('equal')
    
    # 子图2：局部控制性演示
    plt.subplot(2, 2, 2)
    
    # 原始B样条
    original_points = waypoints.copy()
    original_bspline = BSplineCurve(original_points, degree=3)
    
    # 修改一个控制点
    modified_points = waypoints.copy()
    modified_points[3] = Point(6, 0)  # 下移第4个点
    modified_bspline = BSplineCurve(modified_points, degree=3)
    
    # 绘制
    orig_pts = original_bspline.sample_points(100)
    mod_pts = modified_bspline.sample_points(100)
    
    plt.plot([p.x for p in orig_pts], [p.y for p in orig_pts], 
             'b-', linewidth=2, label='原始B样条')
    plt.plot([p.x for p in mod_pts], [p.y for p in mod_pts], 
             'r--', linewidth=2, label='修改后B样条')
    
    # 控制点
    plt.plot([p.x for p in original_points], [p.y for p in original_points], 
             'bo', markersize=6)
    plt.plot([p.x for p in modified_points], [p.y for p in modified_points], 
             'rs', markersize=6)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('局部控制性')
    plt.axis('equal')
    
    # 子图3：连续性对比
    plt.subplot(2, 2, 3)
    
    # 显示Bezier路径的连接处
    for i, segment in enumerate(bezier_path.segments):
        points = segment.sample_points(20)
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        plt.plot(x_coords, y_coords, linewidth=2, 
                label=f'Bezier段{i+1}', alpha=0.7)
    
    plt.plot(bs_x, bs_y, 'k-', linewidth=3, label='B样条(天然连续)', alpha=0.8)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('连续性对比')
    plt.axis('equal')
    
    # 子图4：复杂度对比
    plt.subplot(2, 2, 4)
    
    data = ['控制点数', '曲线段数', '连接点数']
    bspline_data = [len(waypoints), 1, 0]
    bezier_data = [len(bezier_path.segments)*4, len(bezier_path.segments), 
                   len(bezier_path.segments)-1]
    
    x = np.arange(len(data))
    width = 0.35
    
    plt.bar(x - width/2, bspline_data, width, label='B样条', alpha=0.8)
    plt.bar(x + width/2, bezier_data, width, label='分段Bezier', alpha=0.8)
    
    plt.xlabel('属性')
    plt.ylabel('数量')
    plt.title('复杂度对比')
    plt.xticks(x, data)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 输出对比信息
    print("\n=== 对比结果 ===")
    print(f"航点数量: {len(waypoints)}")
    print(f"\nB样条曲线:")
    print(f"  - 控制点数: {len(waypoints)}")
    print(f"  - 曲线段数: 1 (单一曲线)")
    print(f"  - 连续性: 天然C²连续")
    print(f"  - 局部控制: ✓")
    
    print(f"\n分段Bezier:")
    print(f"  - 总控制点数: {len(bezier_path.segments) * 4}")
    print(f"  - 曲线段数: {len(bezier_path.segments)}")
    print(f"  - 连续性: 需要手动保证")
    print(f"  - 局部控制: 有限")

if __name__ == "__main__":
    demo_bspline_vs_bezier()
    
    print("\n=== B样条的优势总结 ===")
    print("1. 🎯 局部控制性：修改一个控制点只影响局部区域")
    print("2. 🔗 更好连续性：天然支持C²连续，无需手动调整")  
    print("3. 📏 适合长路径：单条曲线可处理多个航点")
    print("4. 🔧 灵活性强：通过节点向量可精确控制形状")
    print("5. 🚫 避免龙格现象：高次插值仍然稳定")
    
    print("\n=== 应用建议 ===")
    print("• 长距离路径规划 → 使用B样条")
    print("• 需要局部调整 → 使用B样条")
    print("• 简单短路径 → 使用Bezier曲线")
    print("• 实时计算 → 使用Bezier曲线") 