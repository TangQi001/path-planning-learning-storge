"""
Dubins路径六种序列详细实现
作者：AI助手
日期：2025年1月
功能：实现六种Dubins路径类型的完整计算和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import math

class DubinsPath:
    """Dubins路径类，实现六种路径类型的计算"""
    
    def __init__(self, turning_radius: float = 1.0):
        """
        初始化Dubins路径计算器
        
        Args:
            turning_radius: 最小转弯半径
        """
        self.turning_radius = turning_radius
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
        
    def mod2pi(self, angle: float) -> float:
        """将角度标准化到[0, 2π]范围"""
        return angle - 2 * math.pi * math.floor(angle / (2 * math.pi))
    
    def coordinate_transform(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        坐标变换：将问题转换为标准坐标系
        
        Args:
            start: 起点 (x, y, theta)
            end: 终点 (x, y, theta)
            
        Returns:
            (d, alpha, beta): 标准化距离和角度
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # 计算标准化距离
        d = math.sqrt(dx*dx + dy*dy) / self.turning_radius
        
        # 计算相对角度
        theta = math.atan2(dy, dx)
        alpha = self.mod2pi(start[2] - theta)
        beta = self.mod2pi(end[2] - theta)
        
        return d, alpha, beta
    
    def compute_rsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算RSR路径（右转-直行-右转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        tmp = math.atan2(math.cos(alpha) - math.cos(beta), 
                        d - math.sin(alpha) + math.sin(beta))
        
        t1 = self.mod2pi(alpha - tmp)
        p = math.sqrt(2 + d*d - 2*math.cos(alpha - beta) + 
                     2*d*(math.sin(beta) - math.sin(alpha)))
        t2 = self.mod2pi(-beta + tmp)
        
        return t1, p, t2, True  # RSR总是可行的
    
    def compute_lsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算LSL路径（左转-直行-左转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        tmp = math.atan2(math.cos(beta) - math.cos(alpha), 
                        d + math.sin(alpha) - math.sin(beta))
        
        t1 = self.mod2pi(-alpha + tmp)
        p = math.sqrt(2 + d*d - 2*math.cos(alpha - beta) + 
                     2*d*(math.sin(alpha) - math.sin(beta)))
        t2 = self.mod2pi(beta - tmp)
        
        return t1, p, t2, True  # LSL总是可行的
    
    def compute_rsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算RSL路径（右转-直行-左转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        p_squared = d*d - 2 + 2*math.cos(alpha - beta) - 2*d*(math.sin(alpha) + math.sin(beta))
        
        if p_squared < 0:
            return 0, 0, 0, False
        
        p = math.sqrt(p_squared)
        tmp = math.atan2(math.cos(alpha) + math.cos(beta), 
                        d - math.sin(alpha) - math.sin(beta)) - math.atan2(2, p)
        
        t1 = self.mod2pi(alpha - tmp)
        t2 = self.mod2pi(beta - tmp)
        
        return t1, p, t2, True
    
    def compute_lsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算LSR路径（左转-直行-右转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        p_squared = -2 + d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) + math.sin(beta))
        
        if p_squared < 0:
            return 0, 0, 0, False
        
        p = math.sqrt(p_squared)
        tmp = math.atan2(-math.cos(alpha) - math.cos(beta), 
                        d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2, p)
        
        t1 = self.mod2pi(-alpha + tmp)
        t2 = self.mod2pi(-beta + tmp)
        
        return t1, p, t2, True
    
    def compute_rlr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算RLR路径（右转-左转-右转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) - math.sin(beta))) / 8
        
        if abs(tmp) > 1:
            return 0, 0, 0, False
        
        p = self.mod2pi(2*math.pi - math.acos(tmp))
        t1 = self.mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), 
                                           d - math.sin(alpha) + math.sin(beta)) + p/2)
        t2 = self.mod2pi(alpha - beta - t1 + p)
        
        return t1, p, t2, True
    
    def compute_lrl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """
        计算LRL路径（左转-右转-左转）
        
        Args:
            d: 标准化距离
            alpha: 起点相对角度
            beta: 终点相对角度
            
        Returns:
            (t1, p, t2, feasible): 路径参数和可行性
        """
        tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) - math.sin(beta))) / 8
        
        if abs(tmp) > 1:
            return 0, 0, 0, False
        
        p = self.mod2pi(2*math.pi - math.acos(tmp))
        t1 = self.mod2pi(-alpha + math.atan2(math.cos(alpha) - math.cos(beta), 
                                            d - math.sin(alpha) + math.sin(beta)) + p/2)
        t2 = self.mod2pi(beta - alpha - t1 + p)
        
        return t1, p, t2, True
    
    def compute_all_paths(self, start: Tuple[float, float, float], 
                         end: Tuple[float, float, float]) -> Dict[str, Dict]:
        """
        计算所有六种Dubins路径
        
        Args:
            start: 起点 (x, y, theta)
            end: 终点 (x, y, theta)
            
        Returns:
            包含所有路径信息的字典
        """
        d, alpha, beta = self.coordinate_transform(start, end)
        
        paths = {}
        
        # 计算六种路径类型
        path_functions = [
            self.compute_rsr, self.compute_lsl, self.compute_rsl,
            self.compute_lsr, self.compute_rlr, self.compute_lrl
        ]
        
        for i, path_type in enumerate(self.path_types):
            t1, p, t2, feasible = path_functions[i](d, alpha, beta)
            
            if feasible:
                length = (t1 + p + t2) * self.turning_radius
            else:
                length = float('inf')
            
            paths[path_type] = {
                'segments': (t1, p, t2),
                'length': length,
                'feasible': feasible,
                'normalized_params': (d, alpha, beta)
            }
        
        return paths
    
    def find_shortest_path(self, start: Tuple[float, float, float], 
                          end: Tuple[float, float, float]) -> Tuple[str, Dict]:
        """
        找到最短的可行路径
        
        Args:
            start: 起点 (x, y, theta)
            end: 终点 (x, y, theta)
            
        Returns:
            (路径类型, 路径信息)
        """
        d, alpha, beta = self.coordinate_transform(start, end)
        
        shortest_type = 'RSR'
        shortest_length = float('inf')
        
        # 计算RSR路径
        t1, p, t2, feasible = self.compute_rsr(d, alpha, beta)
        if feasible:
            length = (t1 + p + t2) * self.turning_radius
            if length < shortest_length:
                shortest_length = length
                shortest_type = 'RSR'
        
        # 计算LSL路径
        t1, p, t2, feasible = self.compute_lsl(d, alpha, beta)
        if feasible:
            length = (t1 + p + t2) * self.turning_radius
            if length < shortest_length:
                shortest_length = length
                shortest_type = 'LSL'
        
        return shortest_type, {'length': shortest_length}
    
    def generate_path_points(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float], 
                           path_type: str, 
                           num_points: int = 100) -> np.ndarray:
        """
        生成具体的路径点
        
        Args:
            start: 起点 (x, y, theta)
            end: 终点 (x, y, theta)
            path_type: 路径类型
            num_points: 路径点数量
            
        Returns:
            路径点数组 (N, 3)，每行为 (x, y, theta)
        """
        paths = self.compute_all_paths(start, end)
        
        if path_type not in paths:
            raise ValueError(f"未知的路径类型: {path_type}")
        
        path_info = paths[path_type]
        if not path_info['feasible']:
            raise ValueError(f"路径类型 {path_type} 不可行")
        
        t1, p, t2 = path_info['segments']
        
        # 计算每段的点数
        total_length = t1 + p + t2
        n1 = int(num_points * t1 / total_length) if total_length > 0 else 0
        n2 = int(num_points * p / total_length) if total_length > 0 else 0
        n3 = num_points - n1 - n2
        
        points = []
        
        # 根据路径类型生成点
        if path_type == 'RSR':
            points = self._generate_rsr_points(start, end, t1, p, t2, n1, n2, n3)
        elif path_type == 'LSL':
            points = self._generate_lsl_points(start, end, t1, p, t2, n1, n2, n3)
        elif path_type == 'RSL':
            points = self._generate_rsl_points(start, end, t1, p, t2, n1, n2, n3)
        elif path_type == 'LSR':
            points = self._generate_lsr_points(start, end, t1, p, t2, n1, n2, n3)
        elif path_type == 'RLR':
            points = self._generate_rlr_points(start, end, t1, p, t2, n1, n2, n3)
        elif path_type == 'LRL':
            points = self._generate_lrl_points(start, end, t1, p, t2, n1, n2, n3)
        
        return np.array(points)
    
    def _generate_arc_points(self, center: Tuple[float, float], 
                           radius: float, 
                           start_angle: float, 
                           end_angle: float, 
                           num_points: int,
                           clockwise: bool = True) -> List[Tuple[float, float, float]]:
        """生成圆弧点"""
        if num_points <= 0:
            return []
        
        if clockwise:
            angles = np.linspace(start_angle, start_angle - abs(end_angle - start_angle), num_points)
        else:
            angles = np.linspace(start_angle, start_angle + abs(end_angle - start_angle), num_points)
        
        points = []
        for angle in angles:
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            theta = angle - math.pi/2 if clockwise else angle + math.pi/2
            points.append((x, y, theta))
        
        return points
    
    def _generate_line_points(self, start: Tuple[float, float, float], 
                            length: float, 
                            num_points: int) -> List[Tuple[float, float, float]]:
        """生成直线点"""
        if num_points <= 0:
            return []
        
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = start[0] + t * length * math.cos(start[2])
            y = start[1] + t * length * math.sin(start[2])
            theta = start[2]
            points.append((x, y, theta))
        
        return points
    
    def _generate_rsr_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成RSR路径点"""
        # 第一段右转
        center1 = (start[0] - self.turning_radius * math.sin(start[2]),
                  start[1] + self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius, 
                                          start[2] + math.pi/2, 
                                          start[2] + math.pi/2 - t1, n1, True)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段直线
        points2 = self._generate_line_points(seg1_end, p * self.turning_radius, n2)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段右转
        center3 = (seg2_end[0] - self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] + self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] + math.pi/2,
                                          seg2_end[2] + math.pi/2 - t2, n3, True)
        
        return points1 + points2 + points3
    
    def _generate_lsl_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成LSL路径点"""
        # 第一段左转
        center1 = (start[0] + self.turning_radius * math.sin(start[2]),
                  start[1] - self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius,
                                          start[2] - math.pi/2,
                                          start[2] - math.pi/2 + t1, n1, False)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段直线
        points2 = self._generate_line_points(seg1_end, p * self.turning_radius, n2)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段左转
        center3 = (seg2_end[0] + self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] - self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] - math.pi/2,
                                          seg2_end[2] - math.pi/2 + t2, n3, False)
        
        return points1 + points2 + points3
    
    def _generate_rsl_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成RSL路径点"""
        # 第一段右转
        center1 = (start[0] - self.turning_radius * math.sin(start[2]),
                  start[1] + self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius,
                                          start[2] + math.pi/2,
                                          start[2] + math.pi/2 - t1, n1, True)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段直线
        points2 = self._generate_line_points(seg1_end, p * self.turning_radius, n2)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段左转
        center3 = (seg2_end[0] + self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] - self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] - math.pi/2,
                                          seg2_end[2] - math.pi/2 + t2, n3, False)
        
        return points1 + points2 + points3
    
    def _generate_lsr_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成LSR路径点"""
        # 第一段左转
        center1 = (start[0] + self.turning_radius * math.sin(start[2]),
                  start[1] - self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius,
                                          start[2] - math.pi/2,
                                          start[2] - math.pi/2 + t1, n1, False)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段直线
        points2 = self._generate_line_points(seg1_end, p * self.turning_radius, n2)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段右转
        center3 = (seg2_end[0] - self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] + self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] + math.pi/2,
                                          seg2_end[2] + math.pi/2 - t2, n3, True)
        
        return points1 + points2 + points3
    
    def _generate_rlr_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成RLR路径点"""
        # 第一段右转
        center1 = (start[0] - self.turning_radius * math.sin(start[2]),
                  start[1] + self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius,
                                          start[2] + math.pi/2,
                                          start[2] + math.pi/2 - t1, n1, True)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段左转
        center2 = (seg1_end[0] + self.turning_radius * math.sin(seg1_end[2]),
                  seg1_end[1] - self.turning_radius * math.cos(seg1_end[2]))
        points2 = self._generate_arc_points(center2, self.turning_radius,
                                          seg1_end[2] - math.pi/2,
                                          seg1_end[2] - math.pi/2 + p, n2, False)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段右转
        center3 = (seg2_end[0] - self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] + self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] + math.pi/2,
                                          seg2_end[2] + math.pi/2 - t2, n3, True)
        
        return points1 + points2 + points3
    
    def _generate_lrl_points(self, start, end, t1, p, t2, n1, n2, n3):
        """生成LRL路径点"""
        # 第一段左转
        center1 = (start[0] + self.turning_radius * math.sin(start[2]),
                  start[1] - self.turning_radius * math.cos(start[2]))
        points1 = self._generate_arc_points(center1, self.turning_radius,
                                          start[2] - math.pi/2,
                                          start[2] - math.pi/2 + t1, n1, False)
        
        # 计算第一段结束点
        if points1:
            seg1_end = points1[-1]
        else:
            seg1_end = start
        
        # 第二段右转
        center2 = (seg1_end[0] - self.turning_radius * math.sin(seg1_end[2]),
                  seg1_end[1] + self.turning_radius * math.cos(seg1_end[2]))
        points2 = self._generate_arc_points(center2, self.turning_radius,
                                          seg1_end[2] + math.pi/2,
                                          seg1_end[2] + math.pi/2 - p, n2, True)
        
        # 计算第二段结束点
        if points2:
            seg2_end = points2[-1]
        else:
            seg2_end = seg1_end
        
        # 第三段左转
        center3 = (seg2_end[0] + self.turning_radius * math.sin(seg2_end[2]),
                  seg2_end[1] - self.turning_radius * math.cos(seg2_end[2]))
        points3 = self._generate_arc_points(center3, self.turning_radius,
                                          seg2_end[2] - math.pi/2,
                                          seg2_end[2] - math.pi/2 + t2, n3, False)
        
        return points1 + points2 + points3


def test_dubins_path():
    """测试Dubins路径实现"""
    print("=== Dubins路径测试 ===")
    
    # 创建Dubins路径计算器
    dubins = DubinsPath(turning_radius=2.0)
    
    # 测试案例
    start = (0, 0, math.pi/4)
    end = (10, 8, -math.pi/4)
    
    print(f"起点: ({start[0]:.1f}, {start[1]:.1f}, {math.degrees(start[2]):.1f}°)")
    print(f"终点: ({end[0]:.1f}, {end[1]:.1f}, {math.degrees(end[2]):.1f}°)")
    
    # 计算最短路径
    shortest_type, shortest_info = dubins.find_shortest_path(start, end)
    
    print(f"最优路径类型: {shortest_type}")
    print(f"最优路径长度: {shortest_info['length']:.3f}")


if __name__ == "__main__":
    test_dubins_path() 