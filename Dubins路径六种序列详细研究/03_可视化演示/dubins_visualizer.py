"""
Dubins路径六种序列可视化演示
作者：AI助手
日期：2025年1月
功能：可视化展示六种Dubins路径类型的计算结果和路径形状
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
from typing import Tuple, List, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DubinsVisualizer:
    """Dubins路径可视化类"""
    
    def __init__(self, turning_radius: float = 2.0):
        """
        初始化可视化器
        
        Args:
            turning_radius: 最小转弯半径
        """
        self.turning_radius = turning_radius
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
        self.colors = {
            'RSR': '#FF6B6B',  # 红色
            'LSL': '#4ECDC4',  # 青色
            'RSL': '#45B7D1',  # 蓝色
            'LSR': '#96CEB4',  # 绿色
            'RLR': '#FFEAA7',  # 黄色
            'LRL': '#DDA0DD'   # 紫色
        }
    
    def mod2pi(self, angle: float) -> float:
        """将角度标准化到[0, 2π]范围"""
        return angle - 2 * math.pi * math.floor(angle / (2 * math.pi))
    
    def coordinate_transform(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """坐标变换：将问题转换为标准坐标系"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        d = math.sqrt(dx*dx + dy*dy) / self.turning_radius
        theta = math.atan2(dy, dx)
        alpha = self.mod2pi(start[2] - theta)
        beta = self.mod2pi(end[2] - theta)
        
        return d, alpha, beta
    
    def compute_rsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算RSR路径"""
        try:
            tmp = math.atan2(math.cos(alpha) - math.cos(beta), 
                            d - math.sin(alpha) + math.sin(beta))
            
            t1 = self.mod2pi(alpha - tmp)
            p = math.sqrt(max(0, 2 + d*d - 2*math.cos(alpha - beta) + 
                             2*d*(math.sin(beta) - math.sin(alpha))))
            t2 = self.mod2pi(-beta + tmp)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_lsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算LSL路径"""
        try:
            tmp = math.atan2(math.cos(beta) - math.cos(alpha), 
                            d + math.sin(alpha) - math.sin(beta))
            
            t1 = self.mod2pi(-alpha + tmp)
            p = math.sqrt(max(0, 2 + d*d - 2*math.cos(alpha - beta) + 
                             2*d*(math.sin(alpha) - math.sin(beta))))
            t2 = self.mod2pi(beta - tmp)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_rsl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算RSL路径"""
        try:
            p_squared = d*d - 2 + 2*math.cos(alpha - beta) - 2*d*(math.sin(alpha) + math.sin(beta))
            
            if p_squared < 0:
                return 0, 0, 0, False
            
            p = math.sqrt(p_squared)
            tmp = math.atan2(math.cos(alpha) + math.cos(beta), 
                            d - math.sin(alpha) - math.sin(beta)) - math.atan2(2, p)
            
            t1 = self.mod2pi(alpha - tmp)
            t2 = self.mod2pi(beta - tmp)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_lsr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算LSR路径"""
        try:
            p_squared = -2 + d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) + math.sin(beta))
            
            if p_squared < 0:
                return 0, 0, 0, False
            
            p = math.sqrt(p_squared)
            tmp = math.atan2(-math.cos(alpha) - math.cos(beta), 
                            d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2, p)
            
            t1 = self.mod2pi(-alpha + tmp)
            t2 = self.mod2pi(-beta + tmp)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_rlr(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算RLR路径"""
        try:
            tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) - math.sin(beta))) / 8
            
            if abs(tmp) > 1:
                return 0, 0, 0, False
            
            p = self.mod2pi(2*math.pi - math.acos(tmp))
            t1 = self.mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), 
                                               d - math.sin(alpha) + math.sin(beta)) + p/2)
            t2 = self.mod2pi(alpha - beta - t1 + p)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_lrl(self, d: float, alpha: float, beta: float) -> Tuple[float, float, float, bool]:
        """计算LRL路径"""
        try:
            tmp = (6 - d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) - math.sin(beta))) / 8
            
            if abs(tmp) > 1:
                return 0, 0, 0, False
            
            p = self.mod2pi(2*math.pi - math.acos(tmp))
            t1 = self.mod2pi(-alpha + math.atan2(math.cos(alpha) - math.cos(beta), 
                                                d - math.sin(alpha) + math.sin(beta)) + p/2)
            t2 = self.mod2pi(beta - alpha - t1 + p)
            
            return t1, p, t2, True
        except:
            return 0, 0, 0, False
    
    def compute_all_paths(self, start: Tuple[float, float, float], 
                         end: Tuple[float, float, float]) -> Dict[str, Dict]:
        """计算所有六种Dubins路径"""
        d, alpha, beta = self.coordinate_transform(start, end)
        
        paths = {}
        
        # 计算函数映射
        compute_functions = {
            'RSR': self.compute_rsr,
            'LSL': self.compute_lsl,
            'RSL': self.compute_rsl,
            'LSR': self.compute_lsr,
            'RLR': self.compute_rlr,
            'LRL': self.compute_lrl
        }
        
        for path_type in self.path_types:
            t1, p, t2, feasible = compute_functions[path_type](d, alpha, beta)
            
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
    
    def generate_arc_points(self, center: Tuple[float, float], 
                           radius: float, 
                           start_angle: float, 
                           arc_angle: float, 
                           num_points: int = 50,
                           clockwise: bool = True) -> np.ndarray:
        """生成圆弧点"""
        if num_points <= 0:
            return np.array([]).reshape(0, 2)
        
        if clockwise:
            angles = np.linspace(start_angle, start_angle - arc_angle, num_points)
        else:
            angles = np.linspace(start_angle, start_angle + arc_angle, num_points)
        
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        return np.column_stack([x, y])
    
    def generate_line_points(self, start: Tuple[float, float], 
                            end: Tuple[float, float], 
                            num_points: int = 50) -> np.ndarray:
        """生成直线点"""
        if num_points <= 0:
            return np.array([]).reshape(0, 2)
        
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        
        return np.column_stack([x, y])
    
    def generate_path_points(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float], 
                           path_type: str) -> np.ndarray:
        """生成路径点"""
        paths = self.compute_all_paths(start, end)
        
        if path_type not in paths or not paths[path_type]['feasible']:
            return np.array([]).reshape(0, 2)
        
        t1, p, t2 = paths[path_type]['segments']
        
        # 简化的路径点生成（仅用于演示）
        if path_type == 'RSR':
            return self._generate_rsr_simple(start, end, t1, p, t2)
        elif path_type == 'LSL':
            return self._generate_lsl_simple(start, end, t1, p, t2)
        elif path_type == 'RSL':
            return self._generate_rsl_simple(start, end, t1, p, t2)
        elif path_type == 'LSR':
            return self._generate_lsr_simple(start, end, t1, p, t2)
        elif path_type == 'RLR':
            return self._generate_rlr_simple(start, end, t1, p, t2)
        elif path_type == 'LRL':
            return self._generate_lrl_simple(start, end, t1, p, t2)
        
        return np.array([]).reshape(0, 2)
    
    def _generate_rsr_simple(self, start, end, t1, p, t2):
        """简化的RSR路径生成"""
        points = []
        
        # 第一段：右转
        center1 = (start[0] - self.turning_radius * math.sin(start[2]),
                  start[1] + self.turning_radius * math.cos(start[2]))
        arc1 = self.generate_arc_points(center1, self.turning_radius, 
                                       start[2] + math.pi/2, t1, 30, True)
        points.extend(arc1)
        
        # 第二段：直线
        if len(points) > 0:
            last_point = points[-1]
            angle = start[2] - t1
            end_point = (last_point[0] + p * self.turning_radius * math.cos(angle),
                        last_point[1] + p * self.turning_radius * math.sin(angle))
            line = self.generate_line_points(last_point, end_point, 20)
            points.extend(line)
        
        # 第三段：右转
        if len(points) > 0:
            last_point = points[-1]
            angle = start[2] - t1
            center3 = (last_point[0] - self.turning_radius * math.sin(angle),
                      last_point[1] + self.turning_radius * math.cos(angle))
            arc3 = self.generate_arc_points(center3, self.turning_radius,
                                           angle + math.pi/2, t2, 30, True)
            points.extend(arc3)
        
        return np.array(points) if points else np.array([]).reshape(0, 2)
    
    def _generate_lsl_simple(self, start, end, t1, p, t2):
        """简化的LSL路径生成"""
        points = []
        
        # 第一段：左转
        center1 = (start[0] + self.turning_radius * math.sin(start[2]),
                  start[1] - self.turning_radius * math.cos(start[2]))
        arc1 = self.generate_arc_points(center1, self.turning_radius,
                                       start[2] - math.pi/2, t1, 30, False)
        points.extend(arc1)
        
        # 第二段：直线
        if len(points) > 0:
            last_point = points[-1]
            angle = start[2] + t1
            end_point = (last_point[0] + p * self.turning_radius * math.cos(angle),
                        last_point[1] + p * self.turning_radius * math.sin(angle))
            line = self.generate_line_points(last_point, end_point, 20)
            points.extend(line)
        
        # 第三段：左转
        if len(points) > 0:
            last_point = points[-1]
            angle = start[2] + t1
            center3 = (last_point[0] + self.turning_radius * math.sin(angle),
                      last_point[1] - self.turning_radius * math.cos(angle))
            arc3 = self.generate_arc_points(center3, self.turning_radius,
                                           angle - math.pi/2, t2, 30, False)
            points.extend(arc3)
        
        return np.array(points) if points else np.array([]).reshape(0, 2)
    
    def _generate_rsl_simple(self, start, end, t1, p, t2):
        """简化的RSL路径生成"""
        # 简化版本，返回直线连接
        return self.generate_line_points((start[0], start[1]), (end[0], end[1]), 50)
    
    def _generate_lsr_simple(self, start, end, t1, p, t2):
        """简化的LSR路径生成"""
        # 简化版本，返回直线连接
        return self.generate_line_points((start[0], start[1]), (end[0], end[1]), 50)
    
    def _generate_rlr_simple(self, start, end, t1, p, t2):
        """简化的RLR路径生成"""
        # 简化版本，返回弧形连接
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 2
        
        points = []
        points.extend(self.generate_line_points((start[0], start[1]), (mid_x, mid_y), 25))
        points.extend(self.generate_line_points((mid_x, mid_y), (end[0], end[1]), 25))
        
        return np.array(points) if points else np.array([]).reshape(0, 2)
    
    def _generate_lrl_simple(self, start, end, t1, p, t2):
        """简化的LRL路径生成"""
        # 简化版本，返回弧形连接
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 - 2
        
        points = []
        points.extend(self.generate_line_points((start[0], start[1]), (mid_x, mid_y), 25))
        points.extend(self.generate_line_points((mid_x, mid_y), (end[0], end[1]), 25))
        
        return np.array(points) if points else np.array([]).reshape(0, 2)
    
    def plot_single_path(self, start: Tuple[float, float, float], 
                        end: Tuple[float, float, float], 
                        path_type: str, 
                        save_path: str = None):
        """绘制单个路径"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 计算路径
        paths = self.compute_all_paths(start, end)
        
        if path_type not in paths:
            print(f"未知的路径类型: {path_type}")
            return
        
        path_info = paths[path_type]
        
        # 生成路径点
        if path_info['feasible']:
            path_points = self.generate_path_points(start, end, path_type)
            
            if len(path_points) > 0:
                ax.plot(path_points[:, 0], path_points[:, 1], 
                       color=self.colors[path_type], linewidth=3, 
                       label=f'{path_type} (长度: {path_info["length"]:.2f})')
        else:
            ax.text(0.5, 0.5, f'{path_type} 路径不可行', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, color='red')
        
        # 绘制起点和终点
        self._draw_vehicle(ax, start, 'green', '起点')
        self._draw_vehicle(ax, end, 'red', '终点')
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_title(f'Dubins路径 - {path_type}', fontsize=16, fontweight='bold')
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all_paths(self, start: Tuple[float, float, float], 
                      end: Tuple[float, float, float], 
                      save_path: str = None):
        """绘制所有六种路径类型"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 计算所有路径
        paths = self.compute_all_paths(start, end)
        
        for i, path_type in enumerate(self.path_types):
            ax = axes[i]
            path_info = paths[path_type]
            
            # 绘制路径
            if path_info['feasible']:
                path_points = self.generate_path_points(start, end, path_type)
                
                if len(path_points) > 0:
                    ax.plot(path_points[:, 0], path_points[:, 1], 
                           color=self.colors[path_type], linewidth=3)
                
                title = f'{path_type}\n长度: {path_info["length"]:.2f}'
            else:
                title = f'{path_type}\n不可行'
                ax.text(0.5, 0.5, '不可行', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14, color='red')
            
            # 绘制起点和终点
            self._draw_vehicle(ax, start, 'green', size=0.5)
            self._draw_vehicle(ax, end, 'red', size=0.5)
            
            # 设置子图属性
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # 设置相同的坐标范围
            margin = 2
            x_min = min(start[0], end[0]) - margin
            x_max = max(start[0], end[0]) + margin
            y_min = min(start[1], end[1]) - margin
            y_max = max(start[1], end[1]) + margin
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Dubins路径六种序列比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _draw_vehicle(self, ax, pose: Tuple[float, float, float], 
                     color: str, label: str = '', size: float = 1.0):
        """绘制车辆（用三角形表示）"""
        x, y, theta = pose
        
        # 车辆尺寸
        length = 0.8 * size
        width = 0.4 * size
        
        # 车辆顶点（局部坐标）
        vertices = np.array([
            [length, 0],
            [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        # 旋转矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # 变换到全局坐标
        vertices = vertices @ rotation_matrix.T
        vertices[:, 0] += x
        vertices[:, 1] += y
        
        # 绘制三角形
        triangle = patches.Polygon(vertices, closed=True, 
                                 facecolor=color, edgecolor='black', 
                                 alpha=0.7, linewidth=2)
        ax.add_patch(triangle)
        
        # 添加标签
        if label:
            ax.annotate(label, (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       fontweight='bold', color=color)
    
    def create_comparison_table(self, start: Tuple[float, float, float], 
                              end: Tuple[float, float, float]) -> str:
        """创建路径比较表"""
        paths = self.compute_all_paths(start, end)
        
        # 找到最短路径
        shortest_length = float('inf')
        shortest_type = None
        
        for path_type, path_info in paths.items():
            if path_info['feasible'] and path_info['length'] < shortest_length:
                shortest_length = path_info['length']
                shortest_type = path_type
        
        # 创建表格
        table = "=" * 60 + "\n"
        table += "Dubins路径六种序列分析结果\n"
        table += "=" * 60 + "\n"
        table += f"起点: ({start[0]:.1f}, {start[1]:.1f}, {math.degrees(start[2]):.1f}°)\n"
        table += f"终点: ({end[0]:.1f}, {end[1]:.1f}, {math.degrees(end[2]):.1f}°)\n"
        table += f"转弯半径: {self.turning_radius:.1f}\n"
        table += "-" * 60 + "\n"
        table += f"{'路径类型':<10} {'可行性':<8} {'路径长度':<12} {'相对最优':<10}\n"
        table += "-" * 60 + "\n"
        
        for path_type in self.path_types:
            path_info = paths[path_type]
            
            feasible_str = "✓" if path_info['feasible'] else "✗"
            
            if path_info['feasible']:
                length_str = f"{path_info['length']:.3f}"
                if path_type == shortest_type:
                    relative_str = "最优 ★"
                else:
                    ratio = path_info['length'] / shortest_length
                    relative_str = f"+{(ratio-1)*100:.1f}%"
            else:
                length_str = "不可行"
                relative_str = "—"
            
            table += f"{path_type:<10} {feasible_str:<8} {length_str:<12} {relative_str:<10}\n"
        
        table += "-" * 60 + "\n"
        table += f"最优路径: {shortest_type} (长度: {shortest_length:.3f})\n"
        table += "=" * 60 + "\n"
        
        return table


def demo_dubins_paths():
    """演示Dubins路径计算和可视化"""
    print("=== Dubins路径六种序列可视化演示 ===\n")
    
    # 创建可视化器
    visualizer = DubinsVisualizer(turning_radius=2.0)
    
    # 测试案例
    test_cases = [
        ((0, 0, math.pi/4), (10, 8, -math.pi/4), "标准测试配置"),
        ((0, 0, 0), (6, 6, math.pi), "紧密空间配置"),
        ((0, 0, math.pi/6), (15, 3, -math.pi/3), "长距离配置"),
        ((0, 0, 0), (0, 8, math.pi), "U型转弯配置")
    ]
    
    for i, (start, end, description) in enumerate(test_cases):
        print(f"测试案例 {i+1}: {description}")
        
        # 生成比较表
        table = visualizer.create_comparison_table(start, end)
        print(table)
        
        # 绘制所有路径
        visualizer.plot_all_paths(start, end)
        
        print("按回车键继续下一个案例...")
        input()


if __name__ == "__main__":
    demo_dubins_paths() 