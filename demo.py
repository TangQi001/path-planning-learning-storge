"""
Dubins路径六种序列演示脚本
作者：AI助手
日期：2025年1月
功能：提供简单易用的Dubins路径演示和教学工具
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleDubinsDemo:
    """简化的Dubins路径演示类"""
    
    def __init__(self, turning_radius: float = 2.0):
        """初始化演示类"""
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
        """计算RSR路径（右转-直行-右转）"""
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
        """计算LSL路径（左转-直行-左转）"""
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
        """计算RSL路径（右转-直行-左转）"""
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
        """计算LSR路径（左转-直行-右转）"""
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
        """计算RLR路径（右转-左转-右转）"""
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
        """计算LRL路径（左转-右转-左转）"""
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
    
    def find_shortest_path(self, start: Tuple[float, float, float], 
                          end: Tuple[float, float, float]) -> Tuple[str, Dict]:
        """找到最短的可行路径"""
        paths = self.compute_all_paths(start, end)
        
        shortest_type = None
        shortest_length = float('inf')
        
        for path_type, path_info in paths.items():
            if path_info['feasible'] and path_info['length'] < shortest_length:
                shortest_length = path_info['length']
                shortest_type = path_type
        
        if shortest_type is None:
            raise ValueError("没有找到可行的Dubins路径")
        
        return shortest_type, paths[shortest_type]
    
    def generate_simple_path(self, start: Tuple[float, float, float], 
                            end: Tuple[float, float, float], 
                            path_type: str, 
                            num_points: int = 100) -> np.ndarray:
        """生成简化的路径点用于可视化"""
        if path_type in ['RSR', 'LSL']:
            # 使用简单的曲线连接
            x_points = np.linspace(start[0], end[0], num_points)
            y_points = []
            
            for i, x in enumerate(x_points):
                t = i / (num_points - 1)
                # 使用正弦函数创建平滑曲线
                if path_type == 'RSR':
                    curve_factor = -0.5
                else:
                    curve_factor = 0.5
                
                y = start[1] + t * (end[1] - start[1]) + curve_factor * math.sin(math.pi * t)
                y_points.append(y)
            
            return np.column_stack([x_points, y_points])
        else:
            # 其他路径类型使用直线连接（简化）
            x_points = np.linspace(start[0], end[0], num_points)
            y_points = np.linspace(start[1], end[1], num_points)
            return np.column_stack([x_points, y_points])
    
    def draw_vehicle(self, ax, pose: Tuple[float, float, float], color: str, label: str = ''):
        """绘制车辆（用三角形表示）"""
        x, y, theta = pose
        
        # 车辆尺寸
        length = 1.0
        width = 0.5
        
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
        triangle = plt.Polygon(vertices, closed=True, 
                             facecolor=color, edgecolor='black', 
                             alpha=0.7, linewidth=2)
        ax.add_patch(triangle)
        
        # 添加标签
        if label:
            ax.annotate(label, (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       fontweight='bold', color=color)
    
    def create_summary_table(self, start: Tuple[float, float, float], 
                           end: Tuple[float, float, float]) -> str:
        """创建路径分析汇总表"""
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
        table += "Dubins路径分析汇总\n"
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
    
    def demo_all_paths(self, start: Tuple[float, float, float], 
                      end: Tuple[float, float, float]):
        """演示所有路径类型"""
        print("Dubins路径六种序列演示")
        print("=" * 50)
        
        # 打印分析表
        table = self.create_summary_table(start, end)
        print(table)
        
        # 计算所有路径
        paths = self.compute_all_paths(start, end)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, path_type in enumerate(self.path_types):
            ax = axes[i]
            path_info = paths[path_type]
            
            # 绘制路径
            if path_info['feasible']:
                path_points = self.generate_simple_path(start, end, path_type)
                ax.plot(path_points[:, 0], path_points[:, 1], 
                       color=self.colors[path_type], linewidth=3, alpha=0.8)
                
                title = f'{path_type}\n长度: {path_info["length"]:.2f}'
            else:
                title = f'{path_type}\n不可行'
                ax.text(0.5, 0.5, '不可行', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14, color='red')
            
            # 绘制起点和终点
            self.draw_vehicle(ax, start, 'green')
            self.draw_vehicle(ax, end, 'red')
            
            # 设置子图属性
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # 设置坐标范围
            margin = 2
            x_min = min(start[0], end[0]) - margin
            x_max = max(start[0], end[0]) + margin
            y_min = min(start[1], end[1]) - margin
            y_max = max(start[1], end[1]) + margin
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Dubins路径六种序列比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """主演示函数"""
    print("欢迎使用Dubins路径六种序列演示程序！")
    print("本程序将演示四个经典测试案例的Dubins路径计算")
    print("按回车键开始演示...")
    input()
    
    # 创建演示对象
    demo = SimpleDubinsDemo(turning_radius=2.0)
    
    # 测试案例
    test_cases = [
        ((0, 0, math.pi/4), (10, 8, -math.pi/4), "标准测试配置"),
        ((0, 0, 0), (6, 6, math.pi), "紧密空间配置"),
        ((0, 0, math.pi/6), (15, 3, -math.pi/3), "长距离配置"),
        ((0, 0, 0), (0, 8, math.pi), "U型转弯配置")
    ]
    
    for i, (start, end, description) in enumerate(test_cases):
        print(f"\n案例 {i+1}: {description}")
        print("-" * 50)
        
        # 演示所有路径
        demo.demo_all_paths(start, end)
        
        if i < len(test_cases) - 1:
            print("按回车键继续下一个案例...")
            input()
    
    print("\n演示完成！感谢使用Dubins路径演示程序。")
    print("更多功能请查看项目的其他模块：")
    print("- 02_代码实现/dubins_path.py: 完整算法实现")
    print("- 03_可视化演示/dubins_visualizer.py: 高级可视化")
    print("- 04_交互式教程/interactive_tutorial.py: 交互式学习")
    print("- 05_测试案例/test_cases.py: 完整测试套件")


if __name__ == "__main__":
    main() 