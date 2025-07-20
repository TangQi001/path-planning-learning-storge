
# Fix Chinese font display
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

#!/usr/bin/env python3
"""
Bezier曲线路径规划核心算法实现

Author: AI Assistant
Date: 2024
Description: 实现Bezier曲线的路径规划算法，包括控制点生成、曲线计算、约束检查等核心功能
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import math
import warnings

# 可选的依赖项
HAS_NUMPY = False
HAS_MATPLOTLIB = False
HAS_SCIPY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("注意: numpy未安装，将使用Python内置数学函数")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MATPLOTLIB = True
    
    # 配置matplotlib支持中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except ImportError:
    print("注意: matplotlib未安装，可视化功能将不可用")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    print("注意: scipy未安装，优化功能将使用简化版本")

@dataclass
class Point:
    """三维点类"""
    x: float
    y: float
    z: float = 0.0
    
    def __post_init__(self):
        """确保坐标为浮点数"""
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Point':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other: 'Point') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Point') -> 'Point':
        """叉积（三维）"""
        return Point(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        """向量模长"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Point':
        """归一化向量"""
        mag = self.magnitude()
        if mag < 1e-10:
            return Point(0, 0, 0)
        return self / mag
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])

@dataclass
class Constraints:
    """路径约束条件"""
    max_curvature: float = 0.1  # 最大曲率 (1/米)
    max_velocity: float = 10.0  # 最大速度 (米/秒)
    min_segment_length: float = 1.0  # 最小段长度
    obstacle_bounds: List = None  # 障碍物边界
    
    def __post_init__(self):
        if self.obstacle_bounds is None:
            self.obstacle_bounds = []

class BezierCurve:
    """Bezier曲线类"""
    
    def __init__(self, control_points: List[Point]):
        """
        初始化Bezier曲线
        
        Args:
            control_points: 控制点列表
        """
        if len(control_points) < 2:
            raise ValueError("至少需要2个控制点")
        
        self.control_points = control_points
        self.degree = len(control_points) - 1
        self._cached_length = None
    
    def _bernstein_polynomial(self, n: int, i: int, t: float) -> float:
        """计算Bernstein多项式"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def evaluate(self, t: float) -> Point:
        """
        计算曲线上参数t对应的点
        
        Args:
            t: 参数值 [0, 1]
            
        Returns:
            曲线上的点
        """
        if not 0 <= t <= 1:
            warnings.warn(f"参数t={t}超出范围[0,1]")
            t = max(0, min(1, t))
        
        result = Point(0, 0, 0)
        n = self.degree
        
        for i, cp in enumerate(self.control_points):
            b = self._bernstein_polynomial(n, i, t)
            result = result + cp * b
        
        return result
    
    def evaluate_de_casteljau(self, t: float) -> Point:
        """
        使用De Casteljau算法计算曲线点（数值更稳定）
        
        Args:
            t: 参数值 [0, 1]
            
        Returns:
            曲线上的点
        """
        if not 0 <= t <= 1:
            warnings.warn(f"参数t={t}超出范围[0,1]")
            t = max(0, min(1, t))
        
        # 复制控制点
        points = [Point(cp.x, cp.y, cp.z) for cp in self.control_points]
        
        # De Casteljau递归
        for level in range(self.degree):
            for i in range(len(points) - 1):
                points[i] = points[i] * (1 - t) + points[i + 1] * t
            points.pop()
        
        return points[0]
    
    def derivative(self, t: float, order: int = 1) -> Point:
        """
        计算曲线的导数
        
        Args:
            t: 参数值 [0, 1]
            order: 导数阶数
            
        Returns:
            导数向量
        """
        if order == 0:
            return self.evaluate(t)
        
        if order > self.degree:
            return Point(0, 0, 0)
        
        # 计算导数的控制点
        derivative_points = []
        current_points = self.control_points
        
        for _ in range(order):
            new_points = []
            for i in range(len(current_points) - 1):
                diff = current_points[i + 1] - current_points[i]
                new_points.append(diff * (len(current_points) - 1))
            current_points = new_points
        
        if not current_points:
            return Point(0, 0, 0)
        
        # 使用降阶的Bezier曲线计算导数值
        derivative_curve = BezierCurve(current_points)
        return derivative_curve.evaluate(t)
    
    def curvature(self, t: float) -> float:
        """
        计算曲线在参数t处的曲率
        
        Args:
            t: 参数值 [0, 1]
            
        Returns:
            曲率值
        """
        # 计算一阶和二阶导数
        first_deriv = self.derivative(t, 1)
        second_deriv = self.derivative(t, 2)
        
        # 计算速度的模长
        speed = first_deriv.magnitude()
        
        if speed < 1e-10:
            return 0.0  # 速度为零时曲率定义为0
        
        # 对于2D情况，计算曲率
        if abs(first_deriv.z) < 1e-10 and abs(second_deriv.z) < 1e-10:
            # 2D曲率公式: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = abs(first_deriv.x * second_deriv.y - first_deriv.y * second_deriv.x)
            denominator = speed ** 3
            return numerator / denominator if denominator > 1e-10 else 0.0
        else:
            # 3D曲率公式: |v × a| / |v|^3
            cross_product = first_deriv.cross(second_deriv)
            numerator = cross_product.magnitude()
            denominator = speed ** 3
            return numerator / denominator if denominator > 1e-10 else 0.0
    
    def arc_length(self, num_samples: int = 100) -> float:
        """
        计算曲线弧长（数值积分）
        
        Args:
            num_samples: 采样点数量
            
        Returns:
            弧长
        """
        if self._cached_length is not None:
            return self._cached_length
        
        length = 0.0
        dt = 1.0 / num_samples
        
        for i in range(num_samples):
            t = i * dt
            deriv = self.derivative(t, 1)
            length += deriv.magnitude() * dt
        
        self._cached_length = length
        return length
    
    def sample_points(self, num_points: int) -> List[Point]:
        """
        均匀采样曲线上的点
        
        Args:
            num_points: 采样点数量
            
        Returns:
            采样点列表
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            points.append(self.evaluate(t))
        return points
    
    def adaptive_sample(self, tolerance: float = 0.1) -> List[Tuple[float, Point]]:
        """
        自适应采样：在曲率大的地方采样更密集
        
        Args:
            tolerance: 曲率变化容差
            
        Returns:
            (参数值, 点) 的列表
        """
        samples = [(0.0, self.evaluate(0.0))]
        
        def recursive_sample(t_start: float, t_end: float, depth: int = 0):
            if depth > 10:  # 防止过度递归
                return
            
            t_mid = (t_start + t_end) / 2
            curvature_start = self.curvature(t_start)
            curvature_mid = self.curvature(t_mid)
            curvature_end = self.curvature(t_end)
            
            # 检查曲率变化是否超过容差
            if (abs(curvature_mid - curvature_start) > tolerance or 
                abs(curvature_end - curvature_mid) > tolerance):
                recursive_sample(t_start, t_mid, depth + 1)
                samples.append((t_mid, self.evaluate(t_mid)))
                recursive_sample(t_mid, t_end, depth + 1)
        
        recursive_sample(0.0, 1.0)
        samples.append((1.0, self.evaluate(1.0)))
        
        return sorted(samples, key=lambda x: x[0])

class BezierPath:
    """多段Bezier曲线组成的路径"""
    
    def __init__(self, segments: List[BezierCurve]):
        """
        初始化Bezier路径
        
        Args:
            segments: Bezier曲线段列表
        """
        self.segments = segments
        self._total_length = None
        self._segment_lengths = None
    
    def total_length(self) -> float:
        """计算总路径长度"""
        if self._total_length is None:
            self._total_length = sum(segment.arc_length() for segment in self.segments)
        return self._total_length
    
    def segment_lengths(self) -> List[float]:
        """获取各段长度"""
        if self._segment_lengths is None:
            self._segment_lengths = [segment.arc_length() for segment in self.segments]
        return self._segment_lengths
    
    def evaluate_global(self, s: float) -> Point:
        """
        根据弧长参数计算路径上的点
        
        Args:
            s: 弧长参数 [0, total_length]
            
        Returns:
            路径上的点
        """
        if s <= 0:
            return self.segments[0].evaluate(0)
        
        total_len = self.total_length()
        if s >= total_len:
            return self.segments[-1].evaluate(1)
        
        # 找到对应的段
        cumulative_length = 0
        for i, segment in enumerate(self.segments):
            segment_len = segment.arc_length()
            if s <= cumulative_length + segment_len:
                # 在当前段内
                local_s = s - cumulative_length
                local_t = local_s / segment_len if segment_len > 0 else 0
                return segment.evaluate(local_t)
            cumulative_length += segment_len
        
        # 默认返回最后一个点
        return self.segments[-1].evaluate(1)
    
    def check_constraints(self, constraints: Constraints, num_samples: int = 50) -> Tuple[bool, List[str]]:
        """
        检查路径是否满足约束条件
        
        Args:
            constraints: 约束条件
            num_samples: 每段的采样点数
            
        Returns:
            (是否满足约束, 违反约束的描述列表)
        """
        violations = []
        
        for i, segment in enumerate(self.segments):
            # 检查曲率约束
            for j in range(num_samples):
                t = j / (num_samples - 1)
                curvature = segment.curvature(t)
                if curvature > constraints.max_curvature:
                    violations.append(f"段{i}在t={t:.2f}处曲率{curvature:.4f}超过限制{constraints.max_curvature}")
            
            # 检查段长度约束
            length = segment.arc_length()
            if length < constraints.min_segment_length:
                violations.append(f"段{i}长度{length:.2f}小于最小要求{constraints.min_segment_length}")
        
        return len(violations) == 0, violations

class PathPlanner:
    """Bezier路径规划器"""
    
    def __init__(self, constraints: Optional[Constraints] = None):
        """
        初始化路径规划器
        
        Args:
            constraints: 约束条件
        """
        self.constraints = constraints or Constraints()
    
    def generate_control_points_tangent(self, waypoints: List[Point], alpha: float = 0.3) -> List[List[Point]]:
        """
        使用切线方法生成控制点
        
        Args:
            waypoints: 航点列表
            alpha: 控制点距离系数
            
        Returns:
            每段的控制点列表
        """
        if len(waypoints) < 2:
            raise ValueError("至少需要2个航点")
        
        segments_control_points = []
        
        for i in range(len(waypoints) - 1):
            # 当前段的起点和终点
            p_start = waypoints[i]
            p_end = waypoints[i + 1]
            
            # 计算切线方向
            if i == 0:
                # 第一段：使用下一个点的方向
                tangent_start = (waypoints[i + 1] - waypoints[i]).normalize()
            else:
                # 中间段：使用前后点的方向
                tangent_start = (waypoints[i + 1] - waypoints[i - 1]).normalize()
            
            if i == len(waypoints) - 2:
                # 最后一段：使用前一个点的方向
                tangent_end = (waypoints[i + 1] - waypoints[i]).normalize()
            else:
                # 中间段：使用前后点的方向
                tangent_end = (waypoints[i + 2] - waypoints[i]).normalize()
            
            # 计算距离
            distance = p_start.distance_to(p_end)
            
            # 生成控制点
            control_points = [
                p_start,  # P0
                p_start + tangent_start * (alpha * distance),  # P1
                p_end - tangent_end * (alpha * distance),  # P2
                p_end  # P3
            ]
            
            segments_control_points.append(control_points)
        
        return segments_control_points
    
    def generate_smooth_path(self, waypoints: List[Point], method: str = "tangent") -> BezierPath:
        """
        生成平滑路径
        
        Args:
            waypoints: 航点列表
            method: 控制点生成方法 ("tangent", "optimized")
            
        Returns:
            Bezier路径
        """
        if method == "tangent":
            segments_control_points = self.generate_control_points_tangent(waypoints)
        elif method == "optimized":
            segments_control_points = self.optimize_control_points(waypoints)
        else:
            raise ValueError(f"未知的方法: {method}")
        
        # 创建Bezier曲线段
        segments = []
        for control_points in segments_control_points:
            segments.append(BezierCurve(control_points))
        
        return BezierPath(segments)
    
    def optimize_control_points(self, waypoints: List[Point]) -> List[List[Point]]:
        """
        通过优化方法生成控制点
        
        Args:
            waypoints: 航点列表
            
        Returns:
            优化后的控制点列表
        """
        # 首先使用切线方法生成初始解
        initial_control_points = self.generate_control_points_tangent(waypoints, alpha=0.2)
        
        # 展平控制点以便优化
        def pack_variables(segments_cp):
            """将控制点打包为优化变量"""
            variables = []
            for segment_cp in segments_cp:
                for cp in segment_cp[1:3]:  # 只优化中间控制点P1和P2
                    variables.extend([cp.x, cp.y, cp.z])
            return np.array(variables)
        
        def unpack_variables(variables, base_segments_cp):
            """将优化变量解包为控制点"""
            result = []
            var_idx = 0
            for i, base_cp in enumerate(base_segments_cp):
                segment_cp = [base_cp[0]]  # P0保持不变
                
                # P1
                p1 = Point(variables[var_idx], variables[var_idx + 1], variables[var_idx + 2])
                segment_cp.append(p1)
                var_idx += 3
                
                # P2
                p2 = Point(variables[var_idx], variables[var_idx + 1], variables[var_idx + 2])
                segment_cp.append(p2)
                var_idx += 3
                
                segment_cp.append(base_cp[3])  # P3保持不变
                result.append(segment_cp)
            
            return result
        
        def objective_function(variables):
            """优化目标函数"""
            try:
                segments_cp = unpack_variables(variables, initial_control_points)
                
                total_cost = 0
                for segment_cp in segments_cp:
                    curve = BezierCurve(segment_cp)
                    
                    # 路径长度惩罚
                    length = curve.arc_length()
                    total_cost += 0.1 * length
                    
                    # 曲率惩罚
                    curvature_samples = 20
                    for j in range(curvature_samples):
                        t = j / (curvature_samples - 1)
                        curvature = curve.curvature(t)
                        total_cost += 1.0 * curvature ** 2
                        
                        # 约束惩罚
                        if curvature > self.constraints.max_curvature:
                            total_cost += 100 * (curvature - self.constraints.max_curvature) ** 2
                
                return total_cost
            except:
                return 1e6  # 异常情况返回大值
        
        # 执行优化
        x0 = pack_variables(initial_control_points)
        
        try:
            result = minimize(
                objective_function,
                x0,
                method='BFGS',
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success:
                optimized_control_points = unpack_variables(result.x, initial_control_points)
                return optimized_control_points
            else:
                print(f"优化未收敛，使用初始解: {result.message}")
                return initial_control_points
                
        except Exception as e:
            print(f"优化过程出错，使用初始解: {e}")
            return initial_control_points

def demo_bezier_path_planning():
    """演示Bezier路径规划"""
    print("=== Bezier曲线路径规划演示 ===")
    
    # 定义航点
    waypoints = [
        Point(0, 0),
        Point(3, 2),
        Point(7, 1),
        Point(10, 4),
        Point(15, 3)
    ]
    
    print(f"航点数量: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  航点{i}: ({wp.x:.1f}, {wp.y:.1f})")
    
    # 创建约束条件
    constraints = Constraints(
        max_curvature=0.5,
        max_velocity=5.0,
        min_segment_length=0.5
    )
    
    # 创建路径规划器
    planner = PathPlanner(constraints)
    
    # 生成平滑路径（切线方法）
    print("\n生成平滑路径 (切线方法)...")
    path_tangent = planner.generate_smooth_path(waypoints, method="tangent")
    
    # 生成平滑路径（优化方法）
    print("生成平滑路径 (优化方法)...")
    path_optimized = planner.generate_smooth_path(waypoints, method="optimized")
    
    # 输出路径信息
    print(f"\n切线方法路径:")
    print(f"  段数: {len(path_tangent.segments)}")
    print(f"  总长度: {path_tangent.total_length():.2f}")
    
    print(f"\n优化方法路径:")
    print(f"  段数: {len(path_optimized.segments)}")
    print(f"  总长度: {path_optimized.total_length():.2f}")
    
    # 约束检查
    is_valid_tangent, violations_tangent = path_tangent.check_constraints(constraints)
    is_valid_optimized, violations_optimized = path_optimized.check_constraints(constraints)
    
    print(f"\n切线方法约束检查: {'通过' if is_valid_tangent else '失败'}")
    if violations_tangent:
        for violation in violations_tangent[:3]:  # 只显示前3个违反项
            print(f"  - {violation}")
    
    print(f"优化方法约束检查: {'通过' if is_valid_optimized else '失败'}")
    if violations_optimized:
        for violation in violations_optimized[:3]:  # 只显示前3个违反项
            print(f"  - {violation}")
    
    # 可视化（如果有matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 绘制航点
        wp_x = [wp.x for wp in waypoints]
        wp_y = [wp.y for wp in waypoints]
        plt.plot(wp_x, wp_y, 'ro-', markersize=8, linewidth=2, label='原始航点', alpha=0.7)
        
        # 绘制切线方法路径
        tangent_points = []
        for segment in path_tangent.segments:
            points = segment.sample_points(50)
            tangent_points.extend(points)
        
        tx = [p.x for p in tangent_points]
        ty = [p.y for p in tangent_points]
        plt.plot(tx, ty, 'b-', linewidth=2, label='切线方法')
        
        # 绘制优化方法路径
        opt_points = []
        for segment in path_optimized.segments:
            points = segment.sample_points(50)
            opt_points.extend(points)
        
        ox = [p.x for p in opt_points]
        oy = [p.y for p in opt_points]
        plt.plot(ox, oy, 'g--', linewidth=2, label='优化方法')
        
        # 绘制控制点
        for i, segment in enumerate(path_tangent.segments):
            cp = segment.control_points
            cpx = [p.x for p in cp]
            cpy = [p.y for p in cp]
            plt.plot(cpx, cpy, 'b+', markersize=6, alpha=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Bezier曲线路径规划结果')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # 绘制曲率分布
        plt.figure(figsize=(12, 6))
        
        # 计算切线方法的曲率
        t_values = np.linspace(0, 1, 100)
        curvatures_tangent = []
        for segment in path_tangent.segments:
            for t in t_values:
                curvatures_tangent.append(segment.curvature(t))
        
        # 计算优化方法的曲率
        curvatures_opt = []
        for segment in path_optimized.segments:
            for t in t_values:
                curvatures_opt.append(segment.curvature(t))
        
        x_axis = np.arange(len(curvatures_tangent))
        plt.plot(x_axis, curvatures_tangent, 'b-', label='切线方法', linewidth=2)
        plt.plot(x_axis, curvatures_opt, 'g--', label='优化方法', linewidth=2)
        plt.axhline(y=constraints.max_curvature, color='r', linestyle=':', 
                   label=f'最大曲率限制 ({constraints.max_curvature})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('路径曲率分布')
        plt.xlabel('采样点序号')
        plt.ylabel('曲率')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\n注意: 未安装matplotlib，跳过可视化")

if __name__ == "__main__":
    demo_bezier_path_planning() 