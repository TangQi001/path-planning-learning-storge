from manim import *
import numpy as np
from scipy.interpolate import BSpline


class BSplineDemo(Scene):
    def construct(self):
        # 设置场景标题
        title = Text("B-样条曲线 (B-Spline Curves)", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 介绍文本
        intro_text = Text(
            "B-样条是一种用于生成平滑曲线的数学工具\n"
            "它通过控制点来定义曲线的形状",
            font_size=24,
            color=WHITE
        )
        intro_text.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(intro_text))
        self.wait(2)
        
        # 清除介绍
        self.play(FadeOut(intro_text))
        
        # 创建控制点
        control_points = [
            np.array([-4, -1, 0]),
            np.array([-2, 2, 0]),
            np.array([0, -2, 0]),
            np.array([2, 1, 0]),
            np.array([4, -1, 0])
        ]
        
        # 显示控制点
        dots = VGroup()
        labels = VGroup()
        for i, point in enumerate(control_points):
            dot = Dot(point, color=RED, radius=0.08)
            label = Text(f"P{i}", font_size=20, color=RED)
            label.next_to(dot, UP, buff=0.1)
            dots.add(dot)
            labels.add(label)
        
        # 显示控制多边形
        control_polygon = Polygon(*control_points, color=GRAY, stroke_width=2)
        control_polygon.set_fill(opacity=0)
        
        self.play(
            AnimationGroup(
                *[GrowFromCenter(dot) for dot in dots],
                lag_ratio=0.2
            )
        )
        self.play(Write(labels))
        self.play(Create(control_polygon))
        
        # 添加说明文本
        control_text = Text("控制点和控制多边形", font_size=20, color=GRAY)
        control_text.to_edge(DOWN)
        self.play(Write(control_text))
        self.wait(2)
        
        # 创建B-spline曲线
        def create_bspline_curve(points, degree=3, num_points=100):
            """创建B-spline曲线"""
            n = len(points)
            # 创建节点向量 (均匀节点向量)
            m = n + degree + 1
            knots = np.concatenate([
                np.zeros(degree),
                np.linspace(0, 1, m - 2 * degree),
                np.ones(degree)
            ])
            
            # 将控制点转换为2D数组（去掉z坐标）
            control_points_2d = np.array([[p[0], p[1]] for p in points])
            
            # 创建B-spline对象
            bspline = BSpline(knots, control_points_2d, degree)
            
            # 生成曲线上的点
            t_values = np.linspace(0, 1, num_points)
            curve_points = bspline(t_values)
            
            # 转换回3D坐标
            curve_points_3d = np.column_stack([curve_points, np.zeros(len(curve_points))])
            
            return curve_points_3d
        
        # 生成B-spline曲线点
        bspline_points = create_bspline_curve(control_points)
        
        # 创建曲线的VMobject
        bspline_curve = VMobject(color=BLUE, stroke_width=4)
        bspline_curve.set_points_as_corners(bspline_points)
        
        # 动画显示B-spline曲线
        self.play(FadeOut(control_text))
        curve_text = Text("B-样条曲线", font_size=20, color=BLUE)
        curve_text.to_edge(DOWN)
        self.play(Write(curve_text))
        self.play(Create(bspline_curve, run_time=3))
        self.wait(2)
        
        # 演示控制点对曲线的影响
        self.play(FadeOut(curve_text))
        interaction_text = Text("移动控制点观察曲线变化", font_size=20, color=YELLOW)
        interaction_text.to_edge(DOWN)
        self.play(Write(interaction_text))
        
        # 创建一个可更新的曲线
        def update_bspline():
            current_points = [dot.get_center() for dot in dots]
            new_bspline_points = create_bspline_curve(current_points)
            new_curve = VMobject(color=BLUE, stroke_width=4)
            new_curve.set_points_as_corners(new_bspline_points)
            return new_curve
        
        # 移动第二个控制点
        new_position1 = np.array([-2, -2, 0])
        self.play(
            dots[1].animate.move_to(new_position1),
            labels[1].animate.next_to(new_position1, UP, buff=0.1),
            Transform(bspline_curve, update_bspline()),
            run_time=2
        )
        self.wait(1)
        
        # 移动第四个控制点
        new_position2 = np.array([2, -1.5, 0])
        self.play(
            dots[3].animate.move_to(new_position2),
            labels[3].animate.next_to(new_position2, UP, buff=0.1),
            Transform(bspline_curve, update_bspline()),
            run_time=2
        )
        self.wait(2)
        
        # 展示B-spline的局部性质
        self.play(FadeOut(interaction_text))
        locality_text = Text("B-样条的局部性：移动一个控制点只影响局部曲线", font_size=18, color=GREEN)
        locality_text.to_edge(DOWN)
        self.play(Write(locality_text))
        
        # 高亮显示受影响的区域
        highlight_region = Rectangle(
            width=3, height=2,
            color=YELLOW, stroke_width=3,
            fill_opacity=0.1
        ).move_to(dots[3].get_center())
        
        self.play(Create(highlight_region))
        
        # 移动控制点并观察局部影响
        final_position = np.array([2, 2.5, 0])
        self.play(
            dots[3].animate.move_to(final_position),
            labels[3].animate.next_to(final_position, UP, buff=0.1),
            Transform(bspline_curve, update_bspline()),
            run_time=2
        )
        self.wait(2)
        
        # 显示数学公式
        self.play(FadeOut(locality_text), FadeOut(highlight_region))
        formula_text = Text("B-样条曲线公式", font_size=24, color=BLUE)
        formula_text.to_edge(DOWN, buff=1.5)
        
        # B-spline的数学公式
        formula = MathTex(
            r"C(t) = \sum_{i=0}^{n} P_i \cdot N_{i,k}(t)",
            font_size=36,
            color=WHITE
        )
        formula.next_to(formula_text, UP, buff=0.3)
        
        explanation = Text(
            "其中 P_i 是控制点，N_{i,k}(t) 是B-样条基函数",
            font_size=18,
            color=GRAY
        )
        explanation.next_to(formula, DOWN, buff=0.3)
        
        self.play(Write(formula_text))
        self.play(Write(formula))
        self.play(Write(explanation))
        self.wait(3)
        
        # 结束动画
        self.play(
            FadeOut(VGroup(
                title, dots, labels, control_polygon, bspline_curve,
                formula_text, formula, explanation
            ))
        )
        
        # 结束文本
        end_text = Text(
            "B-样条曲线广泛应用于\n计算机图形学、CAD设计和动画制作",
            font_size=28,
            color=BLUE
        )
        self.play(Write(end_text))
        self.wait(3)
        self.play(FadeOut(end_text))


class BSplineBasicFunctions(Scene):
    """演示B-样条基函数"""
    
    def construct(self):
        title = Text("B-样条基函数", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # 创建坐标系
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": GRAY},
            tips=False
        )
        axes.center()
        
        # 添加坐标轴标签
        x_label = axes.get_x_axis_label("t")
        y_label = axes.get_y_axis_label("N(t)")
        
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        
        # 定义节点向量
        knots = [0, 0, 0, 1, 2, 3, 3, 3]  # 三次B-样条的节点向量
        degree = 2  # 二次B-样条便于演示
        
        # 绘制基函数
        colors = [RED, GREEN, BLUE, ORANGE, PURPLE]
        
        def basis_function(i, k, t, knots):
            """计算B-样条基函数"""
            if k == 0:
                return 1.0 if knots[i] <= t < knots[i+1] else 0.0
            else:
                c1 = 0.0
                if knots[i+k] != knots[i]:
                    c1 = (t - knots[i]) / (knots[i+k] - knots[i]) * basis_function(i, k-1, t, knots)
                
                c2 = 0.0
                if knots[i+k+1] != knots[i+1]:
                    c2 = (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * basis_function(i+1, k-1, t, knots)
                
                return c1 + c2
        
        # 为每个基函数创建图形
        basis_graphs = VGroup()
        for i in range(len(knots) - degree - 1):
            t_values = np.linspace(0, 3, 300)
            y_values = [basis_function(i, degree, t, knots) for t in t_values]
            
            # 过滤掉无效值
            valid_points = [(t, y) for t, y in zip(t_values, y_values) if not np.isnan(y) and y > 1e-10]
            
            if valid_points:
                points_3d = [axes.coords_to_point(t, y) for t, y in valid_points]
                graph = VMobject(color=colors[i % len(colors)], stroke_width=3)
                graph.set_points_as_corners(points_3d)
                basis_graphs.add(graph)
        
        # 动画显示基函数
        self.play(
            AnimationGroup(
                *[Create(graph) for graph in basis_graphs],
                lag_ratio=0.5
            ),
            run_time=4
        )
        
        # 添加说明
        explanation = Text(
            "每个基函数只在局部区间内非零\n这就是B-样条局部性质的来源",
            font_size=20,
            color=WHITE
        )
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(3)
        
        self.play(FadeOut(VGroup(
            title, axes, x_label, y_label, basis_graphs, explanation
        )))


class BSplineInteractive(Scene):
    """交互式B-样条演示"""
    
    def construct(self):
        title = Text("交互式B-样条演示", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        
        # 创建初始控制点
        initial_points = [
            np.array([-5, -1, 0]),
            np.array([-2, 1, 0]),
            np.array([0, -1, 0]),
            np.array([2, 2, 0]),
            np.array([5, 0, 0])
        ]
        
        # 创建ValueTracker来控制动画
        tracker = ValueTracker(0)
        
        def get_animated_points(t):
            """根据时间参数返回动画化的控制点"""
            animated_points = []
            for i, point in enumerate(initial_points):
                # 为每个点添加不同的动画效果
                offset = np.array([
                    0.5 * np.sin(t + i * PI/2),
                    0.3 * np.cos(t * 1.5 + i * PI/3),
                    0
                ])
                animated_points.append(point + offset)
            return animated_points
        
        def create_animated_bspline(t):
            """创建动画化的B-样条曲线"""
            points = get_animated_points(t)
            
            # 简化的B-样条实现（使用贝塞尔曲线近似）
            curve = VMobject(color=BLUE, stroke_width=4)
            
            # 使用分段贝塞尔曲线来近似B-样条
            if len(points) >= 4:
                # 创建平滑的曲线路径
                smooth_points = []
                for i in range(len(points) - 1):
                    p0 = points[max(0, i-1)]
                    p1 = points[i]
                    p2 = points[i+1]
                    p3 = points[min(len(points)-1, i+2)]
                    
                    # 使用Catmull-Rom样条的控制点
                    for t_seg in np.linspace(0, 1, 20):
                        point = (
                            0.5 * (2 * p1 +
                                  (-p0 + p2) * t_seg +
                                  (2*p0 - 5*p1 + 4*p2 - p3) * t_seg**2 +
                                  (-p0 + 3*p1 - 3*p2 + p3) * t_seg**3)
                        )
                        smooth_points.append(point)
                
                curve.set_points_as_corners(smooth_points)
            
            return curve
        
        # 创建控制点的可视化
        dots = always_redraw(lambda: VGroup(*[
            Dot(point, color=RED, radius=0.06)
            for point in get_animated_points(tracker.get_value())
        ]))
        
        # 创建动画曲线
        animated_curve = always_redraw(lambda: create_animated_bspline(tracker.get_value()))
        
        self.add(dots, animated_curve)
        
        # 添加说明文本
        instruction = Text(
            "观察控制点的动态变化如何影响B-样条曲线",
            font_size=20,
            color=WHITE
        )
        instruction.to_edge(DOWN)
        self.play(Write(instruction))
        
        # 运行动画
        self.play(
            tracker.animate.set_value(4 * PI),
            run_time=8,
            rate_func=linear
        )
        
        self.wait(1)
        self.play(FadeOut(VGroup(title, dots, animated_curve, instruction)))


# 主场景组合
class BSplineComplete(Scene):
    def construct(self):
        # 运行所有演示
        demo1 = BSplineDemo()
        demo1.construct()
        
        self.wait(2)
        
        demo2 = BSplineBasicFunctions()
        demo2.construct()
        
        self.wait(2)
        
        demo3 = BSplineInteractive()
        demo3.construct()


if __name__ == "__main__":
    # 可以单独运行各个场景
    pass 