"""
Dubins路径交互式教程
作者：AI助手
日期：2025年1月
功能：提供交互式的Dubins路径学习体验，包括参数调整和实时可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import math
import sys
import os

# 添加上级目录到路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveDubinsTutorial:
    """交互式Dubins路径教程类"""
    
    def __init__(self):
        """初始化交互式教程"""
        self.turning_radius = 2.0
        self.start_pos = [0, 0, 0]
        self.end_pos = [10, 8, -math.pi/4]
        self.current_path_type = 'RSR'
        
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
        self.colors = {
            'RSR': '#FF6B6B',  # 红色
            'LSL': '#4ECDC4',  # 青色
            'RSL': '#45B7D1',  # 蓝色
            'LSR': '#96CEB4',  # 绿色
            'RLR': '#FFEAA7',  # 黄色
            'LRL': '#DDA0DD'   # 紫色
        }
        
        self.setup_interface()
    
    def setup_interface(self):
        """设置用户界面"""
        # 创建主图形
        self.fig = plt.figure(figsize=(16, 10))
        
        # 主绘图区域
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        
        # 信息显示区域
        self.ax_info = plt.subplot2grid((3, 4), (0, 3), rowspan=2)
        self.ax_info.axis('off')
        
        # 滑块区域
        self.ax_sliders = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        self.ax_sliders.axis('off')
        
        # 创建滑块
        self.create_sliders()
        
        # 创建按钮
        self.create_buttons()
        
        # 初始绘图
        self.update_plot()
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
    
    def create_sliders(self):
        """创建参数调整滑块"""
        slider_height = 0.03
        slider_width = 0.15
        
        # 起点参数滑块
        self.slider_start_x = Slider(
            plt.axes([0.1, 0.15, slider_width, slider_height]),
            '起点X', -10, 20, valinit=self.start_pos[0]
        )
        
        self.slider_start_y = Slider(
            plt.axes([0.1, 0.10, slider_width, slider_height]),
            '起点Y', -10, 20, valinit=self.start_pos[1]
        )
        
        self.slider_start_theta = Slider(
            plt.axes([0.1, 0.05, slider_width, slider_height]),
            '起点角度', -180, 180, valinit=math.degrees(self.start_pos[2])
        )
        
        # 终点参数滑块
        self.slider_end_x = Slider(
            plt.axes([0.35, 0.15, slider_width, slider_height]),
            '终点X', -10, 20, valinit=self.end_pos[0]
        )
        
        self.slider_end_y = Slider(
            plt.axes([0.35, 0.10, slider_width, slider_height]),
            '终点Y', -10, 20, valinit=self.end_pos[1]
        )
        
        self.slider_end_theta = Slider(
            plt.axes([0.35, 0.05, slider_width, slider_height]),
            '终点角度', -180, 180, valinit=math.degrees(self.end_pos[2])
        )
        
        # 转弯半径滑块
        self.slider_radius = Slider(
            plt.axes([0.6, 0.10, slider_width, slider_height]),
            '转弯半径', 0.5, 5.0, valinit=self.turning_radius
        )
        
        # 连接滑块事件
        self.slider_start_x.on_changed(self.update_parameters)
        self.slider_start_y.on_changed(self.update_parameters)
        self.slider_start_theta.on_changed(self.update_parameters)
        self.slider_end_x.on_changed(self.update_parameters)
        self.slider_end_y.on_changed(self.update_parameters)
        self.slider_end_theta.on_changed(self.update_parameters)
        self.slider_radius.on_changed(self.update_parameters)
    
    def create_buttons(self):
        """创建控制按钮"""
        # 路径类型选择按钮
        self.ax_radio = plt.axes([0.85, 0.05, 0.1, 0.15])
        self.radio_buttons = RadioButtons(self.ax_radio, self.path_types)
        self.radio_buttons.on_clicked(self.change_path_type)
        
        # 预设案例按钮
        button_width = 0.08
        button_height = 0.03
        
        self.btn_case1 = Button(
            plt.axes([0.1, 0.01, button_width, button_height]),
            '标准配置'
        )
        self.btn_case1.on_clicked(lambda x: self.load_preset_case(1))
        
        self.btn_case2 = Button(
            plt.axes([0.2, 0.01, button_width, button_height]),
            '紧密空间'
        )
        self.btn_case2.on_clicked(lambda x: self.load_preset_case(2))
        
        self.btn_case3 = Button(
            plt.axes([0.3, 0.01, button_width, button_height]),
            '长距离'
        )
        self.btn_case3.on_clicked(lambda x: self.load_preset_case(3))
        
        self.btn_case4 = Button(
            plt.axes([0.4, 0.01, button_width, button_height]),
            'U型转弯'
        )
        self.btn_case4.on_clicked(lambda x: self.load_preset_case(4))
        
        # 显示所有路径按钮
        self.btn_show_all = Button(
            plt.axes([0.5, 0.01, button_width, button_height]),
            '显示全部'
        )
        self.btn_show_all.on_clicked(self.show_all_paths)
    
    def update_parameters(self, val):
        """更新参数"""
        self.start_pos = [
            self.slider_start_x.val,
            self.slider_start_y.val,
            math.radians(self.slider_start_theta.val)
        ]
        
        self.end_pos = [
            self.slider_end_x.val,
            self.slider_end_y.val,
            math.radians(self.slider_end_theta.val)
        ]
        
        self.turning_radius = self.slider_radius.val
        
        self.update_plot()
    
    def change_path_type(self, label):
        """改变路径类型"""
        self.current_path_type = label
        self.update_plot()
    
    def load_preset_case(self, case_num):
        """加载预设案例"""
        cases = {
            1: ((0, 0, math.pi/4), (10, 8, -math.pi/4), 2.0),      # 标准配置
            2: ((0, 0, 0), (6, 6, math.pi), 1.5),                  # 紧密空间
            3: ((0, 0, math.pi/6), (15, 3, -math.pi/3), 3.0),     # 长距离
            4: ((0, 0, 0), (0, 8, math.pi), 2.5)                   # U型转弯
        }
        
        if case_num in cases:
            start, end, radius = cases[case_num]
            
            # 更新滑块值
            self.slider_start_x.set_val(start[0])
            self.slider_start_y.set_val(start[1])
            self.slider_start_theta.set_val(math.degrees(start[2]))
            
            self.slider_end_x.set_val(end[0])
            self.slider_end_y.set_val(end[1])
            self.slider_end_theta.set_val(math.degrees(end[2]))
            
            self.slider_radius.set_val(radius)
            
            # 更新参数
            self.start_pos = list(start)
            self.end_pos = list(end)
            self.turning_radius = radius
            
            self.update_plot()
    
    def show_all_paths(self, event):
        """显示所有路径类型"""
        self.plot_all_paths_comparison()
    
    def on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes == self.ax_main:
            # 在主绘图区域点击时，可以移动起点或终点
            if event.button == 1:  # 左键点击移动起点
                self.start_pos[0] = event.xdata
                self.start_pos[1] = event.ydata
                
                self.slider_start_x.set_val(event.xdata)
                self.slider_start_y.set_val(event.ydata)
                
            elif event.button == 3:  # 右键点击移动终点
                self.end_pos[0] = event.xdata
                self.end_pos[1] = event.ydata
                
                self.slider_end_x.set_val(event.xdata)
                self.slider_end_y.set_val(event.ydata)
            
            self.update_plot()
    
    def mod2pi(self, angle):
        """角度标准化"""
        return angle - 2 * math.pi * math.floor(angle / (2 * math.pi))
    
    def coordinate_transform(self, start, end):
        """坐标变换"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        d = math.sqrt(dx*dx + dy*dy) / self.turning_radius
        theta = math.atan2(dy, dx)
        alpha = self.mod2pi(start[2] - theta)
        beta = self.mod2pi(end[2] - theta)
        
        return d, alpha, beta
    
    def compute_path(self, path_type, start, end):
        """计算指定类型的路径"""
        d, alpha, beta = self.coordinate_transform(start, end)
        
        if path_type == 'RSR':
            return self.compute_rsr(d, alpha, beta)
        elif path_type == 'LSL':
            return self.compute_lsl(d, alpha, beta)
        elif path_type == 'RSL':
            return self.compute_rsl(d, alpha, beta)
        elif path_type == 'LSR':
            return self.compute_lsr(d, alpha, beta)
        elif path_type == 'RLR':
            return self.compute_rlr(d, alpha, beta)
        elif path_type == 'LRL':
            return self.compute_lrl(d, alpha, beta)
        
        return 0, 0, 0, False
    
    def compute_rsr(self, d, alpha, beta):
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
    
    def compute_lsl(self, d, alpha, beta):
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
    
    def compute_rsl(self, d, alpha, beta):
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
    
    def compute_lsr(self, d, alpha, beta):
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
    
    def compute_rlr(self, d, alpha, beta):
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
    
    def compute_lrl(self, d, alpha, beta):
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
    
    def generate_path_points(self, start, end, path_type):
        """生成路径点（简化版）"""
        t1, p, t2, feasible = self.compute_path(path_type, start, end)
        
        if not feasible:
            return np.array([]).reshape(0, 2)
        
        # 简化的路径生成
        if path_type in ['RSR', 'LSL']:
            # 使用简单的曲线连接
            num_points = 100
            x_points = np.linspace(start[0], end[0], num_points)
            
            # 添加一些曲率
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            if path_type == 'RSR':
                y_offset = -self.turning_radius
            else:
                y_offset = self.turning_radius
            
            y_points = []
            for i, x in enumerate(x_points):
                t = i / (num_points - 1)
                # 使用二次贝塞尔曲线
                y = (1-t)**2 * start[1] + 2*(1-t)*t * (mid_y + y_offset) + t**2 * end[1]
                y_points.append(y)
            
            return np.column_stack([x_points, y_points])
        
        else:
            # 其他路径类型使用直线连接（简化）
            return np.array([[start[0], start[1]], [end[0], end[1]]])
    
    def draw_vehicle(self, ax, pose, color, label=''):
        """绘制车辆"""
        x, y, theta = pose
        
        # 车辆尺寸
        length = 1.0
        width = 0.5
        
        # 车辆顶点
        vertices = np.array([
            [length, 0],
            [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        # 旋转变换
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        vertices = vertices @ rotation_matrix.T
        vertices[:, 0] += x
        vertices[:, 1] += y
        
        # 绘制三角形
        triangle = patches.Polygon(vertices, closed=True, 
                                 facecolor=color, edgecolor='black', 
                                 alpha=0.7, linewidth=2)
        ax.add_patch(triangle)
        
        if label:
            ax.annotate(label, (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       fontweight='bold', color=color)
    
    def update_plot(self):
        """更新主绘图"""
        self.ax_main.clear()
        
        # 计算当前路径
        t1, p, t2, feasible = self.compute_path(self.current_path_type, self.start_pos, self.end_pos)
        
        if feasible:
            # 绘制路径
            path_points = self.generate_path_points(self.start_pos, self.end_pos, self.current_path_type)
            
            if len(path_points) > 0:
                self.ax_main.plot(path_points[:, 0], path_points[:, 1], 
                                 color=self.colors[self.current_path_type], 
                                 linewidth=3, alpha=0.8,
                                 label=f'{self.current_path_type}路径')
            
            # 计算路径长度
            length = (t1 + p + t2) * self.turning_radius
        else:
            length = float('inf')
            self.ax_main.text(0.5, 0.5, f'{self.current_path_type} 路径不可行', 
                             transform=self.ax_main.transAxes, ha='center', va='center',
                             fontsize=16, color='red', fontweight='bold')
        
        # 绘制起点和终点
        self.draw_vehicle(self.ax_main, self.start_pos, 'green', '起点')
        self.draw_vehicle(self.ax_main, self.end_pos, 'red', '终点')
        
        # 设置图形属性
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        self.ax_main.set_title(f'Dubins路径交互式演示 - {self.current_path_type}', 
                              fontsize=14, fontweight='bold')
        
        # 设置坐标范围
        margin = 3
        x_range = [min(self.start_pos[0], self.end_pos[0]) - margin,
                  max(self.start_pos[0], self.end_pos[0]) + margin]
        y_range = [min(self.start_pos[1], self.end_pos[1]) - margin,
                  max(self.start_pos[1], self.end_pos[1]) + margin]
        
        self.ax_main.set_xlim(x_range)
        self.ax_main.set_ylim(y_range)
        
        # 更新信息显示
        self.update_info_display(t1, p, t2, feasible, length)
        
        self.fig.canvas.draw()
    
    def update_info_display(self, t1, p, t2, feasible, length):
        """更新信息显示"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"""
路径信息
{'='*20}
路径类型: {self.current_path_type}
可行性: {'✓' if feasible else '✗'}

参数设置
{'='*20}
起点: ({self.start_pos[0]:.1f}, {self.start_pos[1]:.1f}, {math.degrees(self.start_pos[2]):.1f}°)
终点: ({self.end_pos[0]:.1f}, {self.end_pos[1]:.1f}, {math.degrees(self.end_pos[2]):.1f}°)
转弯半径: {self.turning_radius:.1f}

路径分析
{'='*20}
"""
        
        if feasible:
            info_text += f"""第一段长度: {t1:.3f}
直线段长度: {p:.3f}
第三段长度: {t2:.3f}
总长度: {length:.3f}

路径说明
{'='*20}
{self.get_path_description(self.current_path_type)}
"""
        else:
            info_text += "路径不可行\n\n"
            info_text += f"约束条件:\n{self.get_feasibility_condition(self.current_path_type)}"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    def get_path_description(self, path_type):
        """获取路径类型说明"""
        descriptions = {
            'RSR': '右转-直行-右转\n使用外公切线连接\n两个右转圆',
            'LSL': '左转-直行-左转\n使用外公切线连接\n两个左转圆',
            'RSL': '右转-直行-左转\n使用内公切线连接\n右转圆和左转圆',
            'LSR': '左转-直行-右转\n使用内公切线连接\n左转圆和右转圆',
            'RLR': '右转-左转-右转\n纯圆弧组合\n无直线段',
            'LRL': '左转-右转-左转\n纯圆弧组合\n无直线段'
        }
        return descriptions.get(path_type, '未知路径类型')
    
    def get_feasibility_condition(self, path_type):
        """获取可行性条件说明"""
        conditions = {
            'RSR': 'RSR路径总是可行',
            'LSL': 'LSL路径总是可行',
            'RSL': '需要 p² ≥ 0\n两圆需要有内公切线',
            'LSR': '需要 p² ≥ 0\n两圆需要有内公切线',
            'RLR': '需要 |tmp| ≤ 1\n距离不能太远',
            'LRL': '需要 |tmp| ≤ 1\n距离不能太远'
        }
        return conditions.get(path_type, '未知约束条件')
    
    def plot_all_paths_comparison(self):
        """显示所有路径类型的比较"""
        # 创建新窗口
        fig_compare = plt.figure(figsize=(15, 10))
        fig_compare.suptitle('Dubins路径六种序列比较', fontsize=16, fontweight='bold')
        
        # 计算所有路径
        all_paths = {}
        for path_type in self.path_types:
            t1, p, t2, feasible = self.compute_path(path_type, self.start_pos, self.end_pos)
            if feasible:
                length = (t1 + p + t2) * self.turning_radius
            else:
                length = float('inf')
            
            all_paths[path_type] = {
                'segments': (t1, p, t2),
                'length': length,
                'feasible': feasible
            }
        
        # 找到最短路径
        shortest_length = min([info['length'] for info in all_paths.values() if info['feasible']])
        
        # 绘制每个子图
        for i, path_type in enumerate(self.path_types):
            ax = plt.subplot(2, 3, i+1)
            path_info = all_paths[path_type]
            
            if path_info['feasible']:
                # 绘制路径
                path_points = self.generate_path_points(self.start_pos, self.end_pos, path_type)
                
                if len(path_points) > 0:
                    ax.plot(path_points[:, 0], path_points[:, 1], 
                           color=self.colors[path_type], linewidth=3)
                
                # 标题
                is_optimal = path_info['length'] == shortest_length
                title = f'{path_type}\n长度: {path_info["length"]:.2f}'
                if is_optimal:
                    title += ' ★'
                    ax.set_facecolor('#f0f8ff')  # 浅蓝色背景表示最优
            else:
                title = f'{path_type}\n不可行'
                ax.text(0.5, 0.5, '不可行', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12, color='red')
            
            # 绘制起点和终点
            self.draw_vehicle(ax, self.start_pos, 'green', size=0.5)
            self.draw_vehicle(ax, self.end_pos, 'red', size=0.5)
            
            # 设置子图属性
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            # 设置相同的坐标范围
            margin = 2
            x_range = [min(self.start_pos[0], self.end_pos[0]) - margin,
                      max(self.start_pos[0], self.end_pos[0]) + margin]
            y_range = [min(self.start_pos[1], self.end_pos[1]) - margin,
                      max(self.start_pos[1], self.end_pos[1]) + margin]
            
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """运行交互式教程"""
        print("=== Dubins路径交互式教程 ===")
        print("操作说明：")
        print("1. 使用滑块调整起点、终点和转弯半径")
        print("2. 点击路径类型按钮切换不同的路径")
        print("3. 左键点击图中位置移动起点")
        print("4. 右键点击图中位置移动终点")
        print("5. 点击预设案例按钮快速切换配置")
        print("6. 点击'显示全部'按钮比较所有路径类型")
        print("\n开始交互式学习...")
        
        plt.show()


def main():
    """主函数"""
    try:
        tutorial = InteractiveDubinsTutorial()
        tutorial.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请确保已安装必要的依赖库：matplotlib, numpy")


if __name__ == "__main__":
    main() 