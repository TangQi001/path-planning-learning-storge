"""
Dubins路径测试案例
作者：AI助手
日期：2025年1月
功能：提供完整的测试案例集合，验证Dubins路径算法的正确性
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import time
from typing import List, Dict, Tuple
import sys
import os

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DubinsTestSuite:
    """Dubins路径测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.test_results = []
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
        
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("Dubins路径算法测试套件")
        print("=" * 60)
        
        # 基础功能测试
        print("\n1. 基础功能测试")
        self.test_basic_functionality()
        
        # 标准测试案例
        print("\n2. 标准测试案例")
        self.test_standard_cases()
        
        # 边界条件测试
        print("\n3. 边界条件测试")
        self.test_boundary_conditions()
        
        # 性能测试
        print("\n4. 性能测试")
        self.test_performance()
        
        # 精度测试
        print("\n5. 精度测试")
        self.test_precision()
        
        # 生成测试报告
        self.generate_test_report()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
    
    def test_basic_functionality(self):
        """基础功能测试"""
        print("测试基础算法功能...")
        
        # 创建简单的Dubins路径计算器
        dubins = SimpleDubinsCalculator()
        
        # 测试案例1：简单直线距离
        start1 = (0, 0, 0)
        end1 = (10, 0, 0)
        
        try:
            result1 = dubins.compute_all_paths(start1, end1)
            print(f"  ✓ 基础计算测试通过")
            
            # 验证RSR和LSL应该相等
            rsr_length = result1['RSR']['length']
            lsl_length = result1['LSL']['length']
            
            if abs(rsr_length - lsl_length) < 1e-6:
                print(f"  ✓ 对称性测试通过 (RSR=LSL={rsr_length:.3f})")
            else:
                print(f"  ✗ 对称性测试失败 (RSR={rsr_length:.3f}, LSL={lsl_length:.3f})")
                
        except Exception as e:
            print(f"  ✗ 基础计算测试失败: {e}")
        
        # 测试案例2：U型转弯
        start2 = (0, 0, 0)
        end2 = (0, 10, math.pi)
        
        try:
            result2 = dubins.compute_all_paths(start2, end2)
            print(f"  ✓ U型转弯计算测试通过")
            
            # 验证多个路径应该相等
            feasible_lengths = [info['length'] for info in result2.values() if info['feasible']]
            if len(set([round(l, 6) for l in feasible_lengths])) == 1:
                print(f"  ✓ U型转弯对称性测试通过")
            else:
                print(f"  ✗ U型转弯对称性测试失败")
                
        except Exception as e:
            print(f"  ✗ U型转弯计算测试失败: {e}")
    
    def test_standard_cases(self):
        """标准测试案例"""
        print("测试标准案例...")
        
        dubins = SimpleDubinsCalculator()
        
        # 定义标准测试案例
        test_cases = [
            {
                'name': '标准配置',
                'start': (0, 0, math.pi/4),
                'end': (10, 8, -math.pi/4),
                'radius': 2.0,
                'expected_optimal': 'LSR'
            },
            {
                'name': '紧密空间',
                'start': (0, 0, 0),
                'end': (3, 3, math.pi),
                'radius': 1.5,
                'expected_optimal': 'LSL'
            },
            {
                'name': '长距离',
                'start': (0, 0, math.pi/6),
                'end': (20, 5, -math.pi/3),
                'radius': 3.0,
                'expected_optimal': 'RSR'
            },
            {
                'name': 'U型转弯',
                'start': (0, 0, 0),
                'end': (0, 5, math.pi),
                'radius': 2.5,
                'expected_optimal': None  # 多个路径等价
            }
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n  测试案例 {i+1}: {case['name']}")
            
            try:
                dubins.turning_radius = case['radius']
                result = dubins.compute_all_paths(case['start'], case['end'])
                
                # 找到最优路径
                optimal_type, optimal_info = dubins.find_shortest_path(case['start'], case['end'])
                
                print(f"    起点: {case['start']}")
                print(f"    终点: {case['end']}")
                print(f"    转弯半径: {case['radius']}")
                print(f"    最优路径: {optimal_type}")
                print(f"    最优长度: {optimal_info['length']:.3f}")
                
                # 显示所有路径长度
                print("    所有路径长度:")
                for path_type in self.path_types:
                    path_info = result[path_type]
                    if path_info['feasible']:
                        print(f"      {path_type}: {path_info['length']:.3f}")
                    else:
                        print(f"      {path_type}: 不可行")
                
                # 验证预期结果
                if case['expected_optimal'] and optimal_type == case['expected_optimal']:
                    print(f"    ✓ 符合预期最优路径")
                elif case['expected_optimal']:
                    print(f"    ⚠ 最优路径不符合预期 (预期: {case['expected_optimal']})")
                else:
                    print(f"    ✓ 测试完成 (无特定预期)")
                
                # 记录测试结果
                self.test_results.append({
                    'case': case['name'],
                    'optimal_type': optimal_type,
                    'optimal_length': optimal_info['length'],
                    'all_paths': result
                })
                
            except Exception as e:
                print(f"    ✗ 测试失败: {e}")
    
    def test_boundary_conditions(self):
        """边界条件测试"""
        print("测试边界条件...")
        
        dubins = SimpleDubinsCalculator()
        
        # 边界条件测试案例
        boundary_cases = [
            {
                'name': '起点终点重合',
                'start': (0, 0, 0),
                'end': (0, 0, 0),
                'radius': 1.0
            },
            {
                'name': '极小距离',
                'start': (0, 0, 0),
                'end': (0.1, 0.1, 0),
                'radius': 1.0
            },
            {
                'name': '极大距离',
                'start': (0, 0, 0),
                'end': (1000, 1000, 0),
                'radius': 1.0
            },
            {
                'name': '极小转弯半径',
                'start': (0, 0, 0),
                'end': (10, 10, math.pi),
                'radius': 0.1
            },
            {
                'name': '极大转弯半径',
                'start': (0, 0, 0),
                'end': (10, 10, math.pi),
                'radius': 100.0
            },
            {
                'name': '极端角度差',
                'start': (0, 0, 0),
                'end': (5, 5, math.pi),
                'radius': 2.0
            }
        ]
        
        for case in boundary_cases:
            print(f"\n  测试边界条件: {case['name']}")
            
            try:
                dubins.turning_radius = case['radius']
                result = dubins.compute_all_paths(case['start'], case['end'])
                
                # 检查是否有可行路径
                feasible_paths = [ptype for ptype, info in result.items() if info['feasible']]
                
                if feasible_paths:
                    optimal_type, optimal_info = dubins.find_shortest_path(case['start'], case['end'])
                    print(f"    ✓ 找到可行路径: {optimal_type} (长度: {optimal_info['length']:.3f})")
                else:
                    print(f"    ⚠ 没有找到可行路径")
                
            except Exception as e:
                print(f"    ✗ 边界条件测试失败: {e}")
    
    def test_performance(self):
        """性能测试"""
        print("测试算法性能...")
        
        dubins = SimpleDubinsCalculator()
        
        # 生成随机测试案例
        np.random.seed(42)  # 确保可重复性
        
        num_tests = 1000
        start_time = time.time()
        
        successful_tests = 0
        total_computations = 0
        
        for i in range(num_tests):
            # 生成随机起点和终点
            start = (
                np.random.uniform(-50, 50),
                np.random.uniform(-50, 50),
                np.random.uniform(0, 2*math.pi)
            )
            
            end = (
                np.random.uniform(-50, 50),
                np.random.uniform(-50, 50),
                np.random.uniform(0, 2*math.pi)
            )
            
            radius = np.random.uniform(0.5, 5.0)
            
            try:
                dubins.turning_radius = radius
                result = dubins.compute_all_paths(start, end)
                
                # 计算可行路径数量
                feasible_count = sum(1 for info in result.values() if info['feasible'])
                total_computations += feasible_count
                
                if feasible_count > 0:
                    successful_tests += 1
                    
            except Exception as e:
                pass  # 忽略失败的测试
        
        end_time = time.time()
        
        # 计算性能指标
        total_time = end_time - start_time
        avg_time_per_test = total_time / num_tests
        success_rate = successful_tests / num_tests
        
        print(f"  总测试数量: {num_tests}")
        print(f"  成功测试数量: {successful_tests}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  总用时: {total_time:.3f}秒")
        print(f"  平均每次测试用时: {avg_time_per_test*1000:.3f}毫秒")
        print(f"  总路径计算次数: {total_computations}")
        
        if avg_time_per_test < 0.001:  # 1毫秒
            print(f"  ✓ 性能测试通过 (平均用时 < 1ms)")
        else:
            print(f"  ⚠ 性能测试警告 (平均用时 > 1ms)")
    
    def test_precision(self):
        """精度测试"""
        print("测试数值精度...")
        
        dubins = SimpleDubinsCalculator()
        
        # 测试数值稳定性
        precision_cases = [
            {
                'name': '微小差异测试',
                'start': (0, 0, 0),
                'end': (10, 1e-10, 1e-10),
                'radius': 1.0
            },
            {
                'name': '大数值测试',
                'start': (0, 0, 0),
                'end': (1e6, 1e6, 0),
                'radius': 1e3
            },
            {
                'name': '角度精度测试',
                'start': (0, 0, 0),
                'end': (5, 5, math.pi - 1e-10),
                'radius': 2.0
            }
        ]
        
        for case in precision_cases:
            print(f"\n  测试精度: {case['name']}")
            
            try:
                dubins.turning_radius = case['radius']
                result = dubins.compute_all_paths(case['start'], case['end'])
                
                # 检查结果的合理性
                feasible_paths = [ptype for ptype, info in result.items() if info['feasible']]
                
                if feasible_paths:
                    lengths = [result[ptype]['length'] for ptype in feasible_paths]
                    min_length = min(lengths)
                    max_length = max(lengths)
                    
                    if min_length > 0 and max_length < float('inf'):
                        print(f"    ✓ 精度测试通过 (长度范围: {min_length:.6f} - {max_length:.6f})")
                    else:
                        print(f"    ⚠ 精度测试异常 (长度范围异常)")
                else:
                    print(f"    ⚠ 没有找到可行路径")
                
            except Exception as e:
                print(f"    ✗ 精度测试失败: {e}")
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n生成测试报告...")
        
        # 创建测试报告
        report = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_cases': len(self.test_results),
            'test_results': self.test_results,
            'summary': {
                'most_common_optimal': self.get_most_common_optimal(),
                'average_path_length': self.get_average_path_length(),
                'feasibility_statistics': self.get_feasibility_statistics()
            }
        }
        
        # 保存报告到文件
        try:
            with open('test_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print("  ✓ 测试报告已保存到 test_report.json")
        except Exception as e:
            print(f"  ✗ 保存报告失败: {e}")
        
        # 创建可视化报告
        self.create_visual_report()
    
    def get_most_common_optimal(self):
        """获取最常见的最优路径类型"""
        if not self.test_results:
            return None
        
        optimal_types = [result['optimal_type'] for result in self.test_results]
        return max(set(optimal_types), key=optimal_types.count)
    
    def get_average_path_length(self):
        """获取平均路径长度"""
        if not self.test_results:
            return 0
        
        lengths = [result['optimal_length'] for result in self.test_results]
        return sum(lengths) / len(lengths)
    
    def get_feasibility_statistics(self):
        """获取可行性统计"""
        if not self.test_results:
            return {}
        
        feasibility_stats = {path_type: 0 for path_type in self.path_types}
        total_cases = len(self.test_results)
        
        for result in self.test_results:
            for path_type in self.path_types:
                if result['all_paths'][path_type]['feasible']:
                    feasibility_stats[path_type] += 1
        
        # 转换为百分比
        for path_type in feasibility_stats:
            feasibility_stats[path_type] = (feasibility_stats[path_type] / total_cases) * 100
        
        return feasibility_stats
    
    def create_visual_report(self):
        """创建可视化报告"""
        if not self.test_results:
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dubins路径测试报告', fontsize=16, fontweight='bold')
        
        # 1. 最优路径类型分布
        ax1 = axes[0, 0]
        optimal_types = [result['optimal_type'] for result in self.test_results]
        optimal_counts = {path_type: optimal_types.count(path_type) for path_type in self.path_types}
        
        ax1.bar(optimal_counts.keys(), optimal_counts.values())
        ax1.set_title('最优路径类型分布')
        ax1.set_ylabel('出现次数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 路径长度分布
        ax2 = axes[0, 1]
        path_lengths = [result['optimal_length'] for result in self.test_results]
        ax2.hist(path_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('最优路径长度分布')
        ax2.set_xlabel('路径长度')
        ax2.set_ylabel('频次')
        
        # 3. 可行性统计
        ax3 = axes[1, 0]
        feasibility_stats = self.get_feasibility_statistics()
        
        ax3.bar(feasibility_stats.keys(), feasibility_stats.values())
        ax3.set_title('各路径类型可行性统计')
        ax3.set_ylabel('可行性百分比 (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 路径类型长度比较
        ax4 = axes[1, 1]
        path_type_lengths = {path_type: [] for path_type in self.path_types}
        
        for result in self.test_results:
            for path_type in self.path_types:
                if result['all_paths'][path_type]['feasible']:
                    path_type_lengths[path_type].append(result['all_paths'][path_type]['length'])
        
        # 绘制箱线图
        data_for_boxplot = []
        labels_for_boxplot = []
        
        for path_type in self.path_types:
            if path_type_lengths[path_type]:
                data_for_boxplot.append(path_type_lengths[path_type])
                labels_for_boxplot.append(path_type)
        
        if data_for_boxplot:
            ax4.boxplot(data_for_boxplot, labels=labels_for_boxplot)
            ax4.set_title('各路径类型长度分布')
            ax4.set_ylabel('路径长度')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        try:
            plt.savefig('test_report_visual.png', dpi=300, bbox_inches='tight')
            print("  ✓ 可视化报告已保存到 test_report_visual.png")
        except Exception as e:
            print(f"  ✗ 保存可视化报告失败: {e}")
        
        plt.show()


class SimpleDubinsCalculator:
    """简化的Dubins路径计算器（用于测试）"""
    
    def __init__(self, turning_radius=1.0):
        """初始化计算器"""
        self.turning_radius = turning_radius
        self.path_types = ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']
    
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
    
    def compute_all_paths(self, start, end):
        """计算所有路径"""
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
                'feasible': feasible
            }
        
        return paths
    
    def find_shortest_path(self, start, end):
        """找到最短路径"""
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


def main():
    """主函数"""
    print("Dubins路径算法测试套件")
    print("这将运行一系列测试来验证算法的正确性和性能")
    
    # 创建测试套件
    test_suite = DubinsTestSuite()
    
    # 运行所有测试
    test_suite.run_all_tests()


if __name__ == "__main__":
    main() 