
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
RRT算法标准测试场景和基准数据

作者: AICP-7协议实现
功能: 提供标准化测试环境和性能基准
特点: 多种难度级别、可重复测试、性能评估
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# 添加代码实现目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_代码实现'))
from rrt_basic import RRTBasic
from rrt_star import RRTStar

@dataclass
class TestScenario:
    """测试场景数据类"""
    name: str
    description: str
    start: Tuple[float, float]
    goal: Tuple[float, float]
    obstacles: List[Tuple[float, float, float]]  # (x, y, radius)
    boundary: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    difficulty: str  # "easy", "medium", "hard", "extreme"
    expected_length: Optional[float] = None
    tags: List[str] = None

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    scenario_name: str
    algorithm: str
    success: bool
    path_length: float
    computation_time: float
    iterations: int
    nodes_generated: int

class ScenarioGenerator:
    """测试场景生成器"""
    
    @staticmethod
    def create_easy_scenarios():
        """创建简单测试场景"""
        scenarios = []
        
        # 场景1: 开阔环境
        scenarios.append(TestScenario(
            name="easy_open_space",
            description="开阔环境，少量障碍物",
            start=(2, 2),
            goal=(18, 18),
            obstacles=[(10, 10, 2)],
            boundary=(0, 20, 0, 20),
            difficulty="easy",
            expected_length=25.5,
            tags=["open_space", "minimal_obstacles"]
        ))
        
        # 场景2: 简单绕行
        scenarios.append(TestScenario(
            name="easy_simple_detour",
            description="需要简单绕行的场景",
            start=(1, 10),
            goal=(19, 10),
            obstacles=[(10, 10, 3)],
            boundary=(0, 20, 0, 20),
            difficulty="easy",
            expected_length=22.0,
            tags=["detour", "single_obstacle"]
        ))
        
        return scenarios
    
    @staticmethod
    def create_medium_scenarios():
        """创建中等难度测试场景"""
        scenarios = []
        
        # 场景1: 多障碍物环境
        scenarios.append(TestScenario(
            name="medium_multi_obstacles",
            description="多个障碍物的复杂环境",
            start=(2, 2),
            goal=(18, 18),
            obstacles=[
                (6, 6, 1.5),
                (10, 4, 1.2),
                (14, 8, 1.8),
                (8, 12, 1.0),
                (15, 15, 1.5)
            ],
            boundary=(0, 20, 0, 20),
            difficulty="medium",
            expected_length=28.0,
            tags=["multi_obstacles", "zigzag_path"]
        ))
        
        # 场景2: 狭窄通道
        scenarios.append(TestScenario(
            name="medium_narrow_passage",
            description="需要通过狭窄通道",
            start=(2, 10),
            goal=(18, 10),
            obstacles=[
                (8, 5, 4),
                (8, 15, 4),
                (12, 7, 2),
                (12, 13, 2)
            ],
            boundary=(0, 20, 0, 20),
            difficulty="medium",
            expected_length=25.0,
            tags=["narrow_passage", "precision_required"]
        ))
        
        return scenarios
    
    @staticmethod
    def create_hard_scenarios():
        """创建困难测试场景"""
        scenarios = []
        
        # 场景1: 迷宫式环境
        scenarios.append(TestScenario(
            name="hard_maze_like",
            description="类似迷宫的复杂环境",
            start=(1, 1),
            goal=(19, 19),
            obstacles=[
                (5, 2, 1), (5, 4, 1), (5, 6, 1), (5, 8, 1),
                (10, 3, 1), (10, 5, 1), (10, 7, 1), (10, 9, 1),
                (15, 4, 1), (15, 6, 1), (15, 8, 1), (15, 10, 1),
                (3, 12, 1), (7, 12, 1), (11, 12, 1), (15, 12, 1),
                (2, 16, 1), (6, 16, 1), (10, 16, 1), (14, 16, 1)
            ],
            boundary=(0, 20, 0, 20),
            difficulty="hard",
            expected_length=35.0,
            tags=["maze", "complex_navigation"]
        ))
        
        # 场景2: 多层障碍
        scenarios.append(TestScenario(
            name="hard_layered_obstacles",
            description="多层次障碍物布局",
            start=(1, 10),
            goal=(19, 10),
            obstacles=[
                (4, 10, 2.5),
                (8, 7, 1.5), (8, 13, 1.5),
                (12, 10, 2.0),
                (16, 6, 1.2), (16, 14, 1.2),
                (6, 4, 1), (6, 16, 1),
                (14, 3, 1), (14, 17, 1)
            ],
            boundary=(0, 20, 0, 20),
            difficulty="hard",
            expected_length=32.0,
            tags=["layered", "strategic_planning"]
        ))
        
        return scenarios
    
    @staticmethod
    def create_extreme_scenarios():
        """创建极端困难测试场景"""
        scenarios = []
        
        # 场景1: 高密度障碍
        scenarios.append(TestScenario(
            name="extreme_dense_obstacles",
            description="高密度障碍物环境",
            start=(1, 1),
            goal=(19, 19),
            obstacles=[
                (3, 3, 0.8), (3, 6, 0.8), (3, 9, 0.8), (3, 12, 0.8), (3, 15, 0.8),
                (6, 2, 0.8), (6, 5, 0.8), (6, 8, 0.8), (6, 11, 0.8), (6, 14, 0.8), (6, 17, 0.8),
                (9, 3, 0.8), (9, 6, 0.8), (9, 9, 0.8), (9, 12, 0.8), (9, 15, 0.8),
                (12, 2, 0.8), (12, 5, 0.8), (12, 8, 0.8), (12, 11, 0.8), (12, 14, 0.8), (12, 17, 0.8),
                (15, 3, 0.8), (15, 6, 0.8), (15, 9, 0.8), (15, 12, 0.8), (15, 15, 0.8),
                (18, 4, 0.8), (18, 7, 0.8), (18, 10, 0.8), (18, 13, 0.8), (18, 16, 0.8)
            ],
            boundary=(0, 20, 0, 20),
            difficulty="extreme",
            expected_length=45.0,
            tags=["dense_obstacles", "high_complexity"]
        ))
        
        return scenarios
    
    @staticmethod
    def get_all_scenarios():
        """获取所有测试场景"""
        all_scenarios = []
        all_scenarios.extend(ScenarioGenerator.create_easy_scenarios())
        all_scenarios.extend(ScenarioGenerator.create_medium_scenarios())
        all_scenarios.extend(ScenarioGenerator.create_hard_scenarios())
        all_scenarios.extend(ScenarioGenerator.create_extreme_scenarios())
        return all_scenarios

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.results = []
    
    def run_single_test(self, scenario: TestScenario, algorithm_class, algorithm_name: str, 
                       max_iter: int = 3000, num_runs: int = 5):
        """运行单个测试场景"""
        print(f"🧪 测试场景: {scenario.name} ({scenario.difficulty})")
        print(f"📋 描述: {scenario.description}")
        
        run_results = []
        
        for run in range(num_runs):
            print(f"  运行 {run + 1}/{num_runs}...", end=" ")
            
            # 创建算法实例
            if algorithm_class == RRTBasic:
                planner = RRTBasic(
                    start=scenario.start,
                    goal=scenario.goal,
                    obstacle_list=scenario.obstacles,
                    boundary=scenario.boundary,
                    max_iter=max_iter
                )
            elif algorithm_class == RRTStar:
                planner = RRTStar(
                    start=scenario.start,
                    goal=scenario.goal,
                    obstacle_list=scenario.obstacles,
                    boundary=scenario.boundary,
                    max_iter=max_iter
                )
            
            # 运行规划
            start_time = time.time()
            path = planner.plan()
            end_time = time.time()
            
            computation_time = end_time - start_time
            
            if path:
                # 计算路径长度
                path_length = sum(
                    np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
                    for i in range(1, len(path))
                )
                
                result = BenchmarkResult(
                    scenario_name=scenario.name,
                    algorithm=algorithm_name,
                    success=True,
                    path_length=path_length,
                    computation_time=computation_time,
                    iterations=len(planner.node_list),
                    nodes_generated=len(planner.node_list)
                )
                print(f"✅ 成功 (长度: {path_length:.2f}, 时间: {computation_time:.3f}s)")
            else:
                result = BenchmarkResult(
                    scenario_name=scenario.name,
                    algorithm=algorithm_name,
                    success=False,
                    path_length=float('inf'),
                    computation_time=computation_time,
                    iterations=len(planner.node_list),
                    nodes_generated=len(planner.node_list)
                )
                print("❌ 失败")
            
            run_results.append(result)
        
        # 统计结果
        successful_runs = [r for r in run_results if r.success]
        success_rate = len(successful_runs) / num_runs
        
        if successful_runs:
            avg_length = np.mean([r.path_length for r in successful_runs])
            avg_time = np.mean([r.computation_time for r in successful_runs])
            std_length = np.std([r.path_length for r in successful_runs])
            print(f"📊 成功率: {success_rate:.1%}, 平均长度: {avg_length:.2f}±{std_length:.2f}, 平均时间: {avg_time:.3f}s")
        else:
            print(f"📊 成功率: {success_rate:.1%}")
        
        self.results.extend(run_results)
        return run_results
    
    def run_full_benchmark(self, scenarios: List[TestScenario] = None):
        """运行完整基准测试"""
        if scenarios is None:
            scenarios = ScenarioGenerator.get_all_scenarios()
        
        print("🏁 开始RRT算法基准测试")
        print("=" * 60)
        
        algorithms = [
            (RRTBasic, "RRT"),
            (RRTStar, "RRT*")
        ]
        
        for algorithm_class, algorithm_name in algorithms:
            print(f"\n🔍 测试算法: {algorithm_name}")
            print("-" * 40)
            
            for scenario in scenarios:
                self.run_single_test(scenario, algorithm_class, algorithm_name)
                print()
    
    def generate_report(self, save_path: str = None):
        """生成测试报告"""
        if not self.results:
            print("❌ 没有测试结果可用于生成报告")
            return
        
        # 按算法和场景分组
        algorithm_stats = {}
        scenario_stats = {}
        
        for result in self.results:
            # 算法统计
            if result.algorithm not in algorithm_stats:
                algorithm_stats[result.algorithm] = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'path_lengths': [],
                    'computation_times': []
                }
            
            stats = algorithm_stats[result.algorithm]
            stats['total_runs'] += 1
            if result.success:
                stats['successful_runs'] += 1
                stats['path_lengths'].append(result.path_length)
                stats['computation_times'].append(result.computation_time)
            
            # 场景统计
            if result.scenario_name not in scenario_stats:
                scenario_stats[result.scenario_name] = {}
            if result.algorithm not in scenario_stats[result.scenario_name]:
                scenario_stats[result.scenario_name][result.algorithm] = {
                    'success_rate': 0,
                    'avg_length': 0,
                    'avg_time': 0
                }
        
        # 计算统计数据
        for algorithm in algorithm_stats:
            stats = algorithm_stats[algorithm]
            stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
            if stats['path_lengths']:
                stats['avg_length'] = np.mean(stats['path_lengths'])
                stats['std_length'] = np.std(stats['path_lengths'])
                stats['avg_time'] = np.mean(stats['computation_times'])
                stats['std_time'] = np.std(stats['computation_times'])
        
        # 生成报告
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_scenarios': len(set(r.scenario_name for r in self.results)),
            'total_runs': len(self.results),
            'algorithm_performance': algorithm_stats,
            'detailed_results': [
                {
                    'scenario': r.scenario_name,
                    'algorithm': r.algorithm,
                    'success': r.success,
                    'path_length': r.path_length if r.success else None,
                    'computation_time': r.computation_time,
                    'nodes_generated': r.nodes_generated
                } for r in self.results
            ]
        }
        
        # 打印报告摘要
        print("\n" + "=" * 60)
        print("📊 基准测试报告摘要")
        print("=" * 60)
        
        for algorithm in algorithm_stats:
            stats = algorithm_stats[algorithm]
            print(f"\n🔍 算法: {algorithm}")
            print(f"  成功率: {stats['success_rate']:.1%} ({stats['successful_runs']}/{stats['total_runs']})")
            if stats.get('avg_length'):
                print(f"  平均路径长度: {stats['avg_length']:.2f} ± {stats['std_length']:.2f}")
                print(f"  平均计算时间: {stats['avg_time']:.3f} ± {stats['std_time']:.3f} 秒")
        
        # 保存详细报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n💾 详细报告已保存到: {save_path}")
        
        return report
    
    def visualize_results(self):
        """可视化测试结果"""
        if not self.results:
            print("❌ 没有测试结果可用于可视化")
            return
        
        # 准备数据
        algorithms = list(set(r.algorithm for r in self.results))
        scenarios = list(set(r.scenario_name for r in self.results))
        
        # 计算成功率
        success_rates = {}
        path_lengths = {}
        
        for algorithm in algorithms:
            success_rates[algorithm] = []
            path_lengths[algorithm] = []
            
            for scenario in scenarios:
                scenario_results = [r for r in self.results 
                                  if r.algorithm == algorithm and r.scenario_name == scenario]
                
                success_rate = sum(1 for r in scenario_results if r.success) / len(scenario_results)
                success_rates[algorithm].append(success_rate)
                
                successful_results = [r for r in scenario_results if r.success]
                if successful_results:
                    avg_length = np.mean([r.path_length for r in successful_results])
                    path_lengths[algorithm].append(avg_length)
                else:
                    path_lengths[algorithm].append(None)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 成功率对比
        x = np.arange(len(scenarios))
        width = 0.35
        
        for i, algorithm in enumerate(algorithms):
            ax1.bar(x + i * width, success_rates[algorithm], width, 
                   label=algorithm, alpha=0.8)
        
        ax1.set_xlabel('测试场景')
        ax1.set_ylabel('成功率')
        ax1.set_title('算法成功率对比')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 路径长度对比
        for i, algorithm in enumerate(algorithms):
            valid_lengths = [l for l in path_lengths[algorithm] if l is not None]
            valid_scenarios = [scenarios[j] for j, l in enumerate(path_lengths[algorithm]) if l is not None]
            
            if valid_lengths:
                ax2.plot(range(len(valid_lengths)), valid_lengths, 
                        marker='o', label=algorithm, linewidth=2, markersize=6)
        
        ax2.set_xlabel('成功的测试场景')
        ax2.set_ylabel('平均路径长度')
        ax2.set_title('路径长度对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demo_benchmark_testing():
    """基准测试演示"""
    print("🧪 RRT算法基准测试演示")
    
    # 创建测试运行器
    runner = BenchmarkRunner()
    
    # 获取简单和中等难度的场景进行快速演示
    easy_scenarios = ScenarioGenerator.create_easy_scenarios()
    medium_scenarios = ScenarioGenerator.create_medium_scenarios()[:1]  # 只选一个中等难度场景
    
    demo_scenarios = easy_scenarios + medium_scenarios
    
    # 运行基准测试
    runner.run_full_benchmark(demo_scenarios)
    
    # 生成报告
    report = runner.generate_report("benchmark_report.json")
    
    # 可视化结果
    runner.visualize_results()
    
    return runner, report

if __name__ == "__main__":
    demo_benchmark_testing() 