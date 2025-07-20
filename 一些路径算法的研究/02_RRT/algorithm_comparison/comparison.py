
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
RRT算法变种综合对比分析

作者: AICP-7协议实现
功能: 全面对比各种RRT算法性能
特点: 多算法评估、性能基准、可视化对比
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from collections import defaultdict

# 添加相关目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_代码实现'))
from rrt_basic import RRTBasic
from rrt_star import RRTStar

class AlgorithmComparison:
    """算法对比分析类"""
    
    def __init__(self):
        self.results = []
    
    def compare_algorithms(self, scenario_config):
        """对比不同算法在给定场景下的性能"""
        start = scenario_config['start']
        goal = scenario_config['goal']
        obstacles = scenario_config['obstacles']
        boundary = scenario_config['boundary']
        
        algorithms = {
            'RRT': RRTBasic,
            'RRT*': RRTStar
        }
        
        results = {}
        
        print("🔄 开始算法性能对比...")
        print("-" * 50)
        
        for alg_name, alg_class in algorithms.items():
            print(f"🧪 测试 {alg_name} 算法...")
            
            # 多次运行取平均
            num_runs = 5
            run_results = []
            
            for run in range(num_runs):
                print(f"  运行 {run+1}/{num_runs}...", end=" ")
                
                # 创建规划器
                if alg_class == RRTBasic:
                    planner = RRTBasic(start, goal, obstacles, boundary, max_iter=2000)
                else:  # RRT*
                    planner = RRTStar(start, goal, obstacles, boundary, max_iter=2000)
                
                # 计时执行
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
                    
                    run_result = {
                        'success': True,
                        'path_length': path_length,
                        'computation_time': computation_time,
                        'nodes_generated': len(planner.node_list),
                        'path': path
                    }
                    print(f"✅ 成功 (长度: {path_length:.2f})")
                else:
                    run_result = {
                        'success': False,
                        'path_length': float('inf'),
                        'computation_time': computation_time,
                        'nodes_generated': len(planner.node_list),
                        'path': None
                    }
                    print("❌ 失败")
                
                run_results.append(run_result)
            
            # 统计结果
            successful_runs = [r for r in run_results if r['success']]
            success_rate = len(successful_runs) / num_runs
            
            if successful_runs:
                avg_length = np.mean([r['path_length'] for r in successful_runs])
                avg_time = np.mean([r['computation_time'] for r in successful_runs])
                avg_nodes = np.mean([r['nodes_generated'] for r in successful_runs])
                std_length = np.std([r['path_length'] for r in successful_runs])
                
                results[alg_name] = {
                    'success_rate': success_rate,
                    'avg_path_length': avg_length,
                    'std_path_length': std_length,
                    'avg_computation_time': avg_time,
                    'avg_nodes_generated': avg_nodes,
                    'best_path': min(successful_runs, key=lambda x: x['path_length'])['path'],
                    'raw_results': run_results
                }
                
                print(f"📊 成功率: {success_rate:.1%}")
                print(f"📏 平均路径长度: {avg_length:.2f} ± {std_length:.2f}")
                print(f"⏱️ 平均计算时间: {avg_time:.3f}s")
                print(f"🌳 平均节点数: {avg_nodes:.0f}")
            else:
                results[alg_name] = {
                    'success_rate': 0,
                    'avg_path_length': float('inf'),
                    'std_path_length': 0,
                    'avg_computation_time': np.mean([r['computation_time'] for r in run_results]),
                    'avg_nodes_generated': np.mean([r['nodes_generated'] for r in run_results]),
                    'best_path': None,
                    'raw_results': run_results
                }
                print(f"📊 成功率: 0%")
            
            print()
        
        return results
    
    def visualize_comparison(self, results, scenario_config):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RRT算法性能对比分析', fontsize=16, fontweight='bold')
        
        algorithms = list(results.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        
        # 1. 成功率对比
        ax1 = axes[0, 0]
        success_rates = [results[alg]['success_rate'] for alg in algorithms]
        bars = ax1.bar(algorithms, success_rates, color=colors[:len(algorithms)])
        ax1.set_title('算法成功率对比')
        ax1.set_ylabel('成功率')
        ax1.set_ylim(0, 1)
        
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. 路径长度对比
        ax2 = axes[0, 1]
        path_lengths = [results[alg]['avg_path_length'] if results[alg]['avg_path_length'] != float('inf') else 0 
                       for alg in algorithms]
        path_stds = [results[alg]['std_path_length'] for alg in algorithms]
        
        ax2.bar(algorithms, path_lengths, yerr=path_stds, color=colors[:len(algorithms)], alpha=0.7)
        ax2.set_title('平均路径长度对比')
        ax2.set_ylabel('路径长度')
        
        # 3. 计算时间对比
        ax3 = axes[0, 2]
        comp_times = [results[alg]['avg_computation_time'] for alg in algorithms]
        ax3.bar(algorithms, comp_times, color=colors[:len(algorithms)], alpha=0.7)
        ax3.set_title('平均计算时间对比')
        ax3.set_ylabel('时间 (秒)')
        
        # 4. 节点数对比
        ax4 = axes[1, 0]
        node_counts = [results[alg]['avg_nodes_generated'] for alg in algorithms]
        ax4.bar(algorithms, node_counts, color=colors[:len(algorithms)], alpha=0.7)
        ax4.set_title('平均生成节点数对比')
        ax4.set_ylabel('节点数')
        
        # 5. 最优路径可视化
        ax5 = axes[1, 1]
        
        # 绘制环境
        for ox, oy, radius in scenario_config['obstacles']:
            circle = plt.Circle((ox, oy), radius, color='red', alpha=0.6)
            ax5.add_patch(circle)
        
        # 绘制最优路径
        for i, alg in enumerate(algorithms):
            if results[alg]['best_path']:
                path = results[alg]['best_path']
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax5.plot(path_x, path_y, color=colors[i], linewidth=3, 
                        label=f'{alg} (长度: {results[alg]["avg_path_length"]:.1f})', alpha=0.8)
        
        ax5.scatter(*scenario_config['start'], color='blue', s=100, marker='o', label='起点')
        ax5.scatter(*scenario_config['goal'], color='green', s=100, marker='*', label='目标')
        ax5.set_xlim(scenario_config['boundary'][0], scenario_config['boundary'][1])
        ax5.set_ylim(scenario_config['boundary'][2], scenario_config['boundary'][3])
        ax5.set_title('最优路径对比')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # 6. 综合性能雷达图
        ax6 = axes[1, 2]
        ax6.remove()
        ax6 = fig.add_subplot(2, 3, 6, projection='polar')
        
        metrics = ['成功率', '路径质量', '计算速度']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for i, alg in enumerate(algorithms):
            # 归一化指标
            success_score = results[alg]['success_rate']
            
            # 路径质量 (效率 = 直线距离 / 实际路径长度)
            direct_distance = np.sqrt(
                (scenario_config['goal'][0] - scenario_config['start'][0])**2 +
                (scenario_config['goal'][1] - scenario_config['start'][1])**2
            )
            if results[alg]['avg_path_length'] != float('inf'):
                path_quality = direct_distance / results[alg]['avg_path_length']
            else:
                path_quality = 0
            
            # 计算速度 (1 - 归一化时间)
            max_time = max(results[a]['avg_computation_time'] for a in algorithms)
            if max_time > 0:
                speed_score = 1 - (results[alg]['avg_computation_time'] / max_time)
            else:
                speed_score = 1
            
            values = [success_score, path_quality, speed_score]
            values += values[:1]  # 闭合
            
            ax6.plot(angles, values, linewidth=2, label=alg, color=colors[i])
            ax6.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合性能雷达图')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, results):
        """生成对比总结报告"""
        print("\n" + "=" * 60)
        print("📊 RRT算法性能对比总结报告")
        print("=" * 60)
        
        algorithms = list(results.keys())
        
        # 成功率排名
        print("\n🎯 成功率排名:")
        success_ranking = sorted(algorithms, key=lambda x: results[x]['success_rate'], reverse=True)
        for i, alg in enumerate(success_ranking, 1):
            print(f"{i}. {alg}: {results[alg]['success_rate']:.1%}")
        
        # 路径质量排名 (长度越短越好)
        print("\n📏 路径质量排名:")
        path_ranking = sorted(
            [alg for alg in algorithms if results[alg]['avg_path_length'] != float('inf')],
            key=lambda x: results[x]['avg_path_length']
        )
        for i, alg in enumerate(path_ranking, 1):
            print(f"{i}. {alg}: {results[alg]['avg_path_length']:.2f}")
        
        # 计算速度排名 (时间越短越好)
        print("\n⏱️ 计算速度排名:")
        speed_ranking = sorted(algorithms, key=lambda x: results[x]['avg_computation_time'])
        for i, alg in enumerate(speed_ranking, 1):
            print(f"{i}. {alg}: {results[alg]['avg_computation_time']:.3f}s")
        
        # 效率排名 (节点数越少越好)
        print("\n🌳 搜索效率排名:")
        efficiency_ranking = sorted(algorithms, key=lambda x: results[x]['avg_nodes_generated'])
        for i, alg in enumerate(efficiency_ranking, 1):
            print(f"{i}. {alg}: {results[alg]['avg_nodes_generated']:.0f}个节点")
        
        # 综合评价
        print("\n🏆 综合评价:")
        for alg in algorithms:
            print(f"\n{alg}:")
            print(f"  优点: ", end="")
            strengths = []
            
            if results[alg]['success_rate'] == max(results[a]['success_rate'] for a in algorithms):
                strengths.append("成功率最高")
            
            if alg in path_ranking[:1] and path_ranking:
                strengths.append("路径质量最优")
            
            if alg in speed_ranking[:1]:
                strengths.append("计算速度最快")
            
            if alg in efficiency_ranking[:1]:
                strengths.append("搜索效率最高")
            
            if strengths:
                print(", ".join(strengths))
            else:
                print("均衡表现")
        
        print("\n📋 推荐使用场景:")
        print("• RRT: 快速原型验证、实时性要求高的场景")
        print("• RRT*: 路径质量要求高、计算资源充足的场景")

def demo_comprehensive_comparison():
    """综合对比演示"""
    print("⚖️ RRT算法综合性能对比演示")
    
    # 测试场景配置
    scenario = {
        'start': (2, 2),
        'goal': (18, 18),
        'obstacles': [
            (6, 6, 1.5),
            (10, 4, 1.2),
            (14, 8, 1.8),
            (8, 12, 1.0),
            (15, 15, 1.5)
        ],
        'boundary': (0, 20, 0, 20)
    }
    
    # 创建对比分析器
    comparator = AlgorithmComparison()
    
    # 执行性能对比
    results = comparator.compare_algorithms(scenario)
    
    # 生成可视化对比
    comparator.visualize_comparison(results, scenario)
    
    # 生成总结报告
    comparator.generate_summary_report(results)
    
    return results

if __name__ == "__main__":
    demo_comprehensive_comparison() 