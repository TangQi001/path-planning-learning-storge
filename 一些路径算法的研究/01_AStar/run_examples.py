
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
A*算法示例运行脚本
统一入口，展示项目所有功能
"""

import os
import sys
import time

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title):
    """打印章节"""
    print(f"\n--- {title} ---")

def wait_for_user():
    """等待用户确认"""
    input("\n按Enter键继续...")

def run_basic_astar():
    """运行基础A*算法"""
    print_section("基础A*算法演示")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '02_代码实现'))
        from astar_basic import AStar, create_test_grid, print_grid_with_path
        
        print("创建测试网格...")
        grid = create_test_grid()
        astar = AStar(grid)
        
        start = (0, 0)
        goal = (9, 9)
        
        print(f"搜索从 {start} 到 {goal} 的路径...")
        
        # 测试不同启发函数
        heuristics = ['euclidean', 'manhattan', 'diagonal']
        
        for heuristic in heuristics:
            print(f"\n使用 {heuristic} 启发函数:")
            start_time = time.time()
            path = astar.find_path(start, goal, heuristic)
            end_time = time.time()
            
            if path:
                cost = astar.get_path_cost(path)
                print(f"  路径长度: {len(path)} 步")
                print(f"  路径代价: {cost:.2f}")
                print(f"  执行时间: {(end_time - start_time)*1000:.2f}ms")
            else:
                print("  未找到路径")
        
        # 显示路径
        if path:
            print_grid_with_path(grid, path)
            
        return True
        
    except Exception as e:
        print(f"运行基础A*算法时出错: {e}")
        return False

def run_visualization():
    """运行可视化演示"""
    print_section("可视化演示")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '03_可视化演示'))
        from astar_visualization import AStarVisualizer, create_test_grids
        
        print("创建可视化演示...")
        grids = create_test_grids()
        
        # 选择复杂场景进行演示
        grid = grids['complex']
        start = (0, 0)
        goal = (9, 9)
        
        print(f"使用复杂场景，从 {start} 到 {goal}")
        
        visualizer = AStarVisualizer(grid, start, goal)
        path, search_steps = visualizer.visualize_search()
        
        if path:
            print(f"✅ 找到路径! 长度: {len(path)} 步")
            print(f"搜索步数: {len(search_steps)}")
        else:
            print("❌ 未找到路径")
            
        return True
        
    except Exception as e:
        print(f"运行可视化演示时出错: {e}")
        print("提示: 确保已安装matplotlib并且支持图形显示")
        return False

def run_3d_demo():
    """运行3D演示"""
    print_section("三维A*算法演示")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '04_3D应用'))
        from astar_3d import AStar3D, create_test_environment, Terrain3DVisualizer
        
        print("创建三维测试环境...")
        grid, height_map = create_test_environment()
        
        start = (1, 1)
        goal = (18, 18)
        
        print(f"3D路径搜索: {start} → {goal}")
        
        astar_3d = AStar3D(grid, height_map)
        
        # 测试不同启发函数
        methods = ['euclidean_3d', 'euclidean_2d']
        
        for method in methods:
            print(f"\n使用 {method} 启发函数:")
            start_time = time.time()
            path = astar_3d.find_path_3d(start, goal, method)
            end_time = time.time()
            
            if path:
                cost = astar_3d.get_path_cost_3d(path)
                heights = [p[2] for p in path]
                height_change = max(heights) - min(heights)
                
                print(f"  路径长度: {len(path)} 步")
                print(f"  路径代价: {cost:.2f}")
                print(f"  高度变化: {height_change:.2f}")
                print(f"  执行时间: {(end_time - start_time)*1000:.2f}ms")
        
        # 可视化（如果支持）
        if path:
            try:
                print("\n生成3D可视化...")
                visualizer = Terrain3DVisualizer(grid, height_map)
                visualizer.visualize_terrain_and_path(start, goal, path)
            except:
                print("3D可视化需要matplotlib的3D支持")
                
        return True
        
    except Exception as e:
        print(f"运行3D演示时出错: {e}")
        return False

def run_performance_test():
    """运行性能测试"""
    print_section("性能测试")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '05_测试数据'))
        from test_scenarios import TestScenarioGenerator, PerformanceTester
        
        sys.path.append(os.path.join(os.path.dirname(__file__), '02_代码实现'))
        from astar_basic import AStar
        
        print("生成测试场景...")
        generator = TestScenarioGenerator()
        scenarios = generator.generate_all_scenarios()
        
        print(f"生成了 {len(scenarios)} 个测试场景")
        
        # 运行性能测试
        print("\n开始性能测试...")
        tester = PerformanceTester()
        
        # 选择几个代表性场景进行测试
        test_scenarios = {}
        for name in ['simple_line', 'simple_obstacle']:
            if name in scenarios:
                test_scenarios[name] = scenarios[name]
        
        for scenario_name, scenario in test_scenarios.items():
            print(f"\n测试场景: {scenario['name']}")
            
            astar = AStar(scenario['grid'])
            start_time = time.time()
            path = astar.find_path(scenario['start'], scenario['goal'])
            end_time = time.time()
            
            if path:
                print(f"  ✅ 成功，路径长度: {len(path)}")
                print(f"  执行时间: {(end_time - start_time)*1000:.2f}ms")
            else:
                print(f"  ❌ 未找到路径")
                
        return True
        
    except Exception as e:
        print(f"运行性能测试时出错: {e}")
        return False

def run_algorithm_comparison():
    """运行算法对比演示"""
    print_section("算法对比演示")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '07_算法对比'))
        from algorithms_comparison import AlgorithmComparator, create_test_grids
        
        print("创建算法对比器...")
        grids = create_test_grids()
        
        # 选择复杂场景进行对比
        grid = grids['complex']
        start = (0, 0)
        goal = (9, 9)
        
        print(f"对比场景: Complex (10x10)")
        print(f"起点: {start}, 终点: {goal}")
        
        comparator = AlgorithmComparator(grid)
        
        # 运行算法对比
        print("\n开始三种算法对比...")
        results = comparator.compare_algorithms(start, goal)
        
        # 显示对比结果
        print("\n=== 算法性能对比 ===")
        algorithms = ['dijkstra', 'greedy', 'astar']
        algorithm_names = ['Dijkstra算法', '贪心最佳优先算法', 'A*算法']
        
        for alg_key, alg_name in zip(algorithms, algorithm_names):
            if alg_key in results:
                result = results[alg_key]
                print(f"\n【{alg_name}】")
                print(f"  成功: {'是' if result['success'] else '否'}")
                print(f"  探索节点: {result['nodes_explored']}")
                print(f"  执行时间: {result['execution_time']*1000:.2f} ms")
                if result['success']:
                    print(f"  路径长度: {len(result['path'])} 步")
                    print(f"  路径代价: {result['path_cost']:.2f}")
        
        # 可视化对比（如果支持）
        try:
            print(f"\n生成可视化对比图...")
            comparator.visualize_comparison(start, goal, results)
        except:
            print("可视化需要matplotlib支持")
            
        return True
        
    except Exception as e:
        print(f"运行算法对比演示时出错: {e}")
        return False

def run_advanced_features():
    """运行高级特性演示"""
    print_section("高级特性演示")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '06_高级特性'))
        from advanced_features import AdvancedAStar
        
        print("创建高级A*算法实例...")
        
        # 创建测试网格
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        
        advanced_astar = AdvancedAStar(grid)
        start = (0, 0)
        goal = (9, 9)
        
        # 测试动态权重A*
        print(f"\n动态权重A*搜索: {start} → {goal}")
        start_time = time.time()
        path = advanced_astar.dynamic_weight_astar(start, goal)
        end_time = time.time()
        
        if path:
            print(f"  原始路径长度: {len(path)}")
            print(f"  执行时间: {(end_time - start_time)*1000:.2f}ms")
            
            # 路径平滑
            print("\n应用路径平滑...")
            smoothed_path = advanced_astar.smooth_path(path)
            print(f"  平滑后路径长度: {len(smoothed_path)}")
            print(f"  压缩比: {len(smoothed_path)/len(path)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"运行高级特性演示时出错: {e}")
        return False

def main():
    """主函数"""
    print_header("A*算法完整演示")
    
    print("本脚本将演示A*算法的所有功能模块：")
    print("1. 基础算法实现")
    print("2. 可视化演示")
    print("3. 三维应用")
    print("4. 性能测试")
    print("5. 高级特性")
    
    print("\n注意：某些功能需要matplotlib支持，请确保已正确安装。")
    
    choice = input("\n是否继续运行完整演示？(y/n): ").lower().strip()
    if choice != 'y':
        print("演示已取消。")
        return
    
    # 运行各个模块
    modules = [
        ("基础A*算法", run_basic_astar),
        ("可视化演示", run_visualization),
        ("算法对比", run_algorithm_comparison),
        ("三维应用", run_3d_demo),
        ("性能测试", run_performance_test),
        ("高级特性", run_advanced_features)
    ]
    
    results = {}
    
    for module_name, module_func in modules:
        print_header(f"运行 {module_name}")
        try:
            success = module_func()
            results[module_name] = success
            
            if success:
                print(f"\n✅ {module_name} 运行成功！")
            else:
                print(f"\n❌ {module_name} 运行失败。")
                
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断了 {module_name} 的执行。")
            results[module_name] = False
            
        except Exception as e:
            print(f"\n❌ {module_name} 出现异常: {e}")
            results[module_name] = False
        
        # 询问是否继续
        if module_name != modules[-1][0]:  # 不是最后一个模块
            continue_choice = input(f"\n继续下一个模块？(y/n): ").lower().strip()
            if continue_choice != 'y':
                break
    
    # 显示总结
    print_header("演示总结")
    print("各模块运行结果：")
    
    for module_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {module_name}: {status}")
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"\n总体成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 恭喜！所有模块都运行成功！")
        print("您可以继续探索各个模块的详细功能。")
    else:
        print("\n💡 部分模块运行失败，建议检查：")
        print("- Python环境和依赖包是否正确安装")
        print("- 图形界面支持是否正常")
        print("- 查看具体错误信息进行调试")
    
    print("\n感谢使用A*算法学习项目！")

if __name__ == "__main__":
    main() 