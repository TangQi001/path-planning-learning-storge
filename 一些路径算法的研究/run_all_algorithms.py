
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
路径规划算法集成演示
Path Planning Algorithms Integrated Demo

这个脚本展示了项目中所有路径规划算法的基本功能
This script demonstrates basic functionality of all path planning algorithms in the project
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = ['numpy', 'matplotlib', 'scipy', 'networkx']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的依赖包:")
        print(f"pip install {' '.join(missing_packages)}")
        print("或者运行: pip install -r requirements.txt")
        return False
    return True

def load_algorithm_module(algorithm_path, module_name):
    """动态加载算法模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, algorithm_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"加载模块 {module_name} 失败: {e}")
        return None

def test_astar():
    """测试A*算法"""
    print("\n" + "="*50)
    print("测试 A* 路径规划算法")
    print("="*50)
    
    try:
        astar_path = Path("01_AStar/astar_basic.py")
        if not astar_path.exists():
            print("✗ A*算法文件不存在")
            return
        
        # 临时添加路径
        sys.path.insert(0, str(astar_path.parent))
        astar_module = load_algorithm_module(astar_path, "astar_basic")
        
        if astar_module:
            # 创建简单测试场景
            grid = [[0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0]]
            
            start = (0, 0)
            goal = (4, 4)
            
            planner = astar_module.AStarPlanner()
            path = planner.plan(grid, start, goal)
            
            if path:
                print(f"✓ A*算法测试成功! 找到路径，长度: {len(path)}")
                print(f"路径: {path}")
            else:
                print("✗ A*算法未找到路径")
        
        sys.path.remove(str(astar_path.parent))
        
    except Exception as e:
        print(f"✗ A*算法测试失败: {e}")
        traceback.print_exc()

def test_rrt():
    """测试RRT算法"""
    print("\n" + "="*50)
    print("测试 RRT 路径规划算法")
    print("="*50)
    
    try:
        rrt_path = Path("02_RRT/implementation/rrt_basic.py")
        if not rrt_path.exists():
            print("✗ RRT算法文件不存在")
            return
        
        sys.path.insert(0, str(rrt_path.parent))
        rrt_module = load_algorithm_module(rrt_path, "rrt_basic")
        
        if rrt_module:
            # 创建RRT规划器
            start = [1, 1]
            goal = [9, 9]
            obstacles = [{'center': [5, 5], 'radius': 1.5}]
            
            rrt = rrt_module.RRT(start, goal, obstacles, map_bounds=[0, 10, 0, 10])
            path = rrt.plan(max_iterations=1000)
            
            if path and len(path) > 1:
                print(f"✓ RRT算法测试成功! 找到路径，节点数: {len(path)}")
                print(f"起点: {path[0]}, 终点: {path[-1]}")
            else:
                print("✗ RRT算法未找到路径")
        
        sys.path.remove(str(rrt_path.parent))
        
    except Exception as e:
        print(f"✗ RRT算法测试失败: {e}")
        traceback.print_exc()

def test_bezier():
    """测试Bezier曲线路径规划"""
    print("\n" + "="*50)
    print("测试 Bezier曲线路径规划算法")
    print("="*50)
    
    try:
        bezier_path = Path("03_Bezier/implementation/core_algorithm.py")
        if not bezier_path.exists():
            print("✗ Bezier算法文件不存在")
            return
        
        sys.path.insert(0, str(bezier_path.parent))
        bezier_module = load_algorithm_module(bezier_path, "core_algorithm")
        
        if bezier_module:
            # 创建简单的Bezier曲线
            control_points = [bezier_module.Point(0, 0), bezier_module.Point(2, 4), 
                            bezier_module.Point(6, 4), bezier_module.Point(8, 0)]
            bezier = bezier_module.BezierCurve(control_points)
            
            # 计算曲线上的点
            points = bezier.sample_points(num_points=20)
            length = bezier.arc_length()
            
            print(f"✓ Bezier曲线算法测试成功!")
            print(f"控制点数: {len(control_points)}")
            print(f"曲线长度: {length:.2f}")
            print(f"生成路径点数: {len(points)}")
        
        sys.path.remove(str(bezier_path.parent))
        
    except Exception as e:
        print(f"✗ Bezier算法测试失败: {e}")
        traceback.print_exc()

def test_dubins():
    """测试Dubins路径算法"""
    print("\n" + "="*50)
    print("测试 Dubins路径规划算法")
    print("="*50)
    
    try:
        dubins_path = Path("04_Dubins/implementation/dubins_path.py")
        if not dubins_path.exists():
            print("✗ Dubins算法文件不存在")
            return
        
        sys.path.insert(0, str(dubins_path.parent))
        dubins_module = load_algorithm_module(dubins_path, "dubins_path")
        
        if dubins_module:
            # 创建起点和终点配置 (确保距离足够大于转弯半径)
            start_config = [0, 0, 0]  # [x, y, theta]
            end_config = [10, 5, 1.57]  # [x, y, theta] 
            turning_radius = 2.0
            
            planner = dubins_module.DubinsPath(turning_radius)
            all_paths = planner.compute_all_paths(start_config, end_config)
            shortest_path = planner.find_shortest_path(start_config, end_config)
            
            if shortest_path and len(shortest_path) >= 4 and shortest_path[3]:  # 检查feasible标志
                print(f"✓ Dubins路径算法测试成功!")
                print(f"转弯半径: {turning_radius}")
                print(f"计算出的路径类型数: {len(all_paths)}")
                print(f"最短路径参数: t1={shortest_path[0]:.2f}, p={shortest_path[1]:.2f}, t2={shortest_path[2]:.2f}")
            else:
                print("✗ Dubins算法未找到路径")
        
        sys.path.remove(str(dubins_path.parent))
        
    except Exception as e:
        print(f"✗ Dubins算法测试失败: {e}")
        traceback.print_exc()

def test_voronoi():
    """测试Voronoi图路径规划"""
    print("\n" + "="*50)
    print("测试 Voronoi图路径规划算法")
    print("="*50)
    
    try:
        voronoi_path = Path("05_Voronoi/implementation/core_voronoi.py")
        if not voronoi_path.exists():
            print("✗ Voronoi算法文件不存在")
            return
        
        sys.path.insert(0, str(voronoi_path.parent))
        voronoi_module = load_algorithm_module(voronoi_path, "core_voronoi")
        
        if voronoi_module:
            # 创建简单的障碍物场景 (点障碍物格式)
            obstacles = [[2, 2], [6, 3], [4, 6], [7, 7]]
            start = [0, 0]
            goal = [9, 9]
            bounds = [0, 10, 0, 10]
            
            # 先创建规划器，然后添加障碍物
            planner = voronoi_module.VoronoiPathPlanner([], bounds)
            for obs in obstacles:
                planner.add_obstacle(obs, 0.5)  # 添加半径为0.5的圆形障碍物
            path = planner.plan_path(start, goal)
            
            if path and len(path) > 1:
                print(f"✓ Voronoi图算法测试成功!")
                print(f"路径长度: {len(path)}")
                print(f"起点: {path[0]}, 终点: {path[-1]}")
            else:
                print("✗ Voronoi算法未找到路径")
        
        sys.path.remove(str(voronoi_path.parent))
        
    except Exception as e:
        print(f"✗ Voronoi算法测试失败: {e}")
        traceback.print_exc()

def test_euler_spiral():
    """测试Euler螺旋(缓和曲线)算法"""
    print("\n" + "="*50)
    print("测试 Euler螺旋(缓和曲线)算法")
    print("="*50)
    
    try:
        euler_path = Path("06_EulerSpiral/code/euler_spiral_basic.py")
        if not euler_path.exists():
            print("✗ Euler螺旋算法文件不存在")
            return
        
        sys.path.insert(0, str(euler_path.parent))
        euler_module = load_algorithm_module(euler_path, "euler_spiral_basic")
        
        if euler_module:
            # 测试Euler螺旋计算
            clothoid_param = 1.0  # A参数
            
            spiral = euler_module.EulerSpiral(clothoid_param)
            t, x, y = spiral.compute_coordinates(t_max=3.0, num_points=50)
            
            if len(x) > 0 and len(y) > 0:
                arc_length = spiral.compute_arc_length(3.0)
                curvature_at_end = spiral.compute_curvature(3.0)
                
                print(f"✓ Euler螺旋算法测试成功!")
                print(f"缓和曲线参数A: {clothoid_param}")
                print(f"计算点数: {len(x)}")
                print(f"弧长(t=3.0): {arc_length:.2f}")
                print(f"曲率(t=3.0): {curvature_at_end:.2f}")
            else:
                print("✗ Euler螺旋算法计算失败")
        
        sys.path.remove(str(euler_path.parent))
        
    except Exception as e:
        print(f"✗ Euler螺旋算法测试失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("路径规划算法集成测试")
    print("Path Planning Algorithms Integration Test")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装必要的依赖包，然后重新运行此脚本。")
        return
    
    print("\n依赖检查通过，开始测试各个算法...")
    
    # 测试各个算法
    test_astar()
    test_rrt()
    test_bezier()
    test_dubins()
    test_voronoi()
    test_euler_spiral()
    
    print("\n" + "="*60)
    print("所有算法测试完成!")
    print("="*60)
    
    print("\n使用说明:")
    print("1. 每个算法都可以独立运行")
    print("2. 可视化功能需要matplotlib支持")
    print("3. 详细示例请查看各算法文件夹中的演示脚本")

if __name__ == "__main__":
    main()