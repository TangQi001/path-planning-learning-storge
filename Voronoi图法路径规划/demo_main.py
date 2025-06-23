"""
Voronoi图法路径规划 - 主演示程序
================================

本脚本集成展示Voronoi图法路径规划教程的所有核心功能：
1. 基础算法演示
2. 交互式演示启动
3. 性能对比测试
4. 教程使用指南

运行方式：
python demo_main.py

作者：AI教程生成器
日期：2024
"""

import os
import sys
import time
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def print_banner():
    """显示欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                Voronoi图法路径规划教程                         ║
    ║                                                              ║
    ║   基于Voronoi图的路径规划算法完整教程                          ║
    ║   从理论基础到实践应用的全面学习资源                            ║
    ║                                                              ║
    ║   特色功能：                                                  ║
    ║   • 详细理论讲解与数学推导                                     ║
    ║   • 高质量Python代码实现                                      ║
    ║   • 交互式可视化演示                                          ║
    ║   • 与其他算法的全面对比                                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_menu():
    """显示主菜单"""
    menu = """
    📚 教程功能菜单：
    
    1. 🎯 基础算法演示 - 运行核心Voronoi路径规划示例
    2. 🎮 交互式演示 - 启动可视化交互界面
    3. 📊 性能对比测试 - 与其他算法的性能对比
    4. 📖 教程目录结构 - 查看完整教程内容
    5. 🔧 环境检查 - 检查依赖库安装情况
    6. ❌ 退出程序
    
    请选择功能 (1-6): """
    
    return input(menu).strip()

def check_dependencies():
    """检查必要的依赖库"""
    print("🔍 正在检查依赖库...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
        ('networkx', 'NetworkX')
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {display_name} - 已安装")
        except ImportError:
            print(f"  ❌ {display_name} - 未安装")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖库: {', '.join(missing_packages)}")
        print("请运行以下命令安装：")
        print("pip install numpy matplotlib scipy networkx")
        return False
    else:
        print("✅ 所有依赖库已正确安装！")
        return True

def run_basic_demo():
    """运行基础算法演示"""
    print("🎯 启动基础算法演示...")
    
    try:
        # 导入核心模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_path = os.path.join(current_dir, '02_代码实现')
        sys.path.insert(0, core_path)
        from core_voronoi import demo_basic_planning, demo_comparison
        
        print("运行基础路径规划演示...")
        demo_basic_planning()
        
        print("\n运行路径对比演示...")
        demo_comparison()
        
        print("✅ 基础演示完成！")
        
    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        print("请确保所有文件都在正确位置，并且依赖库已正确安装。")

def run_interactive_demo():
    """启动交互式演示"""
    print("🎮 启动交互式演示...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        demo_path = os.path.join(current_dir, '03_可视化演示')
        sys.path.insert(0, demo_path)
        from interactive_demo import InteractiveVoronoiDemo
        
        print("正在启动交互式界面...")
        print("请在弹出的图形窗口中进行交互操作")
        
        demo = InteractiveVoronoiDemo()
        demo.run()
        
        print("✅ 交互式演示结束！")
        
    except Exception as e:
        print(f"❌ 交互式演示启动失败: {e}")
        print("请确保matplotlib支持图形界面显示。")

def run_performance_test():
    """运行性能测试"""
    print("📊 运行性能对比测试...")
    
    try:
        # 简化的性能测试
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_path = os.path.join(current_dir, '02_代码实现')
        sys.path.insert(0, core_path)
        from core_voronoi import VoronoiPathPlanner, Point
        
        print("测试不同障碍物密度下的性能...")
        
        test_scenarios = [
            ("稀疏环境", [(5, 5, 1.5), (15, 10, 2.0)]),
            ("中等密度", [(3, 4, 1), (7, 8, 1.5), (12, 6, 1), (16, 11, 1.2)]),
            ("密集环境", [(2, 3, 0.8), (5, 6, 0.9), (8, 4, 1), (11, 9, 0.8), 
                        (14, 7, 1.1), (17, 12, 0.9)])
        ]
        
        results = []
        
        for scenario_name, obstacles in test_scenarios:
            print(f"\n测试场景: {scenario_name}")
            
            planner = VoronoiPathPlanner(bounds=(0, 0, 20, 15))
            planner.add_obstacles(obstacles)
            
            start_time = time.time()
            planner.construct_voronoi()
            construction_time = time.time() - start_time
            
            start_time = time.time()
            path, distance = planner.plan_path(Point(1, 1), Point(19, 14))
            planning_time = time.time() - start_time
            
            results.append({
                'scenario': scenario_name,
                'obstacles': len(obstacles),
                'construction_time': construction_time * 1000,  # ms
                'planning_time': planning_time * 1000,  # ms
                'path_length': distance if path else float('inf'),
                'success': path is not None
            })
            
            print(f"  障碍物数量: {len(obstacles)}")
            print(f"  Voronoi构造时间: {construction_time*1000:.2f}ms")
            print(f"  路径规划时间: {planning_time*1000:.2f}ms")
            print(f"  路径长度: {distance:.2f}" if path else "  无可行路径")
        
        # 显示汇总结果
        print("\n📊 性能测试汇总:")
        print("-" * 80)
        print(f"{'场景':<12} {'障碍物':<8} {'构造时间(ms)':<12} {'规划时间(ms)':<12} {'路径长度':<10} {'成功率'}")
        print("-" * 80)
        
        for result in results:
            success_rate = "100%" if result['success'] else "0%"
            path_length = f"{result['path_length']:.2f}" if result['success'] else "N/A"
            print(f"{result['scenario']:<12} {result['obstacles']:<8} "
                  f"{result['construction_time']:<12.2f} {result['planning_time']:<12.2f} "
                  f"{path_length:<10} {success_rate}")
        
        print("✅ 性能测试完成！")
        
    except Exception as e:
        print(f"❌ 性能测试出错: {e}")

def show_tutorial_structure():
    """显示教程目录结构"""
    structure = """
    📁 Voronoi图法路径规划教程目录结构：
    
    📂 01_理论基础/
    ├── 📄 README.md - 理论基础概述
    ├── 📄 basic_concepts.md - Voronoi图基本概念
    └── 📄 voronoi_theory.md - 深入数学理论
    
    📂 02_代码实现/
    ├── 📄 implementation_theory.md - 实现理论说明
    ├── 🐍 core_voronoi.py - 核心算法实现
    └── 🐍 path_finder.py - 路径搜索实现
    
    📂 03_可视化演示/
    ├── 📄 visualization_guide.md - 可视化指南
    ├── 🐍 voronoi_visualizer.py - 可视化工具
    └── 🐍 interactive_demo.py - 交互式演示
    
    📂 04_3D应用/
    ├── 📄 3d_extension_theory.md - 3D扩展理论
    └── 🐍 voronoi_3d.py - 3D实现
    
    📂 05_高级特性/
    ├── 📄 advanced_features.md - 高级特性说明
    ├── 🐍 improved_voronoi.py - 改进算法
    └── 🐍 dubins_integration.py - Dubins曲线集成
    
    📂 06_算法对比/
    ├── 📄 comparison.md - 详细算法对比
    └── 🐍 performance_test.py - 性能测试代码
    
    🐍 demo_main.py - 主演示程序（当前文件）
    📄 README.md - 项目主说明文档
    
    💡 建议学习路径：
    1. 先阅读 01_理论基础/ 中的理论文档
    2. 查看 02_代码实现/ 中的核心代码
    3. 运行 03_可视化演示/ 中的可视化程序
    4. 根据需要探索 04_3D应用/ 和 05_高级特性/
    5. 参考 06_算法对比/ 了解算法优势和局限性
    """
    
    print(structure)

def main():
    """主函数"""
    print_banner()
    
    # 首先检查依赖
    if not check_dependencies():
        print("\n请先安装必要的依赖库后再运行程序。")
        return
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_basic_demo()
            
        elif choice == '2':
            run_interactive_demo()
            
        elif choice == '3':
            run_performance_test()
            
        elif choice == '4':
            show_tutorial_structure()
            
        elif choice == '5':
            check_dependencies()
            
        elif choice == '6':
            print("👋 感谢使用Voronoi图法路径规划教程！")
            print("如有问题或建议，欢迎反馈。")
            break
            
        else:
            print("❌ 无效选择，请输入1-6之间的数字。")
        
        # 等待用户按键继续
        input("\n按回车键继续...")

if __name__ == "__main__":
    main() 