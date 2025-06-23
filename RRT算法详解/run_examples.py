#!/usr/bin/env python3
"""
RRT算法学习 - 快速开始示例

作者: AICP-7协议实现
功能: 一键运行所有核心演示
使用: python run_examples.py
"""

import sys
import os
import traceback

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '02_代码实现'))
sys.path.append(os.path.join(os.path.dirname(__file__), '03_可视化演示'))

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def run_basic_rrt_demo():
    """运行基础RRT演示"""
    print_header("🚀 基础RRT算法演示")
    
    try:
        from rrt_basic import demo_basic_rrt
        path = demo_basic_rrt()
        if path:
            print(f"✅ 基础RRT演示完成，找到路径包含 {len(path)} 个点")
        else:
            print("⚠️  基础RRT演示完成，但未找到路径")
        return True
    except Exception as e:
        print(f"❌ 基础RRT演示失败: {e}")
        traceback.print_exc()
        return False

def run_rrt_star_demo():
    """运行RRT*演示"""
    print_header("🌟 RRT*算法演示")
    
    try:
        from rrt_star import demo_rrt_star
        result = demo_rrt_star()
        if result[0]:
            print(f"✅ RRT*演示完成，找到优化路径")
        else:
            print("⚠️  RRT*演示完成，但未找到路径")
        return True
    except Exception as e:
        print(f"❌ RRT*演示失败: {e}")
        traceback.print_exc()
        return False

def run_animation_demo():
    """运行动画演示"""
    print_header("🎬 动态可视化演示")
    
    try:
        from rrt_animation import demo_basic_rrt_animation
        print("正在启动动画演示...")
        print("💡 提示: 关闭动画窗口以继续下一个演示")
        demo_basic_rrt_animation()
        print("✅ 动画演示完成")
        return True
    except Exception as e:
        print(f"❌ 动画演示失败: {e}")
        traceback.print_exc()
        return False

def run_comparison_demo():
    """运行对比演示"""
    print_header("⚖️ 算法对比演示")
    
    try:
        from rrt_basic import comparison_demo
        print("正在进行参数对比...")
        comparison_demo()
        print("✅ 对比演示完成")
        return True
    except Exception as e:
        print(f"❌ 对比演示失败: {e}")
        traceback.print_exc()
        return False

def run_3d_terrain_demo():
    """运行3D地形规划演示"""
    print_header("🏔️ 3D地形路径规划演示")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '04_3D应用'))
        from terrain_planning import demo_3d_terrain_planning
        print("正在启动3D地形规划演示...")
        demo_3d_terrain_planning()
        print("✅ 3D演示完成")
        return True
    except Exception as e:
        print(f"❌ 3D演示失败: {e}")
        print("请确保安装了matplotlib和numpy的3D绘图支持")
        traceback.print_exc()
        return False

def run_benchmark_demo():
    """运行基准测试演示"""
    print_header("🧪 基准测试评估演示")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '05_测试数据'))
        from benchmark_scenarios import demo_benchmark_testing
        print("正在启动基准测试...")
        demo_benchmark_testing()
        print("✅ 基准测试完成")
        return True
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        traceback.print_exc()
        return False

def run_advanced_features_demo():
    """运行高级特性演示"""
    print_header("🚀 高级特性演示")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '06_高级特性'))
        from advanced_rrt import demo_informed_rrt_star
        print("正在启动Informed RRT*演示...")
        demo_informed_rrt_star()
        print("✅ 高级特性演示完成")
        return True
    except Exception as e:
        print(f"❌ 高级特性演示失败: {e}")
        traceback.print_exc()
        return False

def run_algorithm_comparison_demo():
    """运行算法性能对比演示"""
    print_header("📊 算法性能对比演示")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '07_算法对比'))
        from comparison import demo_comprehensive_comparison
        print("正在启动性能对比分析...")
        demo_comprehensive_comparison()
        print("✅ 性能对比完成")
        return True
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        traceback.print_exc()
        return False

def check_dependencies():
    """检查依赖"""
    print_header("🔍 检查依赖包")
    
    required_packages = [
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('scipy', 'scipy'),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name if import_name else package_name)
            print(f"✅ {package_name} - 已安装")
        except ImportError:
            print(f"❌ {package_name} - 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已正确安装")
    return True

def interactive_menu():
    """交互式菜单"""
    print_header("🎯 RRT算法学习 - 交互式演示菜单")
    
    menu_options = {
        '1': ('基础RRT算法演示', run_basic_rrt_demo),
        '2': ('RRT*优化算法演示', run_rrt_star_demo),
        '3': ('动态可视化演示', run_animation_demo),
        '4': ('算法参数对比', run_comparison_demo),
        '5': ('3D地形路径规划', run_3d_terrain_demo),
        '6': ('基准测试评估', run_benchmark_demo),
        '7': ('高级特性演示', run_advanced_features_demo),
        '8': ('算法性能对比', run_algorithm_comparison_demo),
        '9': ('运行所有演示', run_all_demos),
        '0': ('退出程序', lambda: False)
    }
    
    while True:
        print("\n📋 选择要运行的演示:")
        for key, (name, _) in menu_options.items():
            print(f"  {key}. {name}")
        
        choice = input("\n请输入选项 (0-9): ").strip()
        
        if choice in menu_options:
            name, func = menu_options[choice]
            if choice == '0':
                print("👋 感谢使用RRT算法学习系统!")
                break
            
            print(f"\n🚀 开始执行: {name}")
            success = func()
            
            if success:
                print(f"✅ {name} 执行完成")
            else:
                print(f"❌ {name} 执行失败")
                
            input("\n按回车键继续...")
        else:
            print("❌ 无效选项，请重新选择")

def run_all_demos():
    """运行所有演示"""
    print_header("🎪 运行所有核心演示")
    
    demos = [
        ("基础RRT", run_basic_rrt_demo),
        ("RRT*优化", run_rrt_star_demo),
        ("参数对比", run_comparison_demo),
    ]
    
    results = []
    
    for name, demo_func in demos:
        print(f"\n🔄 正在运行: {name}")
        success = demo_func()
        results.append((name, success))
        
        if not success:
            print(f"⚠️  {name} 演示遇到问题，但继续执行其他演示...")
    
    # 显示总结
    print_header("📊 演示结果总结")
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    print(f"\n🎯 成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    return True

def show_learning_guide():
    """显示学习指南"""
    print_header("📚 RRT算法学习指南")
    
    learning_path = [
        "1. 📖 理论基础 - 阅读 01_理论基础/RRT算法理论基础.md",
        "2. 💻 代码实现 - 研究 02_代码实现/ 目录下的源码",
        "3. 🎬 可视化理解 - 运行动态演示观察算法过程",
        "4. 🔧 参数调优 - 尝试不同参数组合",
        "5. 🚁 3D应用 - 探索三维环境下的应用",
        "6. 🔬 深入研究 - 学习高级特性和算法变种"
    ]
    
    print("🎯 建议的学习路径:")
    for step in learning_path:
        print(f"  {step}")
    
    print("\n💡 学习建议:")
    print("  • 先理解理论，再分析代码实现")
    print("  • 通过可视化加深对算法过程的理解")
    print("  • 尝试修改参数观察效果变化")
    print("  • 对比不同算法的优缺点")
    
    print("\n📁 目录结构:")
    dirs = [
        "01_理论基础/    - 数学原理和算法流程",
        "02_代码实现/    - Python实现源码",
        "03_可视化演示/  - 动态演示和交互界面",
        "04_3D应用/      - 三维环境应用案例",
        "05_测试数据/    - 标准测试场景",
        "06_高级特性/    - 算法变种和优化",
        "07_算法对比/    - 与其他算法的比较"
    ]
    
    for dir_info in dirs:
        print(f"  {dir_info}")

def main():
    """主函数"""
    print("🎯 欢迎使用RRT算法学习系统!")
    print("基于AICP-7协议构建的完整学习体验")
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装所需包")
        return
    
    # 显示学习指南
    show_learning_guide()
    
    # 询问运行模式
    print("\n🚀 选择运行模式:")
    print("  1. 交互式菜单 (推荐)")
    print("  2. 运行所有演示")
    print("  3. 仅显示学习指南")
    
    mode = input("\n请选择模式 (1-3): ").strip()
    
    if mode == "1":
        interactive_menu()
    elif mode == "2":
        run_all_demos()
        print("\n📚 建议继续阅读理论基础文档和代码实现")
    elif mode == "3":
        print("\n📖 请根据学习指南逐步学习各个模块")
    else:
        print("使用默认交互模式...")
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        traceback.print_exc() 