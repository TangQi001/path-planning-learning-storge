#!/usr/bin/env python3
"""
测试导入模块 - 验证模块导入是否正常
==========================================

这个脚本用于测试所有必要的模块是否能够正常导入
"""

import os
import sys

def test_core_module():
    """测试核心模块导入"""
    print("🔍 测试core_voronoi模块导入...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_path = os.path.join(current_dir, '02_代码实现')
        sys.path.insert(0, core_path)
        
        from core_voronoi import VoronoiPathPlanner, Point, demo_basic_planning, demo_comparison
        print("✅ core_voronoi模块导入成功")
        
        # 测试基本功能
        print("📊 测试基本功能...")
        planner = VoronoiPathPlanner(bounds=(0, 0, 10, 10))
        print(f"  规划器创建成功，边界: {planner.bounds}")
        
        point = Point(1, 1)
        print(f"  点创建成功: ({point.x}, {point.y})")
        
        print("✅ 核心模块功能正常")
        return True
        
    except Exception as e:
        print(f"❌ core_voronoi模块导入失败: {e}")
        return False

def test_interactive_module():
    """测试交互模块导入"""
    print("\n🔍 测试interactive_demo模块导入...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        demo_path = os.path.join(current_dir, '03_可视化演示')
        sys.path.insert(0, demo_path)
        
        from interactive_demo import InteractiveVoronoiDemo
        print("✅ interactive_demo模块导入成功")
        
        # 测试基本功能（不启动GUI）
        print("📊 测试基本功能...")
        # 只测试类的创建，不运行GUI
        demo_class = InteractiveVoronoiDemo
        print("  交互演示类加载成功")
        
        print("✅ 交互模块功能正常")
        return True
        
    except Exception as e:
        print(f"❌ interactive_demo模块导入失败: {e}")
        return False

def test_dependencies():
    """测试依赖库"""
    print("\n🔍 测试依赖库...")
    dependencies = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('scipy', 'SciPy'),
        ('networkx', 'NetworkX')
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} - 已安装")
        except ImportError:
            print(f"❌ {name} - 未安装")
            all_ok = False
    
    return all_ok

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 Voronoi图法路径规划 - 模块导入测试")
    print("=" * 60)
    
    # 测试依赖库
    deps_ok = test_dependencies()
    
    # 测试核心模块
    core_ok = test_core_module()
    
    # 测试交互模块
    interactive_ok = test_interactive_module()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试结果总结:")
    print("=" * 60)
    
    if deps_ok and core_ok and interactive_ok:
        print("🎉 所有测试通过！模块导入正常，可以运行主程序。")
        print("\n💡 现在可以运行以下命令启动主程序：")
        print("   python demo_main.py")
    else:
        print("⚠️  存在问题，请检查以下项目：")
        if not deps_ok:
            print("   - 依赖库安装")
        if not core_ok:
            print("   - 核心模块导入")
        if not interactive_ok:
            print("   - 交互模块导入")

if __name__ == "__main__":
    main() 