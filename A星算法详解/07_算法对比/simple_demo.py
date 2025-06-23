#!/usr/bin/env python3
"""
简单算法对比演示
快速展示A*、Dijkstra、贪心最佳优先算法的差异
"""

import sys
import os
import time

# 简化的算法实现
def simple_dijkstra_demo():
    """Dijkstra算法演示"""
    print("🔹 Dijkstra算法演示")
    print("特点: 保证找到最优路径，但搜索范围大")
    print("算法: 总是选择距离起点最近的未访问节点")
    print("公式: f(n) = g(n)  (只考虑实际代价)")
    
    # 模拟搜索过程
    print("\n搜索过程模拟:")
    nodes = ["Start", "A", "B", "C", "D", "Goal"]
    distances = [0, 2, 3, 4, 5, 6]
    
    for i, (node, dist) in enumerate(zip(nodes, distances)):
        time.sleep(0.5)
        print(f"  步骤 {i+1}: 访问节点 {node}, 距离起点 {dist}")
    
    print("✅ 找到最优路径，但探索了所有可能的节点")

def simple_greedy_demo():
    """贪心最佳优先算法演示"""
    print("\n🔹 贪心最佳优先算法演示")
    print("特点: 速度快，直奔目标，但可能不是最优")
    print("算法: 总是选择看起来最接近目标的节点")
    print("公式: f(n) = h(n)  (只考虑启发函数)")
    
    # 模拟搜索过程
    print("\n搜索过程模拟:")
    nodes = ["Start", "C", "Goal"]
    heuristics = [10, 5, 0]
    
    for i, (node, h) in enumerate(zip(nodes, heuristics)):
        time.sleep(0.5)
        print(f"  步骤 {i+1}: 访问节点 {node}, 预估距离目标 {h}")
    
    print("✅ 快速找到路径，但可能不是最优解")

def simple_astar_demo():
    """A*算法演示"""
    print("\n🔹 A*算法演示") 
    print("特点: 平衡最优性和效率，通常表现最佳")
    print("算法: 结合实际距离和预估距离")
    print("公式: f(n) = g(n) + h(n)  (实际代价 + 启发函数)")
    
    # 模拟搜索过程
    print("\n搜索过程模拟:")
    nodes = ["Start", "A", "B", "Goal"]
    g_values = [0, 2, 3, 5]
    h_values = [10, 6, 4, 0]
    f_values = [g + h for g, h in zip(g_values, h_values)]
    
    for i, (node, g, h, f) in enumerate(zip(nodes, g_values, h_values, f_values)):
        time.sleep(0.5)
        print(f"  步骤 {i+1}: 访问节点 {node}, g={g}, h={h}, f={f}")
    
    print("✅ 找到最优路径，搜索效率高")

def algorithm_comparison_table():
    """算法对比表格"""
    print("\n" + "="*60)
    print("📊 三种算法特点对比")
    print("="*60)
    
    comparison_data = [
        ("特性", "Dijkstra", "贪心BFS", "A*"),
        ("最优性保证", "✅ 是", "❌ 否", "✅ 条件性"),
        ("搜索效率", "❌ 低", "✅ 很高", "✅ 高"),
        ("内存使用", "❌ 高", "✅ 低", "⚖️ 中等"),
        ("启发信息", "❌ 不使用", "✅ 仅启发", "✅ 平衡使用"),
        ("适用场景", "最短路径", "快速路径", "路径规划"),
        ("计算复杂度", "O(E+V)logV", "O(b^m)", "O(b^d)"),
    ]
    
    # 打印表格
    for row in comparison_data:
        print(f"{row[0]:<12} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12}")
        if row[0] == "特性":
            print("-" * 60)

def practical_applications():
    """实际应用场景"""
    print("\n" + "="*60)
    print("🌍 实际应用场景")
    print("="*60)
    
    applications = {
        "Dijkstra算法": [
            "🌐 网络路由协议 (OSPF)",
            "📍 GPS导航系统的基础算法",
            "🎯 社交网络中的最短关系路径",
            "💰 金融网络中的最低成本路径"
        ],
        "贪心最佳优先算法": [
            "🎮 实时策略游戏中的快速寻路",
            "🚨 紧急情况下的快速路径规划",
            "📱 移动设备上的轻量级导航",
            "🤖 资源受限机器人的简单导航"
        ],
        "A*算法": [
            "🎮 游戏AI中的角色移动",
            "🤖 机器人路径规划",
            "🚗 自动驾驶汽车导航",
            "🏭 工业自动化中的路径优化"
        ]
    }
    
    for algorithm, apps in applications.items():
        print(f"\n【{algorithm}】")
        for app in apps:
            print(f"  {app}")

def performance_simulation():
    """性能模拟对比"""
    print("\n" + "="*60)
    print("⚡ 性能模拟对比")
    print("="*60)
    
    # 模拟数据
    scenarios = ["简单场景", "复杂场景", "迷宫场景"]
    
    # 探索节点数 (相对值)
    dijkstra_nodes = [25, 85, 95]
    greedy_nodes = [8, 15, 40]
    astar_nodes = [12, 35, 60]
    
    # 路径长度 (相对值)
    dijkstra_path = [10, 15, 20]  # 最优
    greedy_path = [12, 18, 28]    # 可能次优
    astar_path = [10, 15, 20]     # 通常最优
    
    print(f"{'场景':<12} | {'算法':<12} | {'探索节点':<8} | {'路径长度':<8} | {'特点'}")
    print("-" * 65)
    
    for i, scenario in enumerate(scenarios):
        print(f"{scenario:<12} | Dijkstra   | {dijkstra_nodes[i]:<8} | {dijkstra_path[i]:<8} | 最优但慢")
        print(f"{'':>12} | 贪心BFS    | {greedy_nodes[i]:<8} | {greedy_path[i]:<8} | 快但可能次优")
        print(f"{'':>12} | A*         | {astar_nodes[i]:<8} | {astar_path[i]:<8} | 平衡最优")
        if i < len(scenarios) - 1:
            print("-" * 65)

def main():
    """主函数"""
    print("🎯 路径搜索算法简单对比演示")
    print("="*60)
    
    print("本演示将通过简单的例子展示三种经典路径搜索算法：")
    print("• Dijkstra算法 (1959年)")
    print("• 贪心最佳优先算法")
    print("• A*算法 (1968年)")
    
    print("\n按Enter键开始演示...")
    input()
    
    # 运行各个演示
    try:
        simple_dijkstra_demo()
        input("\n按Enter键继续...")
        
        simple_greedy_demo()
        input("\n按Enter键继续...")
        
        simple_astar_demo()
        input("\n按Enter键继续...")
        
        algorithm_comparison_table()
        input("\n按Enter键继续...")
        
        performance_simulation()
        input("\n按Enter键继续...")
        
        practical_applications()
        
    except KeyboardInterrupt:
        print("\n\n演示被中断。")
        return
    
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    
    print("\n总结：")
    print("• Dijkstra：保证最优，但搜索范围大")
    print("• 贪心BFS：速度最快，但可能找到次优解")
    print("• A*：平衡效率和最优性，实际应用最广泛")
    
    print(f"\n要查看完整的算法实现和可视化演示，请运行：")
    print(f"python algorithms_comparison.py")
    
    print(f"\n感谢使用算法对比演示！")

if __name__ == "__main__":
    main() 