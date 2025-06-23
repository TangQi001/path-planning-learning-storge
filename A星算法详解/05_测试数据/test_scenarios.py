"""
A*算法测试场景生成器
Author: AI Assistant
Description: 生成各种复杂度的测试用例，用于验证A*算法性能
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any

class TestScenarioGenerator:
    """测试场景生成器"""
    
    def __init__(self):
        self.scenarios = {}
    
    def generate_simple_scenarios(self) -> Dict[str, Any]:
        """生成简单测试场景"""
        scenarios = {}
        
        # 场景1: 简单直线路径
        scenarios['simple_line'] = {
            'name': '简单直线路径',
            'description': '无障碍物的5x5网格',
            'grid': [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            'start': (0, 0),
            'goal': (4, 4),
            'difficulty': 'easy'
        }
        
        # 场景2: 简单障碍绕行
        scenarios['simple_obstacle'] = {
            'name': '简单障碍绕行',
            'description': '包含一个障碍物，需要绕行',
            'grid': [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            'start': (0, 2),
            'goal': (4, 2),
            'difficulty': 'easy'
        }
        
        return scenarios
    
    def generate_medium_scenarios(self) -> Dict[str, Any]:
        """生成中等复杂度测试场景"""
        scenarios = {}
        
        # 场景1: 迷宫入门
        scenarios['maze_beginner'] = {
            'name': '迷宫入门',
            'description': '10x10的简单迷宫',
            'grid': [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            'start': (0, 0),
            'goal': (9, 9),
            'expected_path_length': 20,
            'difficulty': 'medium'
        }
        
        # 场景2: 多路径选择
        scenarios['multiple_paths'] = {
            'name': '多路径选择',
            'description': '存在多条可能路径，测试算法选择最优路径',
            'grid': [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            'start': (0, 0),
            'goal': (7, 7),
            'expected_path_length': 15,
            'difficulty': 'medium'
        }
        
        return scenarios
    
    def generate_hard_scenarios(self) -> Dict[str, Any]:
        """生成高难度测试场景"""
        scenarios = {}
        
        # 场景1: 复杂迷宫
        scenarios['complex_maze'] = {
            'name': '复杂迷宫',
            'description': '15x15的复杂迷宫，多个死胡同',
            'grid': self._generate_complex_maze(15, 15),
            'start': (0, 0),
            'goal': (14, 14),
            'expected_path_length': 30,
            'difficulty': 'hard'
        }
        
        # 场景2: 稀疏通道
        scenarios['sparse_passage'] = {
            'name': '稀疏通道',
            'description': '密集障碍物，只有少数通道',
            'grid': self._generate_sparse_passage(12, 12),
            'start': (0, 0),
            'goal': (11, 11),
            'expected_path_length': 25,
            'difficulty': 'hard'
        }
        
        return scenarios
    
    def generate_edge_cases(self) -> Dict[str, Any]:
        """生成边界测试用例"""
        scenarios = {}
        
        # 场景1: 无解路径
        scenarios['no_solution'] = {
            'name': '无解路径',
            'description': '起点和终点被完全隔离',
            'grid': [
                [0, 1, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0]
            ],
            'start': (0, 0),
            'goal': (2, 2),
            'expected_path_length': None,  # 无解
            'difficulty': 'edge_case'
        }
        
        # 场景2: 起点即终点
        scenarios['same_start_goal'] = {
            'name': '起点即终点',
            'description': '起始位置和目标位置相同',
            'grid': [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            'start': (1, 1),
            'goal': (1, 1),
            'expected_path_length': 1,
            'difficulty': 'edge_case'
        }
        
        # 场景3: 最小网格
        scenarios['minimal_grid'] = {
            'name': '最小网格',
            'description': '1x1网格测试',
            'grid': [[0]],
            'start': (0, 0),
            'goal': (0, 0),
            'expected_path_length': 1,
            'difficulty': 'edge_case'
        }
        
        return scenarios
    
    def _generate_complex_maze(self, rows: int, cols: int) -> List[List[int]]:
        """生成复杂迷宫"""
        np.random.seed(42)  # 固定随机种子以获得可重现结果
        
        # 创建密集障碍物网格
        grid = np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])
        
        # 确保起点和终点可通行
        grid[0, 0] = 0
        grid[rows-1, cols-1] = 0
        
        # 创建一些保证的通道
        for i in range(0, rows, 2):
            grid[i, :] = np.random.choice([0, 1], size=cols, p=[0.8, 0.2])
        
        return grid.tolist()
    
    def _generate_sparse_passage(self, rows: int, cols: int) -> List[List[int]]:
        """生成稀疏通道"""
        # 创建主要是障碍物的网格
        grid = np.ones((rows, cols))
        
        # 创建主通道
        for i in range(rows):
            grid[i, i] = 0  # 对角线通道
            if i > 0:
                grid[i-1, i] = 0  # 辅助通道
        
        # 添加一些分支通道
        for i in range(0, rows, 3):
            for j in range(cols):
                if np.random.random() > 0.7:
                    grid[i, j] = 0
        
        # 确保起点和终点可通行
        grid[0, 0] = 0
        grid[rows-1, cols-1] = 0
        
        return grid.tolist()
    
    def generate_all_scenarios(self) -> Dict[str, Any]:
        """生成所有测试场景"""
        all_scenarios = {}
        
        all_scenarios.update(self.generate_simple_scenarios())
        all_scenarios.update(self.generate_medium_scenarios())
        all_scenarios.update(self.generate_hard_scenarios())
        all_scenarios.update(self.generate_edge_cases())
        
        return all_scenarios
    
    def save_scenarios_to_file(self, filename: str = 'test_scenarios.json'):
        """保存测试场景到文件"""
        scenarios = self.generate_all_scenarios()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, ensure_ascii=False, indent=2)
        
        print(f"测试场景已保存到 {filename}")
        return scenarios
    
    def load_scenarios_from_file(self, filename: str = 'test_scenarios.json') -> Dict[str, Any]:
        """从文件加载测试场景"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            print(f"从 {filename} 加载了 {len(scenarios)} 个测试场景")
            return scenarios
        except FileNotFoundError:
            print(f"文件 {filename} 不存在，生成新的测试场景")
            return self.generate_all_scenarios()

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.results = []
    
    def run_performance_test(self, astar_algorithm, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """运行性能测试"""
        import time
        
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\n测试场景: {scenario['name']}")
            
            try:
                start_time = time.time()
                path = astar_algorithm.find_path(scenario['start'], scenario['goal'])
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                result = {
                    'scenario': scenario_name,
                    'success': path is not None,
                    'execution_time': execution_time,
                    'path_length': len(path) if path else 0,
                    'expected_length': scenario.get('expected_path_length'),
                    'difficulty': scenario['difficulty'],
                    'grid_size': (len(scenario['grid']), len(scenario['grid'][0]))
                }
                
                # 验证路径长度
                expected_length = scenario.get('expected_path_length')
                if expected_length is not None and path:
                    length_diff = abs(len(path) - expected_length)
                    result['length_accuracy'] = length_diff <= 2  # 允许2步误差
                
                results[scenario_name] = result
                
                # 打印结果
                if path:
                    print(f"  ✅ 成功找到路径，长度: {len(path)}, 用时: {execution_time:.4f}s")
                else:
                    print(f"  ❌ 未找到路径，用时: {execution_time:.4f}s")
                    
            except Exception as e:
                print(f"  ⚠️ 测试失败: {str(e)}")
                results[scenario_name] = {
                    'scenario': scenario_name,
                    'success': False,
                    'error': str(e),
                    'difficulty': scenario['difficulty']
                }
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """生成性能报告"""
        report = "\n" + "="*60 + "\n"
        report += "A*算法性能测试报告\n"
        report += "="*60 + "\n"
        
        # 统计信息
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        
        report += f"总测试数: {total_tests}\n"
        report += f"成功测试: {successful_tests}\n"
        report += f"成功率: {successful_tests/total_tests*100:.1f}%\n\n"
        
        # 按难度分组统计
        difficulty_stats = {}
        for result in results.values():
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'total': 0, 'success': 0, 'time': 0}
            
            difficulty_stats[difficulty]['total'] += 1
            if result.get('success', False):
                difficulty_stats[difficulty]['success'] += 1
                difficulty_stats[difficulty]['time'] += result.get('execution_time', 0)
        
        report += "按难度统计:\n"
        for difficulty, stats in difficulty_stats.items():
            success_rate = stats['success'] / stats['total'] * 100
            avg_time = stats['time'] / max(stats['success'], 1)
            report += f"  {difficulty}: {stats['success']}/{stats['total']} ({success_rate:.1f}%), 平均用时: {avg_time:.4f}s\n"
        
        report += "\n详细结果:\n"
        for scenario_name, result in results.items():
            if result.get('success', False):
                report += f"  ✅ {scenario_name}: {result.get('path_length', 0)}步, {result.get('execution_time', 0):.4f}s\n"
            else:
                report += f"  ❌ {scenario_name}: 失败\n"
        
        return report

def main():
    """主函数 - 演示测试场景生成"""
    print("=== A*算法测试场景生成器 ===")
    
    # 创建测试场景生成器
    generator = TestScenarioGenerator()
    
    # 生成并保存所有测试场景
    scenarios = generator.save_scenarios_to_file()
    
    # 显示场景统计
    difficulty_count = {}
    for scenario in scenarios.values():
        difficulty = scenario['difficulty']
        difficulty_count[difficulty] = difficulty_count.get(difficulty, 0) + 1
    
    print(f"\n生成的测试场景统计:")
    print(f"总场景数: {len(scenarios)}")
    for difficulty, count in difficulty_count.items():
        print(f"  {difficulty}: {count} 个")
    
    # 显示场景列表
    print(f"\n场景列表:")
    for name, scenario in scenarios.items():
        print(f"  - {scenario['name']} ({scenario['difficulty']})")
        print(f"    {scenario['description']}")
        print(f"    网格大小: {len(scenario['grid'])}x{len(scenario['grid'][0])}")
        print()

if __name__ == "__main__":
    main() 