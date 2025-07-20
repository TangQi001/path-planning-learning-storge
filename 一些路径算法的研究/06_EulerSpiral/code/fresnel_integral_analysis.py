
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
# -*- coding: utf-8 -*-
"""
Fresnel积分与Euler螺旋数学分析

作者：AI助手
日期：2025年6月
功能：
1. Fresnel积分的数值计算与分析
2. 级数展开与收敛性分析
3. 数值积分方法比较
4. Euler螺旋的几何特性分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from scipy.integrate import quad, simpson
import time
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FresnelAnalyzer:
    """Fresnel积分分析器"""
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def fresnel_series_expansion(self, x, n_terms=10):
        """
        使用级数展开计算Fresnel积分
        
        参数:
        x: 输入值
        n_terms: 级数项数
        
        返回:
        S, C: Fresnel正弦积分和余弦积分
        """
        x = np.asarray(x)
        
        # 初始化
        S = np.zeros_like(x, dtype=float)
        C = np.zeros_like(x, dtype=float)
        
        # 计算级数
        for n in range(n_terms):
            # S(x) 级数
            coeff_s = (-1)**n / (math.factorial(2*n + 1) * (4*n + 3))
            term_s = coeff_s * x**(4*n + 3)
            S += term_s
            
            # C(x) 级数
            coeff_c = (-1)**n / (math.factorial(2*n) * (4*n + 1))
            term_c = coeff_c * x**(4*n + 1)
            C += term_c
        
        return S, C
    
    def numerical_integration(self, x, method='quad'):
        """
        使用数值积分计算Fresnel积分
        
        参数:
        x: 积分上限
        method: 积分方法 ('quad', 'simpson')
        
        返回:
        S, C: Fresnel积分值
        """
        def integrand_s(t):
            return np.sin(np.pi * t**2 / 2)
        
        def integrand_c(t):
            return np.cos(np.pi * t**2 / 2)
        
        if method == 'quad':
            S, _ = quad(integrand_s, 0, x)
            C, _ = quad(integrand_c, 0, x)
        elif method == 'simpson':
            t = np.linspace(0, x, 1001)  # 使用奇数个点
            y_s = integrand_s(t)
            y_c = integrand_c(t)
            S = simpson(y_s, t)
            C = simpson(y_c, t)
        
        return S, C
    
    def analyze_convergence(self, x_val=2.0, max_terms=20):
        """
        分析级数展开的收敛性
        
        参数:
        x_val: 测试点
        max_terms: 最大项数
        
        返回:
        results: 收敛分析结果
        """
        # 精确值（使用scipy）
        S_exact, C_exact = fresnel(x_val)
        
        # 不同项数的计算
        n_terms_list = range(1, max_terms + 1)
        errors_S = []
        errors_C = []
        
        for n in n_terms_list:
            S_approx, C_approx = self.fresnel_series_expansion(x_val, n)
            
            error_S = abs(S_approx - S_exact)
            error_C = abs(C_approx - C_exact)
            
            errors_S.append(error_S)
            errors_C.append(error_C)
        
        results = {
            'n_terms': list(n_terms_list),
            'errors_S': errors_S,
            'errors_C': errors_C,
            'S_exact': S_exact,
            'C_exact': C_exact
        }
        
        return results
    
    def compare_methods(self, x_range=(0, 5), n_points=50):
        """
        比较不同计算方法的性能和精度
        
        参数:
        x_range: x值范围
        n_points: 测试点数
        
        返回:
        comparison: 比较结果
        """
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        
        # 方法1：scipy内置函数
        start_time = time.time()
        S_scipy, C_scipy = fresnel(x_vals)
        time_scipy = time.time() - start_time
        
        # 方法2：级数展开（10项）
        start_time = time.time()
        S_series, C_series = self.fresnel_series_expansion(x_vals, 10)
        time_series = time.time() - start_time
        
        # 方法3：数值积分（对部分点）
        x_test = x_vals[::10]  # 每10个点测试一个
        S_quad = []
        C_quad = []
        
        start_time = time.time()
        for x in x_test:
            S_val, C_val = self.numerical_integration(x, 'quad')
            S_quad.append(S_val)
            C_quad.append(C_val)
        time_quad = time.time() - start_time
        
        # 计算误差
        error_S_series = np.abs(S_series - S_scipy)
        error_C_series = np.abs(C_series - C_scipy)
        
        error_S_quad = np.abs(np.array(S_quad) - S_scipy[::10])
        error_C_quad = np.abs(np.array(C_quad) - C_scipy[::10])
        
        comparison = {
            'x_vals': x_vals,
            'S_scipy': S_scipy,
            'C_scipy': C_scipy,
            'S_series': S_series,
            'C_series': C_series,
            'S_quad': S_quad,
            'C_quad': C_quad,
            'error_S_series': error_S_series,
            'error_C_series': error_C_series,
            'error_S_quad': error_S_quad,
            'error_C_quad': error_C_quad,
            'time_scipy': time_scipy,
            'time_series': time_series,
            'time_quad': time_quad,
            'x_test': x_test
        }
        
        return comparison
    
    def plot_fresnel_functions(self):
        """绘制Fresnel函数图像"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fresnel积分函数分析', fontsize=16, fontweight='bold')
        
        # 1. Fresnel积分函数图像
        x = np.linspace(0, 5, 1000)
        S, C = fresnel(x)
        
        ax1.plot(x, S, 'r-', linewidth=2, label='S(x) - 正弦积分')
        ax1.plot(x, C, 'b-', linewidth=2, label='C(x) - 余弦积分')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('积分值')
        ax1.set_title('Fresnel积分函数')
        ax1.legend()
        
        # 标记极限值
        limit_val = np.sqrt(np.pi/8)
        ax1.axhline(y=limit_val, color='gray', linestyle='--', alpha=0.7, label=f'极限值 = {limit_val:.3f}')
        ax1.axhline(y=-limit_val, color='gray', linestyle='--', alpha=0.7)
        
        # 2. Euler螺旋（参数图）
        ax2.plot(C, S, 'purple', linewidth=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('C(x)')
        ax2.set_ylabel('S(x)')
        ax2.set_title('Euler螺旋（羊角螺线）')
        ax2.axis('equal')
        
        # 标记特殊点
        for i, x_val in enumerate([1, 2, 3, 4]):
            if x_val < len(S):
                idx = int(x_val * 200)  # 近似索引
                ax2.plot(C[idx], S[idx], 'ro', markersize=6)
                ax2.annotate(f'x={x_val}', (C[idx], S[idx]), 
                           xytext=(5, 5), textcoords='offset points')
        
        # 3. 收敛性分析
        conv_results = self.analyze_convergence(2.0, 15)
        ax3.semilogy(conv_results['n_terms'], conv_results['errors_S'], 'r-o', label='S(x)误差')
        ax3.semilogy(conv_results['n_terms'], conv_results['errors_C'], 'b-s', label='C(x)误差')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('级数项数')
        ax3.set_ylabel('绝对误差')
        ax3.set_title('级数展开收敛性分析 (x=2.0)')
        ax3.legend()
        
        # 4. 被积函数图像
        t = np.linspace(0, 3, 1000)
        integrand_s = np.sin(np.pi * t**2 / 2)
        integrand_c = np.cos(np.pi * t**2 / 2)
        
        ax4.plot(t, integrand_s, 'r-', linewidth=2, label='sin(πt²/2)')
        ax4.plot(t, integrand_c, 'b-', linewidth=2, label='cos(πt²/2)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('t')
        ax4.set_ylabel('被积函数值')
        ax4.set_title('Fresnel积分的被积函数')
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_method_comparison(self):
        """绘制方法比较图"""
        comparison = self.compare_methods()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('不同计算方法比较', fontsize=16, fontweight='bold')
        
        # 1. 函数值比较
        x = comparison['x_vals']
        ax1.plot(x, comparison['S_scipy'], 'k-', linewidth=2, label='SciPy (精确)')
        ax1.plot(x, comparison['S_series'], 'r--', linewidth=2, label='级数展开 (10项)')
        ax1.plot(comparison['x_test'], np.array(comparison['S_quad']), 'bo', 
                markersize=4, label='数值积分')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('S(x)')
        ax1.set_title('Fresnel正弦积分比较')
        ax1.legend()
        
        # 2. 误差分析
        ax2.semilogy(x, comparison['error_S_series'], 'r-', linewidth=2, label='级数展开误差')
        ax2.semilogy(comparison['x_test'], comparison['error_S_quad'], 'bo', 
                    markersize=4, label='数值积分误差')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('绝对误差')
        ax2.set_title('S(x) 计算误差')
        ax2.legend()
        
        # 3. 余弦积分比较
        ax3.plot(x, comparison['C_scipy'], 'k-', linewidth=2, label='SciPy (精确)')
        ax3.plot(x, comparison['C_series'], 'b--', linewidth=2, label='级数展开 (10项)')
        ax3.plot(comparison['x_test'], np.array(comparison['C_quad']), 'ro', 
                markersize=4, label='数值积分')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('x')
        ax3.set_ylabel('C(x)')
        ax3.set_title('Fresnel余弦积分比较')
        ax3.legend()
        
        # 4. 计算时间比较
        methods = ['SciPy', '级数展开', '数值积分']
        times = [comparison['time_scipy'], comparison['time_series'], comparison['time_quad']]
        colors = ['green', 'orange', 'red']
        
        bars = ax4.bar(methods, times, color=colors, alpha=0.7)
        ax4.set_ylabel('计算时间 (秒)')
        ax4.set_title('不同方法计算时间比较')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{time_val:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def geometric_analysis(self):
        """几何特性分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Euler螺旋几何特性分析', fontsize=16, fontweight='bold')
        
        # 1. 曲率分析
        t = np.linspace(0, 3, 1000)
        curvature = 2 * t  # 标准化的Euler螺旋曲率
        
        ax1.plot(t, curvature, 'purple', linewidth=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('参数 t')
        ax1.set_ylabel('曲率 κ')
        ax1.set_title('曲率随参数线性变化')
        
        # 2. 切线角分析
        tangent_angle = t**2  # 切线角
        
        ax2.plot(t, np.degrees(tangent_angle), 'orange', linewidth=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('参数 t')
        ax2.set_ylabel('切线角 (度)')
        ax2.set_title('切线角随参数变化')
        
        # 3. 弧长分析
        # 对于标准化Euler螺旋，弧长 ≈ t（近似）
        arc_length_approx = t
        arc_length_exact = np.array([quad(lambda s: np.sqrt(1 + (2*s)**2), 0, t_val)[0] 
                                   for t_val in t[::50]])  # 采样计算
        
        ax3.plot(t, arc_length_approx, 'b-', linewidth=2, label='近似值 (s≈t)')
        ax3.plot(t[::50], arc_length_exact, 'ro', markersize=4, label='精确值')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('参数 t')
        ax3.set_ylabel('弧长 s')
        ax3.set_title('弧长分析')
        ax3.legend()
        
        # 4. 螺旋展开
        S, C = fresnel(t)
        
        # 显示螺旋的"展开"过程
        colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
        for i in range(0, len(t), 50):
            ax4.plot(C[i], S[i], 'o', color=colors[i], markersize=3)
        
        # 绘制完整螺旋
        ax4.plot(C, S, 'k-', linewidth=1, alpha=0.5)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('C(t)')
        ax4.set_ylabel('S(t)')
        ax4.set_title('螺旋展开过程（颜色代表时间）')
        ax4.axis('equal')
        
        plt.tight_layout()
        return fig

def main():
    """主函数"""
    print("=" * 60)
    print("Fresnel积分与Euler螺旋数学分析")
    print("=" * 60)
    
    analyzer = FresnelAnalyzer()
    
    # 1. 基础函数分析
    print("\n1. 生成Fresnel函数分析图...")
    fig1 = analyzer.plot_fresnel_functions()
    fig1.savefig('images/fresnel_functions_analysis.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: images/fresnel_functions_analysis.png")
    
    # 2. 方法比较
    print("\n2. 生成计算方法比较图...")
    fig2 = analyzer.plot_method_comparison()
    fig2.savefig('images/method_comparison.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: images/method_comparison.png")
    
    # 3. 几何特性分析
    print("\n3. 生成几何特性分析图...")
    fig3 = analyzer.geometric_analysis()
    fig3.savefig('images/geometric_analysis.png', dpi=300, bbox_inches='tight')
    print("   图形已保存: images/geometric_analysis.png")
    
    # 4. 数值分析
    print("\n4. 数值分析结果:")
    print("-" * 40)
    
    # 收敛性分析
    conv_results = analyzer.analyze_convergence(2.0, 10)
    print(f"在 x=2.0 处:")
    print(f"精确值: S = {conv_results['S_exact']:.6f}, C = {conv_results['C_exact']:.6f}")
    print(f"10项级数: 误差 S = {conv_results['errors_S'][-1]:.2e}, C = {conv_results['errors_C'][-1]:.2e}")
    
    # 方法比较
    comparison = analyzer.compare_methods((0, 3), 30)
    print(f"\n计算时间比较:")
    print(f"SciPy方法: {comparison['time_scipy']:.4f} 秒")
    print(f"级数展开: {comparison['time_series']:.4f} 秒")
    print(f"数值积分: {comparison['time_quad']:.4f} 秒")
    
    # 精度分析
    print(f"\n精度分析:")
    avg_error_s = np.mean(comparison['error_S_series'])
    avg_error_c = np.mean(comparison['error_C_series'])
    print(f"级数展开平均误差: S = {avg_error_s:.2e}, C = {avg_error_c:.2e}")
    
    print("\n数学分析完成！")
    print("请查看生成的图形文件以了解详细的数学特性。")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    main() 