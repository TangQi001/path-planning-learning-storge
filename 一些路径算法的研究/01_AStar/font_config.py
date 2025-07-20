#!/usr/bin/env python3
"""
Font Configuration for Chinese Display
中文字体配置模块

This module provides cross-platform Chinese font configuration for matplotlib
to fix the issue where Chinese characters appear as squares.

Author: AI Assistant
Date: 2025-01
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import os
import warnings


def configure_chinese_font():
    """
    Configure matplotlib to properly display Chinese characters
    配置matplotlib以正确显示中文字符
    """
    
    system = platform.system()
    
    # Suppress font warnings
    warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
    
    if system == "Darwin":  # macOS
        font_candidates = [
            'Arial Unicode MS',
            'Heiti TC',
            'Heiti SC', 
            'STHeiti',
            'PingFang SC',
            'Hiragino Sans GB',
            'Microsoft YaHei',
            'SimHei'
        ]
    elif system == "Windows":  # Windows
        font_candidates = [
            'Microsoft YaHei',
            'SimHei',
            'KaiTi',
            'FangSong',
            'SimSun',
            'Arial Unicode MS'
        ]
    else:  # Linux and others
        font_candidates = [
            'DejaVu Sans',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Source Han Sans CN',
            'Microsoft YaHei',
            'SimHei'
        ]
    
    # Find available font
    available_font = None
    for font in font_candidates:
        try:
            # Test if font exists
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            available_font = font
            break
        except:
            continue
    
    if available_font:
        print(f"Using Chinese font: {available_font}")
        plt.rcParams['font.sans-serif'] = [available_font] + plt.rcParams['font.sans-serif']
    else:
        print("Warning: No suitable Chinese font found. Using fallback configuration.")
        # Fallback configuration
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    # Configure matplotlib parameters
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Set backend if needed
    try:
        import tkinter
        matplotlib.use('TkAgg')
    except ImportError:
        try:
            matplotlib.use('Qt5Agg')
        except:
            matplotlib.use('Agg')  # Fallback to non-interactive backend
    
    return available_font


def test_chinese_display():
    """
    Test Chinese character display
    测试中文字符显示
    """
    import numpy as np
    
    # Configure font
    font_used = configure_chinese_font()
    
    # Create test plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='正弦函数 sin(x)')
    ax.set_xlabel('X轴坐标')
    ax.set_ylabel('Y轴坐标')
    ax.set_title('中文字体测试 - Chinese Font Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(np.pi/2, 0.5, '峰值点', fontsize=12, ha='center')
    ax.text(3*np.pi/2, -0.5, '谷值点', fontsize=12, ha='center')
    
    plt.tight_layout()
    
    print(f"Font test complete. Used font: {font_used}")
    print("If Chinese characters display correctly, the font configuration is working.")
    print("如果中文字符显示正确，字体配置工作正常。")
    
    plt.show()
    return True


def get_system_fonts():
    """
    Get list of available system fonts
    获取系统可用字体列表
    """
    font_list = []
    
    try:
        # Get all available fonts
        available_fonts = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
        
        # Filter Chinese fonts
        chinese_keywords = ['微软雅黑', 'Microsoft YaHei', 'SimHei', '黑体', 'Heiti', 
                           'PingFang', 'Hiragino', 'WenQuanYi', 'Noto Sans CJK']
        
        for font in available_fonts:
            for keyword in chinese_keywords:
                if keyword.lower() in font.lower():
                    if font not in font_list:
                        font_list.append(font)
                    break
        
        return sorted(list(set(font_list)))
    
    except Exception as e:
        print(f"Error getting fonts: {e}")
        return []


if __name__ == "__main__":
    print("=== 中文字体配置测试 ===")
    print("=== Chinese Font Configuration Test ===\n")
    
    # Show system info
    print(f"Operating System: {platform.system()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Matplotlib Version: {matplotlib.__version__}\n")
    
    # Show available Chinese fonts
    print("Available Chinese fonts:")
    chinese_fonts = get_system_fonts()
    if chinese_fonts:
        for font in chinese_fonts[:10]:  # Show first 10
            print(f"  - {font}")
        if len(chinese_fonts) > 10:
            print(f"  ... and {len(chinese_fonts) - 10} more")
    else:
        print("  No Chinese fonts found")
    
    print("\n" + "="*50)
    
    # Test font configuration
    try:
        test_chinese_display()
    except Exception as e:
        print(f"Font test failed: {e}")
        print("Please install Chinese fonts or check matplotlib configuration.")