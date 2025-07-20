"""
Enhanced Chinese Font Configuration Module
增强中文字体配置模块

Author: AI Assistant
Features: Cross-platform font support, automatic detection, fallback mechanism
功能: 跨平台字体支持、自动检测、fallback机制
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import platform
import os
import warnings

def setup_chinese_font():
    """
    Enhanced configuration for Chinese font display in matplotlib
    配置matplotlib中文字体显示（增强版）
    
    Supports Windows, macOS, Linux systems with better font detection
    支持Windows、macOS、Linux系统，具有更好的字体检测功能
    """
    # Suppress font warnings
    warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    system = platform.system()
    
    # Enhanced font candidates with more options
    font_candidates = []
    
    if system == "Windows":
        font_candidates = [
            'Microsoft YaHei',      # 微软雅黑
            'Microsoft YaHei UI',   # 微软雅黑 UI
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
            'Microsoft JhengHei',   # 微软正黑体
            'Arial Unicode MS'      # Arial Unicode MS
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            'PingFang SC',          # 苹方-简
            'PingFang TC',          # 苹方-繁
            'Hiragino Sans GB',     # 冬青黑体-简体中文
            'Hiragino Sans',        # 冬青黑体
            'STHeiti',              # 华文黑体
            'Heiti SC',             # 黑体-简
            'Heiti TC',             # 黑体-繁
            'Arial Unicode MS',     # Arial Unicode MS
            'Apple LiGothic'        # 苹果俪黑
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Noto Sans CJK TC',     # 思源黑体-繁体
            'Source Han Sans CN',   # 思源黑体
            'Droid Sans Fallback',  # Droid备用字体
            'AR PL UMing CN',       # 文鼎PL细上海宋
            'AR PL UKai CN'         # 文鼎PL中楷
        ]
    
    # Universal fallback fonts
    font_candidates.extend([
        'DejaVu Sans',
        'Liberation Sans',
        'Arial',
        'Helvetica',
        'sans-serif'
    ])
    
    # Get available fonts with better detection
    try:
        available_fonts = set()
        
        # Method 1: Use font manager
        for font in fm.fontManager.ttflist:
            available_fonts.add(font.name)
        
        # Method 2: Alternative detection
        system_fonts = [f.name for f in fm.findSystemFonts()]
        available_fonts.update(system_fonts)
        
    except Exception as e:
        print(f"Warning: Font detection error: {e}")
        available_fonts = set(['DejaVu Sans', 'Arial', 'sans-serif'])
    
    # Select the first available font
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
        
        # Try partial matching for font families
        for available in available_fonts:
            if font.lower() in available.lower() or available.lower() in font.lower():
                selected_font = available
                break
        
        if selected_font:
            break
    
    if selected_font is None:
        selected_font = 'DejaVu Sans'  # Ultimate fallback
    
    # Enhanced matplotlib configuration
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    # Font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    
    # Additional display settings
    plt.rcParams['axes.titlepad'] = 8
    plt.rcParams['axes.labelpad'] = 4
    
    # Set appropriate backend
    try:
        current_backend = matplotlib.get_backend()
        if current_backend == 'pdf' or current_backend == 'svg':
            # Keep current backend for file output
            pass
        else:
            # Try to set interactive backend
            try:
                import tkinter
                if current_backend != 'TkAgg':
                    matplotlib.use('TkAgg')
            except ImportError:
                try:
                    if current_backend != 'Qt5Agg':
                        matplotlib.use('Qt5Agg')
                except:
                    # Use non-interactive backend as last resort
                    matplotlib.use('Agg')
    except Exception as e:
        print(f"Backend configuration warning: {e}")
    
    print(f"✅ Enhanced Chinese font configuration complete: {selected_font}")
    print(f"   Platform: {system}, Backend: {matplotlib.get_backend()}")
    return selected_font

def test_chinese_display():
    """测试中文显示是否正常"""
    import numpy as np
    
    # 创建简单的测试图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标') 
    ax.set_title('中文字体显示测试')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax.annotate('峰值点', xy=(np.pi/2, 1), xytext=(2, 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 中文字体测试完成")

def get_font_info():
    """获取当前字体配置信息"""
    current_font = plt.rcParams['font.sans-serif'][0]
    print(f"当前使用字体: {current_font}")
    print(f"系统平台: {platform.system()}")
    print(f"matplotlib版本: {matplotlib.__version__}")
    
    return {
        'font': current_font,
        'platform': platform.system(),
        'matplotlib_version': matplotlib.__version__
    }

# 自动配置中文字体
if __name__ == "__main__":
    setup_chinese_font()
    test_chinese_display()
else:
    # 被导入时自动配置
    setup_chinese_font() 