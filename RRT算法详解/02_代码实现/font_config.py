"""
中文字体配置模块

作者: AICP-7协议实现
功能: 解决matplotlib中文显示乱码问题
特点: 跨平台字体支持、自动检测、fallback机制
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import os

def setup_chinese_font():
    """
    配置matplotlib中文字体显示
    
    支持Windows、macOS、Linux系统
    自动检测并配置合适的中文字体
    """
    system = platform.system()
    
    # 尝试的字体列表（按优先级排序）
    font_candidates = []
    
    if system == "Windows":
        font_candidates = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong'              # 仿宋
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            'PingFang SC',          # 苹方-简
            'Hiragino Sans GB',     # 冬青黑体-简体中文
            'STHeiti',              # 华文黑体
            'Arial Unicode MS'      # Arial Unicode MS
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Droid Sans Fallback'   # Droid备用字体
        ]
    
    # 通用字体（作为备选）
    font_candidates.extend([
        'DejaVu Sans',
        'Arial',
        'sans-serif'
    ])
    
    # 获取系统可用字体
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    # 选择第一个可用的字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font is None:
        selected_font = font_candidates[-1]  # 使用最后一个作为默认
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [selected_font] + ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置默认字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    
    print(f"✅ 中文字体配置完成: {selected_font}")
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