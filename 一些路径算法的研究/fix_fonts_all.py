#!/usr/bin/env python3
"""
Universal Font Fix Script for Path Planning Algorithms
路径规划算法通用字体修复脚本

This script automatically fixes Chinese font display issues across all Python files
in the path planning algorithms project.

Usage:
    python fix_fonts_all.py

Author: AI Assistant
Date: 2025-01
"""

import os
import re
import sys
import glob
from pathlib import Path

def fix_font_in_file(file_path):
    """
    Fix font configuration in a Python file
    修复Python文件中的字体配置
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file already has font configuration
        if 'font_config' in content or 'configure_chinese_font' in content:
            print(f"✓ {file_path} already has font configuration")
            return False
        
        # Check if file uses matplotlib
        if 'matplotlib' not in content and 'plt.' not in content:
            print(f"- {file_path} doesn't use matplotlib, skipping")
            return False
        
        # Find the import section
        lines = content.split('\n')
        import_end_idx = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end_idx = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # Add font configuration import and setup
        font_config_lines = [
            "",
            "# Fix Chinese font display",
            "try:",
            "    from font_config import configure_chinese_font",
            "    configure_chinese_font()",
            "except ImportError:",
            "    # Fallback font configuration",
            "    import matplotlib.pyplot as plt",
            "    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']",
            "    plt.rcParams['axes.unicode_minus'] = False",
            ""
        ]
        
        # Insert font configuration after imports
        lines[import_end_idx:import_end_idx] = font_config_lines
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Fixed font configuration in {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def find_python_files(directory):
    """
    Find all Python files that use matplotlib
    查找所有使用matplotlib的Python文件
    """
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files

def create_font_config_copies():
    """
    Create copies of font_config.py in relevant directories
    在相关目录中创建font_config.py的副本
    """
    base_dir = os.getcwd()
    font_config_source = os.path.join(base_dir, 'font_config.py')
    
    if not os.path.exists(font_config_source):
        print("❌ font_config.py not found in current directory")
        return
    
    # Read the source font config
    with open(font_config_source, 'r', encoding='utf-8') as f:
        font_config_content = f.read()
    
    # Directories that need font_config.py
    target_dirs = [
        '01_AStar',
        '02_RRT/implementation',
        '02_RRT/visualization', 
        '03_Bezier/implementation',
        '03_Bezier/visualization',
        '03_Bezier/3d_applications',
        '04_Dubins/implementation',
        '04_Dubins/visualization',
        '05_Voronoi/implementation',
        '05_Voronoi/visualization',
        '05_Voronoi/3d_applications',
        '05_Voronoi/advanced_features',
        '06_EulerSpiral/code'
    ]
    
    for target_dir in target_dirs:
        target_path = os.path.join(base_dir, target_dir)
        if os.path.exists(target_path):
            font_config_target = os.path.join(target_path, 'font_config.py')
            
            # Only copy if it doesn't exist or is different
            if not os.path.exists(font_config_target):
                with open(font_config_target, 'w', encoding='utf-8') as f:
                    f.write(font_config_content)
                print(f"✅ Created font_config.py in {target_dir}")

def fix_specific_dubins_files():
    """
    Apply specific fixes to Dubins visualization files
    对Dubins可视化文件应用特定修复
    """
    base_dir = os.getcwd()
    
    # Update original dubins_visualizer.py to use enhanced font config
    original_dubins = os.path.join(base_dir, '04_Dubins/visualization/dubins_visualizer.py')
    
    if os.path.exists(original_dubins):
        try:
            with open(original_dubins, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the simple font configuration with enhanced version
            old_font_config = """# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False"""
            
            new_font_config = """# Enhanced Chinese font configuration
try:
    from font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback font configuration
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False"""
            
            if old_font_config in content:
                content = content.replace(old_font_config, new_font_config)
                
                with open(original_dubins, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print("✅ Updated original Dubins visualizer with enhanced font config")
            
        except Exception as e:
            print(f"❌ Error updating Dubins visualizer: {e}")

def main():
    """
    Main function to fix all font issues
    修复所有字体问题的主函数
    """
    print("=" * 60)
    print("Universal Font Fix for Path Planning Algorithms")
    print("路径规划算法通用字体修复工具")
    print("=" * 60)
    
    base_dir = os.getcwd()
    print(f"Working directory: {base_dir}")
    
    # Step 1: Create font_config.py copies
    print("\n📁 Step 1: Creating font_config.py copies...")
    create_font_config_copies()
    
    # Step 2: Fix specific Dubins files
    print("\n🔧 Step 2: Applying specific fixes...")
    fix_specific_dubins_files()
    
    # Step 3: Find and fix all Python files
    print("\n🔍 Step 3: Finding Python files that use matplotlib...")
    python_files = find_python_files(base_dir)
    matplotlib_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'matplotlib' in content or 'plt.' in content:
                matplotlib_files.append(file_path)
        except:
            continue
    
    print(f"Found {len(matplotlib_files)} Python files using matplotlib")
    
    # Step 4: Apply font fixes
    print("\n🎨 Step 4: Applying font fixes...")
    fixed_count = 0
    
    for file_path in matplotlib_files:
        # Skip font_config.py files and enhanced versions
        if 'font_config.py' in file_path or 'enhanced_' in file_path:
            continue
            
        if fix_font_in_file(file_path):
            fixed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY / 总结")
    print("=" * 60)
    print(f"📊 Total matplotlib files found: {len(matplotlib_files)}")
    print(f"✅ Files successfully fixed: {fixed_count}")
    print(f"📁 Font config files created in multiple directories")
    print(f"🎯 Enhanced Dubins visualizer available")
    
    print("\n🚀 Font fix complete! Chinese characters should now display correctly.")
    print("   字体修复完成！中文字符现在应该可以正确显示。")
    
    # Test recommendation
    print("\n💡 To test the fix, run:")
    print("   python font_config.py")
    print("   python 04_Dubins/visualization/enhanced_dubins_visualizer.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()