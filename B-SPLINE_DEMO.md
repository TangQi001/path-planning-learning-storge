# B-样条曲线 Manim 演示

## 🚀 快速开始

### 方法一：从主目录运行（推荐）

```bash
# 预览模式（快速渲染）
python run_bspline.py --preview

# 中等质量渲染
python run_bspline.py

# 高质量渲染
python run_bspline.py --quality high

# 运行特定场景
python run_bspline.py --scene BSplineDemo
python run_bspline.py --scene BSplineBasicFunctions
python run_bspline.py --scene BSplineInteractive
```

### 方法二：进入 mnimi 目录运行

```bash
cd mnimi
python run_bspline_demo.py --preview
```

### 方法三：直接使用 Manim 命令

```bash
cd mnimi
manim -p -ql bspline_demo.py BSplineDemo
```

## 📋 前置要求

1. **安装依赖**：
   ```bash
   pip install manim scipy numpy
   ```

2. **测试环境**：
   ```bash
   cd mnimi
   python test_environment.py
   ```

## 🎬 演示场景

- **BSplineDemo**: 基础演示（3-4分钟）
- **BSplineBasicFunctions**: 基函数演示（2-3分钟）  
- **BSplineInteractive**: 交互式演示（2-3分钟）
- **BSplineComplete**: 完整演示（8-10分钟）

## 🔧 故障排除

如果遇到 "FileNotFoundError" 错误：
1. 确保在正确的目录中运行脚本
2. 使用 `python run_bspline.py` 从主目录运行
3. 或者 `cd mnimi && python run_bspline_demo.py` 从子目录运行

## 📁 文件结构

```
.
├── run_bspline.py           # 主目录启动脚本
├── B-SPLINE_DEMO.md         # 本说明文件
└── mnimi/                   # 演示文件目录
    ├── bspline_demo.py      # 主要演示代码
    ├── run_bspline_demo.py  # 运行脚本
    ├── test_environment.py  # 环境测试
    ├── requirements.txt     # 依赖列表
    └── README.md            # 详细文档
```

渲染完成的视频将保存在 `mnimi/media/videos/` 目录中。 