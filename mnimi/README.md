# B-样条曲线 (B-Spline) Manim 演示

这是一个使用 Manim 制作的 B-样条曲线教学演示视频项目。通过动画的方式展示 B-样条曲线的基本概念、数学原理和应用特性。

## 🎯 项目特色

- **全面的理论介绍**：从基础概念到数学公式的完整讲解
- **可视化演示**：控制点、控制多边形、曲线生成的动态展示
- **交互性展示**：演示控制点变化对曲线形状的实时影响
- **基函数可视化**：B-样条基函数的图形化展示
- **局部性质演示**：展示 B-样条的重要特性
- **中文支持**：完全中文的教学内容

## 📋 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows、macOS 或 Linux

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install manim scipy numpy
```

### 2. 下载项目文件

确保您有以下文件：
- `bspline_demo.py` - 主要的演示代码
- `run_bspline_demo.py` - 运行脚本
- `README.md` - 本说明文件

### 3. 运行演示

#### 基础使用
```bash
# 运行默认演示
python run_bspline_demo.py

# 预览模式（快速渲染）
python run_bspline_demo.py --preview

# 高质量渲染
python run_bspline_demo.py --quality high
```

#### 运行特定场景
```bash
# 基础 B-样条演示
python run_bspline_demo.py --scene BSplineDemo

# B-样条基函数演示
python run_bspline_demo.py --scene BSplineBasicFunctions

# 交互式演示
python run_bspline_demo.py --scene BSplineInteractive

# 完整演示（所有场景）
python run_bspline_demo.py --scene BSplineComplete
```

#### 直接使用 Manim 命令
```bash
# 如果您熟悉 Manim，也可以直接使用
manim -p -ql bspline_demo.py BSplineDemo
```

## 📺 演示内容

### 1. BSplineDemo - 基础演示
- **时长**：约 3-4 分钟
- **内容**：
  - B-样条曲线的基本概念介绍
  - 控制点和控制多边形的展示
  - B-样条曲线的生成过程
  - 控制点移动对曲线的影响
  - B-样条的局部性质演示
  - 数学公式展示

### 2. BSplineBasicFunctions - 基函数演示
- **时长**：约 2-3 分钟
- **内容**：
  - B-样条基函数的可视化
  - 不同基函数的局部支撑特性
  - 基函数与曲线局部性的关系

### 3. BSplineInteractive - 交互式演示
- **时长**：约 2-3 分钟
- **内容**：
  - 动态变化的控制点
  - 实时更新的 B-样条曲线
  - 展示曲线的连续性和平滑性

### 4. BSplineComplete - 完整演示
- **时长**：约 8-10 分钟
- **内容**：包含上述所有场景的完整演示

## 🛠️ 技术实现

### 核心技术
- **Manim Community**: 用于动画制作和渲染
- **SciPy**: 提供 B-样条数学计算支持
- **NumPy**: 数值计算和数组操作

### 主要特性
- 使用 `scipy.interpolate.BSpline` 进行精确的 B-样条计算
- 利用 Manim 的 `VMobject` 和 `ParametricFunction` 进行曲线渲染
- 通过 `always_redraw` 和 `ValueTracker` 实现动态更新
- 支持不同阶数的 B-样条曲线

## 📁 文件结构

```
.
├── bspline_demo.py          # 主要演示代码
├── run_bspline_demo.py      # 运行脚本
├── README.md                # 说明文档
└── media/                   # 渲染输出目录（自动创建）
    └── videos/
        └── bspline_demo/
            ├── 480p15/      # 低质量视频
            ├── 720p30/      # 中等质量视频
            └── 1080p60/     # 高质量视频
```

## 🎨 自定义和扩展

### 修改控制点
在 `bspline_demo.py` 中修改 `control_points` 列表：
```python
control_points = [
    np.array([-4, -1, 0]),
    np.array([-2, 2, 0]),
    # 添加更多控制点...
]
```

### 调整 B-样条参数
修改 `create_bspline_curve` 函数中的参数：
```python
def create_bspline_curve(points, degree=3, num_points=100):
    # degree: B-样条的阶数
    # num_points: 曲线上采样点的数量
```

### 添加新的演示场景
创建新的 Scene 类并继承相关功能：
```python
class MyCustomBSplineDemo(Scene):
    def construct(self):
        # 您的自定义演示代码
        pass
```

## 🔧 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'manim'**
   ```bash
   pip install manim
   ```

2. **ModuleNotFoundError: No module named 'scipy'**
   ```bash
   pip install scipy
   ```

3. **渲染速度慢**
   - 使用预览模式：`--preview`
   - 降低质量：`--quality low`

4. **视频无法播放**
   - 检查 media/videos/ 目录中的输出文件
   - 确保您的视频播放器支持 MP4 格式

### 性能优化

- **快速预览**：使用 `-ql` (低质量) 参数
- **最终渲染**：使用 `-qh` (高质量) 参数
- **调试模式**：添加 `--dry_run` 参数检查场景而不渲染

## 📚 学习资源

### B-样条相关
- [B-样条曲线基础理论](https://en.wikipedia.org/wiki/B-spline)
- [NURBS 和 B-样条在 CAD 中的应用](https://www.cadtutor.net/tutorials/nurbs/)

### Manim 相关
- [Manim Community 官方文档](https://docs.manim.community/)
- [Manim 教程和示例](https://docs.manim.community/en/stable/examples.html)

## 🤝 贡献

欢迎提交问题报告、功能请求或改进建议！

### 贡献方式
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🙏 致谢

- **Manim Community** - 提供优秀的数学动画框架
- **SciPy** - 提供强大的科学计算支持
- **3Blue1Brown** - 启发了数学可视化的理念

---

**享受您的 B-样条曲线学习之旅！** 🎓✨ 