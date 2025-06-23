# 02 代码实现

## 模块概述

本模块提供A*算法的Python实现，包含完整的算法逻辑、详细的代码注释和实用的测试用例。

## 文件说明

### astar_basic.py
- **功能**: A*算法的基础实现
- **特点**:
  - 面向对象设计
  - 支持多种启发函数
  - 详细的代码注释
  - 内置测试用例

## 核心类和方法

### Node类
```python
@dataclass
class Node:
    x: int          # X坐标
    y: int          # Y坐标  
    g: float        # 从起点的实际代价
    h: float        # 启发函数值
    f: float        # 总评估代价
    parent: Node    # 父节点
```

### AStar类
```python
class AStar:
    def __init__(self, grid)                    # 初始化网格
    def find_path(self, start, goal)            # 主搜索方法
    def heuristic(self, node, goal, method)     # 启发函数
    def get_neighbors(self, node)               # 获取邻居节点
    def reconstruct_path(self, goal_node)       # 重构路径
```

## 支持的启发函数

1. **欧几里得距离 (euclidean)**
   ```
   h(n) = √[(x₂-x₁)² + (y₂-y₁)²]
   ```
   - 适用于: 任意方向移动
   - 特点: 最精确的距离估计

2. **曼哈顿距离 (manhattan)**
   ```
   h(n) = |x₂-x₁| + |y₂-y₁|
   ```
   - 适用于: 四方向移动
   - 特点: 计算简单，适合网格环境

3. **对角距离 (diagonal)**
   ```
   h(n) = max(|x₂-x₁|, |y₂-y₁|)
   ```
   - 适用于: 八方向移动
   - 特点: 考虑对角线移动

## 运行示例

### 基本使用
```python
from astar_basic import AStar, create_test_grid

# 创建网格和算法实例
grid = create_test_grid()
astar = AStar(grid)

# 搜索路径
start = (0, 0)
goal = (9, 9)
path = astar.find_path(start, goal)

# 显示结果
if path:
    print(f"找到路径，长度: {len(path)}")
    cost = astar.get_path_cost(path)
    print(f"路径代价: {cost:.2f}")
```

### 测试不同启发函数
```python
heuristics = ['euclidean', 'manhattan', 'diagonal']

for heuristic in heuristics:
    path = astar.find_path(start, goal, heuristic)
    if path:
        print(f"{heuristic}: 路径长度 {len(path)}")
```

## 运行方法

### 直接运行
```bash
python astar_basic.py
```

### 交互式使用
```python
python
>>> from astar_basic import AStar
>>> # 使用算法...
```

## 预期输出

```
=== A*算法测试 ===
起点: (0, 0)
终点: (9, 9)

--- 使用 euclidean 启发函数 ---
路径搜索完成！探索了 42 个节点
路径长度: 13 步
路径代价: 17.31

网格地图 (S:起点, G:终点, *:路径, 1:障碍物, 0:空地):
S 0 0 0 0 0 0 0 0 0
0 1 1 * 0 0 1 1 1 0
0 0 1 * 0 0 0 0 1 0
0 0 0 * 1 1 0 0 0 0
0 0 0 * 1 1 0 0 0 0
0 0 0 * * * * * 0 0
0 0 1 1 0 0 1 * 0 0
0 0 0 1 0 0 1 * 0 0
0 0 0 0 0 0 0 * 0 0
0 0 0 0 0 0 0 * * G
```

## 自定义网格

### 创建自定义网格
```python
# 0: 可通行, 1: 障碍物
custom_grid = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
]

astar = AStar(custom_grid)
path = astar.find_path((0, 0), (4, 4))
```

## 性能特点

- **时间复杂度**: O(b^d)，其中b是分支因子，d是解的深度
- **空间复杂度**: O(b^d)，需要存储开放和关闭列表
- **最优性**: 在启发函数可接受的条件下保证找到最优解

## 使用注意事项

1. **网格表示**: 使用0表示可通行，1表示障碍物
2. **坐标系统**: 使用(row, col)格式，从(0,0)开始
3. **路径格式**: 返回路径为坐标列表，从起点到终点
4. **启发函数选择**: 根据移动规则选择合适的启发函数

## 扩展建议

1. 添加更多启发函数
2. 支持不同地形的移动代价
3. 实现路径平滑算法
4. 添加动态障碍物支持

## 下一步

学习完基础实现后，可以进入：
- `03_可视化演示`: 查看算法搜索过程
- `04_3D应用`: 学习三维空间应用
- `06_高级特性`: 探索算法优化技术 