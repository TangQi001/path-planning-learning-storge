# 01_理论基础

## 概述

本部分深入介绍Voronoi图法路径规划的理论基础，包括：

- Voronoi图的数学定义和几何性质
- 路径规划中的应用原理
- 算法的优势与局限性
- 与其他路径规划方法的关系

## 学习路径

```mermaid
graph LR
    A[基本概念] --> B[Voronoi图理论]
    B --> C[路径规划应用]
    C --> D[算法分析]
    D --> E[实际案例]
```

## 文件说明

- **basic_concepts.md**: Voronoi图的基本概念和定义
- **voronoi_theory.md**: 深入的数学理论和几何性质
- **path_planning_principles.md**: 在路径规划中的应用原理

## 核心公式

Voronoi单元的数学定义：
$$V(p_i) = \{x \in \mathbb{R}^2 | d(x, p_i) \leq d(x, p_j) \text{ for all } j \neq i\}$$

其中：
- $p_i$ 是第i个种子点（障碍物）
- $d(x, p_i)$ 是点x到种子点$p_i$的欧氏距离
- $V(p_i)$ 是以$p_i$为中心的Voronoi单元

## 下一步

完成本部分学习后，请进入 `02_代码实现/` 查看具体的算法实现。 