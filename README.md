[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/j3qjeOEc)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=18891628)
# 计算物理实验 - 第五周作业

本仓库包含计算物理实践课程第五周的编程作业，涵盖多个随机过程的数值模拟与数据分析。

## 目录结构
```
cp2025-practice-week5/
├── src/                                 # 源代码目录（学生实现）
│   ├── waiting_times.py                 # 等待时间分布分析
│   ├── random_walk.py                   # 随机行走轨迹模拟
│   ├── endpoints_analysis.py            # 随机行走终点分布分析
│   ├── mean_square_displacement.py      # 随机行走均方位移分析
│   ├── displacement_distribution.py     # 随机行走位移分布分析
│   └── poisson_simulation.py            # 泊松分布数值模拟
├── results/                             # 结果目录（学生填写）
│   ├── 等待时间分布实验报告.md            # 等待时间分布实验报告
│   ├── 随机行走轨迹实验报告.md            # 随机行走轨迹实验报告
│   ├── 随机行走终点分布实验报告.md         # 随机行走终点分布实验报告
│   ├── 随机行走与均方位移之间关系实验报告.md # 均方位移实验报告
│   ├── 随机行走位移分布实验报告.md         # 位移分布实验报告
│   └── 泊松分布数值模拟实验报告.md         # 泊松分布实验报告
├── solutions/                           # 参考解答目录
│   ├── waiting_times_solution.py        # 等待时间分布参考解答
│   ├── random_walk_solution.py          # 随机行走轨迹参考解答
│   ├── endpoints_analysis_solution.py    # 随机行走终点分布参考解答
│   ├── mean_square_displacement_solution.py # 均方位移参考解答
│   ├── displacement_distribution_solution.py # 位移分布参考解答
│   └── poisson_simulation_solution.py   # 泊松分布参考解答
├── tests/                               # 测试文件目录
│   ├── test_waiting_times.py            # 等待时间分布测试
│   ├── test_random_walk.py              # 随机行走轨迹测试
│   ├── test_endpoints_analysis.py       # 随机行走终点分布测试
│   ├── test_mean_square_displacement.py # 均方位移测试
│   ├── test_displacement_distribution.py # 位移分布测试
│   └── test_poisson_simulation.py       # 泊松分布测试
├── docs/                                # 文档目录
│   ├── 等待时间分布.md                   # 等待时间分布实验说明
│   ├── 随机行走轨迹.md                   # 随机行走轨迹实验说明
│   ├── 绘制随机行走终点图像-项目说明.md     # 随机行走终点分布实验说明
│   ├── 随机行走与均方位移之间的关系.md      # 均方位移实验说明
│   ├── 随机行走位移分布.md                # 位移分布实验说明
│   └── 泊松分布的数值模拟.md              # 泊松分布实验说明
├── requirements.txt                     # 项目依赖
└── README.md                            # 本文件
```

## 作业内容

本次作业包含六个独立的随机过程模拟实验：

1. **等待时间分布分析**：研究稀有事件之间的等待时间分布。
2. **随机行走轨迹模拟**：实现二维随机行走，可视化其轨迹。
3. **随机行走终点分布**：分析多次随机行走的终点分布特性。
4. **随机行走与均方位移**：探究均方位移与步数之间的关系。
5. **随机行走位移分布**：分析随机行走的位移统计特性。
6. **泊松分布数值模拟**：模拟泊松过程并与理论分布比较。

## 使用说明

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_waiting_times.py -v
```
### 提交作业
1. 完成 src/ 目录下的所有实现文件
2. 确保所有测试通过
3. 提交到GitHub仓库
4. 在 results/ 目录下对应的markdown文件中上传程序运行结果，包括生成的图像文件和结果讨论
## 评分标准
每个实验模块占总分的六分之一，总分为60分。评分将通过自动测试完成，测试通过即得满分。

## 参考资料
- 《Python物理建模初学者指南》第六章