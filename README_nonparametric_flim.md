# 非参数贝叶斯荧光寿命分析 (Nonparametric Bayesian FLIM Analysis)

## 概述

这个项目实现了非参数贝叶斯荧光寿命分析框架，包含以下四个主要步骤：

1. **模拟双指数衰减光子直方图数据（加背景）**
2. **构建可能寿命的概率库 P(τ)**
3. **进行非参数最大似然估计（NPMLE）获取先验分布**
4. **对每个像素用 EM 进行最大后验估计（MAP）恢复参数**

## 文件结构

```
├── nonparametric_bayesian_flim.py    # 主分析类
├── config_nonparametric_flim.json    # 配置文件
├── example_usage.py                   # 使用示例
├── README_nonparametric_flim.md      # 说明文档
└── nonparametric_flim_results/       # 输出目录（自动创建）
```

## 安装依赖

```bash
pip install numpy matplotlib scipy pandas
```

## 快速开始

### 1. 基本使用

```python
from nonparametric_bayesian_flim import NonparametricBayesianFLIM

# 初始化分析器
analyzer = NonparametricBayesianFLIM(
    tau_range=(0.1, 10.0),      # 寿命范围 (ns)
    num_tau_points=50,          # 寿命空间点数
    time_range=12.5,            # 时间范围 (ns)
    num_channels=256            # 时间通道数
)

# 生成训练数据
training_histograms = analyzer.generate_training_data(num_samples=500)

# 训练先验分布
prior = analyzer.train_prior(training_histograms)

# 生成测试数据
test_histogram = analyzer.simulate_dual_exponential_histogram(
    tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
)

# 分析数据
result = analyzer.analyze_pixel(test_histogram)

# 显示结果
print(f"平均寿命: {result['mean_tau']:.3f} ns")
print(f"显著成分数: {result['num_components']}")
```

### 2. 运行完整示例

```bash
python example_usage.py
```

这将运行以下示例：
- 基本分析示例
- 参数研究示例
- 图像分析示例
- 与传统方法比较示例

## 主要功能

### NonparametricBayesianFLIM 类

#### 初始化参数
- `tau_range`: 寿命范围 (ns)
- `num_tau_points`: 寿命空间中的点数
- `time_range`: 时间范围 (ns)
- `num_channels`: 时间通道数

#### 主要方法

##### 数据生成
- `simulate_dual_exponential_histogram()`: 模拟双指数衰减直方图
- `generate_training_data()`: 生成训练数据

##### 先验训练
- `build_probability_library()`: 构建寿命概率库
- `npmle_estimation()`: 非参数最大似然估计
- `train_prior()`: 训练先验分布

##### 数据分析
- `em_map_estimation()`: EM-MAP 估计
- `analyze_pixel()`: 分析单个像素
- `analyze_image()`: 分析整个图像

##### 可视化
- `plot_prior_distribution()`: 绘制先验分布
- `plot_analysis_result()`: 绘制分析结果

## 配置说明

### config_nonparametric_flim.json

```json
{
  "analysis_parameters": {
    "tau_range": [0.1, 10.0],           // 寿命范围
    "num_tau_points": 50,               // 寿命空间点数
    "time_range": 12.5,                 // 时间范围
    "num_channels": 256,                // 时间通道数
    "convergence_tolerance": 1e-6,      // 收敛容差
    "max_iterations": 100,              // 最大迭代次数
    "significance_threshold": 0.01      // 显著性阈值
  },
  
  "simulation_parameters": {
    "training_samples": 500,            // 训练样本数
    "tau1_range": [1.5, 3.0],          // 长寿命范围
    "tau2_range": [0.5, 1.2],          // 短寿命范围
    "fraction1_range": [0.4, 0.8],     // 比例范围
    "total_photons_range": [50000, 200000], // 光子数范围
    "background_ratio_range": [0.05, 0.2],  // 背景比例范围
    "noise_level": 0.05                // 噪声水平
  }
}
```

## 算法原理

### 1. 非参数最大似然估计 (NPMLE)

NPMLE 用于从训练数据中学习寿命的先验分布：

```python
# 目标函数：最大化对数似然
def objective_function(weights):
    mixed_distribution = sum(weight_i * exp(-t/tau_i) for i, weight_i in enumerate(weights))
    return -sum(log_likelihood(histogram, mixed_distribution) for histogram in training_data)
```

### 2. 期望最大化-最大后验估计 (EM-MAP)

EM-MAP 算法用于估计单个像素的寿命分布：

#### E步（期望步骤）
```python
responsibilities[i,j] = weights[i] * exp(-time[j]/tau[i]) / sum(weights[k] * exp(-time[j]/tau[k]))
```

#### M步（最大化步骤）
```python
new_weights[i] = sum(responsibilities[i,j] * histogram[j]) * prior[i]
```

## 输出结果

### 分析结果格式

```python
result = {
    'posterior_weights': array,           # 后验权重分布
    'significant_components': [           # 显著成分列表
        {
            'tau': 2.14,                  # 寿命值
            'weight': 0.6,                # 权重
            'index': 15                   # 索引
        }
    ],
    'mean_tau': 1.85,                    # 平均寿命
    'num_components': 2                   # 显著成分数
}
```

### 输出文件

- `prior_distribution.png`: 先验分布图
- `analysis_result.png`: 分析结果图
- `image_analysis_result.png`: 图像分析结果
- `comparison_result.png`: 与传统方法比较

## 优势特点

### 1. 非参数性
- 不需要假设特定的寿命分布形式
- 能够适应复杂的多成分荧光系统

### 2. 鲁棒性
- 对噪声和背景更鲁棒
- 能够处理低信噪比数据

### 3. 自适应性
- 通过训练数据自动学习先验分布
- 能够适应不同的实验条件

### 4. 不确定性量化
- 提供参数估计的不确定性
- 后验分布反映估计的置信度

## 应用场景

### 1. 复杂生物样本
- 具有多种荧光寿命成分的样本
- 环境异质性较大的样本

### 2. 低信噪比数据
- 光子数较少的实验
- 背景噪声较大的情况

### 3. 高分辨率成像
- 需要像素级的精确分析
- 空间异质性分析

## 与传统方法比较

| 方面 | 传统最小二乘法 | 非参数贝叶斯方法 |
|------|----------------|------------------|
| **参数假设** | 固定成分数 | 自适应成分数 |
| **噪声处理** | 对噪声敏感 | 鲁棒性更强 |
| **不确定性** | 点估计 | 概率分布 |
| **先验知识** | 不使用 | 自动学习 |
| **计算复杂度** | 低 | 中等 |

## 注意事项

### 1. 计算时间
- 训练阶段需要较长时间（取决于训练样本数）
- 分析阶段相对较快

### 2. 内存使用
- 大量训练样本可能占用较多内存
- 建议根据系统配置调整参数

### 3. 参数选择
- `num_tau_points`: 影响分辨率和计算时间
- `significance_threshold`: 影响成分检测灵敏度

## 扩展功能

### 1. 添加新的噪声模型
```python
def custom_noise_model(self, histogram, noise_type):
    # 实现自定义噪声模型
    pass
```

### 2. 支持更多荧光模型
```python
def multi_exponential_model(self, taus, weights):
    # 实现多指数模型
    pass
```

### 3. 集成仪器响应函数
```python
def add_irf_convolution(self, histogram, irf):
    # 添加IRF卷积
    pass
```

## 参考文献

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Laird, N. M. (1978). Nonparametric maximum likelihood estimation of a mixing distribution.
3. Bayesian methods in fluorescence lifetime imaging microscopy.

## 联系方式

如有问题或建议，请联系开发团队。

## 许可证

本项目采用 MIT 许可证。
