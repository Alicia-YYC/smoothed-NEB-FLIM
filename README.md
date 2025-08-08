# 非参数贝叶斯FLIM分析实验包

本实验包包含了复现论文"Non-parametric Bayesian FLIM Analysis"中所有实验的完整代码和文档。

## 文件说明

### 核心代码文件
- `paper_experiment_reproduction_fixed.py` - 主要的实验复现代码，包含三个实验：
  - 实验1：先验分布估计性能评估
  - 实验2：像素级寿命恢复性能比较
  - 实验3：计算效率评估
- `nonparametric_bayesian_flim.py` - 非参数贝叶斯FLIM分析的核心实现
- `example_usage.py` - 使用示例和演示代码

### 配置文件
- `config_nonparametric_flim.json` - 分析参数配置文件
- `requirements.txt` - Python依赖包列表

### 文档和结果
- `README_nonparametric_flim.md` - 详细的技术文档
- `paper_experiment_results_fixed.png` - 实验结果的图表

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行实验
```bash
python paper_experiment_reproduction_fixed.py
```

### 3. 查看示例
```bash
python example_usage.py
```

## 实验参数

### 实验设置
- 图像大小：32×32像素
- 时间范围：10 ns (10000 ps)
- 时间通道数：256
- 背景光子比例：0.001
- IRF：高斯分布，均值1500 ps，标准差100 ps

### 实验1：先验分布估计
- 光子数范围：[10, 32, 100, 316, 1000]
- 区间数范围：[400, 600, 800]
- 重复次数：3次
- 评估指标：L2距离

### 实验2：像素级恢复
- 光子数范围：[100, 316, 1000, 3162]
- 重复次数：3次
- 评估指标：均方误差(MSE)

### 实验3：计算效率
- 图像大小：[16×16, 32×32, 64×64]
- 光子数：1000/像素
- 评估指标：计算时间

## 算法实现

### 非参数贝叶斯FLIM分析
1. **双指数衰减模拟**：生成包含背景的photon histogram数据
2. **概率库构建**：构建可能寿命的概率库P(τ)
3. **NPMLE估计**：非参数最大似然估计获取先验分布
4. **EM-MAP估计**：期望最大化最大后验估计恢复像素参数

### 对比方法
- **像素级分析**：仅使用单个像素的光子进行拟合
- **全局分析**：全局估计两个组分的寿命，然后估计每个像素的组分贡献
- **NEB-FLIM**：经验贝叶斯分析，结合局部和全局信息

## 结果说明

实验结果保存在`paper_experiment_results_fixed.png`中，包含：
- 不同光子数和区间数下的先验分布估计性能
- 像素级寿命恢复性能比较
- 计算效率对比

## 技术细节

### 数值稳定性改进
- 限制信号值范围避免`lam value too large`错误
- 逐通道生成Poisson样本提高数值稳定性
- 使用非交互式matplotlib后端避免显示问题

### 性能优化
- 减少实验参数范围以加快调试
- 使用try-except块处理异常情况
- 优化内存使用和计算效率

## 注意事项

1. 确保Python环境中有足够的计算资源
2. 实验可能需要较长时间运行，建议在后台执行
3. 如果遇到数值问题，可以调整参数范围
4. 结果图表会自动保存，无需手动显示

## 引用

本实验包基于以下论文实现：
"Non-parametric Bayesian FLIM Analysis" - Biomedical Optics Express, Vol. 10, No. 11
