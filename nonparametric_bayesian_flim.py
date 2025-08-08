"""
非参数贝叶斯荧光寿命分析 (Nonparametric Bayesian FLIM Analysis)

功能：
1. 模拟双指数衰减光子直方图数据（加背景）
2. 构建可能寿命的概率库 P(τ)
3. 进行非参数最大似然估计（NPMLE）获取先验分布
4. 对每个像素用 EM 进行最大后验估计（MAP）恢复参数

作者：AI Assistant
日期：2025-01-27
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import itertools
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NonparametricBayesianFLIM:
    """
    非参数贝叶斯荧光寿命分析类
    """
    
    def __init__(self, tau_range: Tuple[float, float] = (0.1, 10.0), 
                 num_tau_points: int = 100, time_range: float = 12.5, 
                 num_channels: int = 256):
        """
        初始化参数
        
        Args:
            tau_range: 寿命范围 (ns)
            num_tau_points: 寿命空间中的点数
            time_range: 时间范围 (ns)
            num_channels: 时间通道数
        """
        self.tau_range = tau_range
        self.num_tau_points = num_tau_points
        self.time_range = time_range
        self.num_channels = num_channels
        
        # 构建寿命空间（对数分布）
        self.tau_space = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), num_tau_points)
        
        # 构建时间通道
        self.time_channels = np.linspace(0, time_range, num_channels + 1)
        self.time_channel_centers = np.linspace(0, time_range, num_channels)
        
        # 初始化先验分布
        self.prior_distribution = None
        
    def simulate_dual_exponential_histogram(self, tau1: float, tau2: float, 
                                          fraction1: float, total_photons: int = 100000,
                                          background_ratio: float = 0.1,
                                          noise_level: float = 0.05) -> np.ndarray:
        """
        模拟双指数衰减光子直方图
        
        Args:
            tau1: 第一个寿命成分 (ns)
            tau2: 第二个寿命成分 (ns)
            fraction1: 第一个成分的比例
            total_photons: 总光子数
            background_ratio: 背景光子比例
            noise_level: 噪声水平
            
        Returns:
            光子直方图数组
        """
        # 计算理想的双指数衰减
        ideal_decay = (fraction1 * np.exp(-self.time_channel_centers / tau1) + 
                      (1 - fraction1) * np.exp(-self.time_channel_centers / tau2))
        
        # 归一化
        ideal_decay = ideal_decay / np.sum(ideal_decay)
        
        # 分配光子数
        signal_photons = int(total_photons * (1 - background_ratio))
        background_photons = int(total_photons * background_ratio)
        
        # 生成信号光子
        signal_histogram = np.random.poisson(ideal_decay * signal_photons)
        
        # 生成背景光子（均匀分布）
        background_histogram = np.random.poisson(background_photons / self.num_channels, 
                                               size=self.num_channels)
        
        # 合并信号和背景
        total_histogram = signal_histogram + background_histogram
        
        # 添加额外噪声
        noise = np.random.normal(0, noise_level * np.sqrt(total_histogram))
        total_histogram = np.maximum(0, total_histogram + noise).astype(int)
        
        return total_histogram
    
    def generate_training_data(self, num_samples: int = 1000) -> List[np.ndarray]:
        """
        生成训练数据用于构建先验分布
        
        Args:
            num_samples: 样本数量
            
        Returns:
            训练直方图列表
        """
        print(f"生成 {num_samples} 个训练样本...")
        
        training_histograms = []
        
        for i in range(num_samples):
            # 随机化参数
            tau1 = np.random.uniform(1.5, 3.0)  # 长寿命范围
            tau2 = np.random.uniform(0.5, 1.2)  # 短寿命范围
            fraction1 = np.random.uniform(0.4, 0.8)  # 比例范围
            total_photons = np.random.uniform(50000, 200000)  # 光子数范围
            background_ratio = np.random.uniform(0.05, 0.2)  # 背景比例范围
            
            # 生成直方图
            histogram = self.simulate_dual_exponential_histogram(
                tau1, tau2, fraction1, int(total_photons), background_ratio
            )
            
            training_histograms.append(histogram)
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1} 个样本")
        
        return training_histograms
    
    def build_probability_library(self, histograms: List[np.ndarray]) -> np.ndarray:
        """
        构建寿命概率库
        
        Args:
            histograms: 训练直方图列表
            
        Returns:
            寿命概率数组
        """
        print("构建寿命概率库...")
        
        tau_probabilities = np.zeros(self.num_tau_points)
        
        for i, histogram in enumerate(histograms):
            # 对每个直方图进行拟合
            fitted_taus = self.fit_histogram_to_taus(histogram)
            
            # 更新概率库
            for tau in fitted_taus:
                # 找到最近的寿命点
                tau_idx = np.argmin(np.abs(self.tau_space - tau))
                tau_probabilities[tau_idx] += 1
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 个直方图")
        
        # 归一化
        tau_probabilities = tau_probabilities / np.sum(tau_probabilities)
        
        return tau_probabilities
    
    def fit_histogram_to_taus(self, histogram: np.ndarray, max_components: int = 3) -> List[float]:
        """
        对直方图拟合多个可能的寿命
        
        Args:
            histogram: 光子直方图
            max_components: 最大成分数
            
        Returns:
            拟合的寿命列表
        """
        best_taus = []
        
        # 尝试不同数量的成分
        for num_components in range(1, max_components + 1):
            # 从寿命空间中采样组合
            tau_combinations = list(itertools.combinations(self.tau_space, num_components))
            
            best_likelihood = -np.inf
            best_tau_combo = None
            
            # 限制组合数量以避免计算过载
            max_combinations = min(100, len(tau_combinations))
            if len(tau_combinations) > max_combinations:
                selected_indices = np.random.choice(len(tau_combinations), max_combinations, replace=False)
                selected_combinations = [tau_combinations[i] for i in selected_indices]
            else:
                selected_combinations = tau_combinations
            
            for tau_combo in selected_combinations:
                # 拟合这个寿命组合
                likelihood = self.calculate_likelihood(histogram, tau_combo)
                
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_tau_combo = tau_combo
            
            if best_tau_combo is not None:
                best_taus.extend(best_tau_combo)
        
        return best_taus
    
    def calculate_likelihood(self, histogram: np.ndarray, taus: Tuple[float, ...]) -> float:
        """
        计算给定寿命组合的似然
        
        Args:
            histogram: 光子直方图
            taus: 寿命组合
            
        Returns:
            对数似然
        """
        # 构建模型
        model = np.zeros(self.num_channels)
        
        # 假设等权重
        weight = 1.0 / len(taus)
        
        for tau in taus:
            exponential = np.exp(-self.time_channel_centers / tau)
            model += weight * exponential
        
        # 归一化
        model = model / np.sum(model)
        
        # 缩放模型以匹配总光子数
        total_photons = np.sum(histogram)
        model = model * total_photons
        
        # 计算泊松似然
        log_likelihood = np.sum(poisson.logpmf(histogram, model))
        
        return log_likelihood
    
    def npmle_estimation(self, histograms: List[np.ndarray], 
                        initial_probabilities: np.ndarray) -> np.ndarray:
        """
        非参数最大似然估计
        
        Args:
            histograms: 训练直方图列表
            initial_probabilities: 初始概率分布
            
        Returns:
            优化后的概率分布
        """
        print("进行非参数最大似然估计...")
        
        def objective_function(weights):
            """目标函数：负对数似然"""
            # 构建混合分布
            mixed_distribution = np.zeros(self.num_channels)
            
            for i, weight in enumerate(weights):
                tau = self.tau_space[i]
                exponential = np.exp(-self.time_channel_centers / tau)
                mixed_distribution += weight * exponential
            
            # 归一化
            mixed_distribution = mixed_distribution / np.sum(mixed_distribution)
            
            # 计算对数似然
            log_likelihood = 0
            for histogram in histograms:
                # 缩放模型
                total_photons = np.sum(histogram)
                model = mixed_distribution * total_photons
                
                # 泊松似然
                log_likelihood += np.sum(poisson.logpmf(histogram, model))
            
            return -log_likelihood
        
        # 约束条件：权重和为1，权重非负
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0, 1)] * self.num_tau_points
        
        # 初始猜测
        initial_weights = initial_probabilities.copy()
        
        # 优化
        result = minimize(objective_function, initial_weights, 
                         constraints=constraints, bounds=bounds,
                         method='SLSQP', options={'maxiter': 1000})
        
        if result.success:
            print("NPMLE 优化成功")
            return result.x
        else:
            print("NPMLE 优化失败，使用初始概率")
            return initial_probabilities
    
    def em_map_estimation(self, pixel_histogram: np.ndarray, 
                         max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        期望最大化-最大后验估计
        
        Args:
            pixel_histogram: 像素直方图
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            后验权重分布
        """
        if self.prior_distribution is None:
            raise ValueError("请先训练先验分布")
        
        # 初始化权重
        current_weights = np.ones(self.num_tau_points) / self.num_tau_points
        
        for iteration in range(max_iterations):
            # E步：计算期望
            responsibilities = self.expectation_step(pixel_histogram, current_weights)
            
            # M步：最大化后验
            new_weights = self.maximization_step(responsibilities, pixel_histogram)
            
            # 检查收敛
            if np.allclose(current_weights, new_weights, rtol=tolerance):
                break
                
            current_weights = new_weights
        
        return current_weights
    
    def expectation_step(self, histogram: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        期望步骤
        
        Args:
            histogram: 光子直方图
            weights: 当前权重
            
        Returns:
            责任矩阵
        """
        responsibilities = np.zeros((self.num_tau_points, self.num_channels))
        
        for i, tau in enumerate(self.tau_space):
            exponential = np.exp(-self.time_channel_centers / tau)
            responsibilities[i] = weights[i] * exponential
        
        # 归一化
        responsibilities = responsibilities / np.sum(responsibilities, axis=0, keepdims=True)
        
        return responsibilities
    
    def maximization_step(self, responsibilities: np.ndarray, histogram: np.ndarray) -> np.ndarray:
        """
        最大化步骤（包含先验）
        
        Args:
            responsibilities: 责任矩阵
            histogram: 光子直方图
            
        Returns:
            新的权重分布
        """
        # 计算后验权重
        posterior_weights = np.sum(responsibilities * histogram[np.newaxis, :], axis=1)
        
        # 结合先验
        posterior_weights = posterior_weights * self.prior_distribution
        
        # 归一化
        posterior_weights = posterior_weights / np.sum(posterior_weights)
        
        return posterior_weights
    
    def extract_significant_components(self, posterior_weights: np.ndarray, 
                                     threshold: float = 0.01) -> List[Dict]:
        """
        提取显著成分
        
        Args:
            posterior_weights: 后验权重
            threshold: 显著性阈值
            
        Returns:
            显著成分列表
        """
        significant_components = []
        
        # 找到超过阈值的成分
        significant_indices = np.where(posterior_weights > threshold)[0]
        
        for idx in significant_indices:
            component = {
                'tau': self.tau_space[idx],
                'weight': posterior_weights[idx],
                'index': idx
            }
            significant_components.append(component)
        
        # 按权重排序
        significant_components.sort(key=lambda x: x['weight'], reverse=True)
        
        return significant_components
    
    def train_prior(self, training_histograms: List[np.ndarray]) -> np.ndarray:
        """
        训练先验分布
        
        Args:
            training_histograms: 训练直方图列表
            
        Returns:
            先验分布
        """
        print("开始训练先验分布...")
        
        # 步骤1：构建概率库
        tau_probabilities = self.build_probability_library(training_histograms)
        
        # 步骤2：NPMLE估计
        self.prior_distribution = self.npmle_estimation(training_histograms, tau_probabilities)
        
        print("先验分布训练完成")
        return self.prior_distribution
    
    def analyze_pixel(self, pixel_histogram: np.ndarray) -> Dict:
        """
        分析单个像素
        
        Args:
            pixel_histogram: 像素直方图
            
        Returns:
            分析结果
        """
        # 使用EM-MAP估计
        posterior_weights = self.em_map_estimation(pixel_histogram)
        
        # 提取主要寿命成分
        significant_components = self.extract_significant_components(posterior_weights)
        
        # 计算平均寿命
        mean_tau = np.sum(posterior_weights * self.tau_space)
        
        result = {
            'posterior_weights': posterior_weights,
            'significant_components': significant_components,
            'mean_tau': mean_tau,
            'num_components': len(significant_components)
        }
        
        return result
    
    def analyze_image(self, image_histograms: List[np.ndarray]) -> List[Dict]:
        """
        分析整个图像
        
        Args:
            image_histograms: 图像直方图列表
            
        Returns:
            分析结果列表
        """
        print(f"分析 {len(image_histograms)} 个像素...")
        
        results = []
        for i, pixel_histogram in enumerate(image_histograms):
            pixel_result = self.analyze_pixel(pixel_histogram)
            results.append(pixel_result)
            
            if (i + 1) % 100 == 0:
                print(f"已分析 {i + 1} 个像素")
        
        return results
    
    def plot_prior_distribution(self, save_path: Optional[str] = None):
        """
        绘制先验分布
        
        Args:
            save_path: 保存路径
        """
        if self.prior_distribution is None:
            print("请先训练先验分布")
            return
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.tau_space, self.prior_distribution, 'b-', linewidth=2)
        plt.xlabel('Lifetime (ns)')
        plt.ylabel('Prior Probability')
        plt.title('Nonparametric Prior Distribution')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_analysis_result(self, pixel_histogram: np.ndarray, result: Dict, 
                           save_path: Optional[str] = None):
        """
        绘制分析结果
        
        Args:
            pixel_histogram: 像素直方图
            result: 分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始数据
        axes[0, 0].plot(self.time_channel_centers, pixel_histogram, 'ko-', markersize=3)
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('Photon Counts')
        axes[0, 0].set_title('Original Histogram')
        axes[0, 0].grid(True)
        
        # 后验分布
        axes[0, 1].semilogx(self.tau_space, result['posterior_weights'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Lifetime (ns)')
        axes[0, 1].set_ylabel('Posterior Weight')
        axes[0, 1].set_title('Posterior Distribution')
        axes[0, 1].grid(True)
        
        # 显著成分
        if result['significant_components']:
            taus = [comp['tau'] for comp in result['significant_components']]
            weights = [comp['weight'] for comp in result['significant_components']]
            axes[1, 0].bar(range(len(taus)), weights)
            axes[1, 0].set_xlabel('Component Index')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Significant Components')
            axes[1, 0].set_xticks(range(len(taus)))
            axes[1, 0].set_xticklabels([f'{tau:.2f}ns' for tau in taus])
        
        # 拟合结果
        model = np.zeros(self.num_channels)
        for comp in result['significant_components']:
            tau = comp['tau']
            weight = comp['weight']
            exponential = np.exp(-self.time_channel_centers / tau)
            model += weight * exponential
        
        # 归一化和缩放
        model = model / np.sum(model)
        total_photons = np.sum(pixel_histogram)
        model = model * total_photons
        
        axes[1, 1].plot(self.time_channel_centers, pixel_histogram, 'ko-', markersize=3, label='Data')
        axes[1, 1].plot(self.time_channel_centers, model, 'r-', linewidth=2, label='Fit')
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Photon Counts')
        axes[1, 1].set_title(f'Fit Result (Mean τ = {result["mean_tau"]:.2f}ns)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """
    主函数：演示非参数贝叶斯荧光寿命分析
    """
    print("=== 非参数贝叶斯荧光寿命分析演示 ===\n")
    
    # 1. 初始化分析器
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=50,
        time_range=12.5,
        num_channels=256
    )
    
    # 2. 生成训练数据
    training_histograms = analyzer.generate_training_data(num_samples=500)
    
    # 3. 训练先验分布
    prior = analyzer.train_prior(training_histograms)
    
    # 4. 绘制先验分布
    analyzer.plot_prior_distribution(save_path='prior_distribution.png')
    
    # 5. 生成测试数据
    print("\n生成测试数据...")
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
    )
    
    # 6. 分析测试数据
    print("分析测试数据...")
    result = analyzer.analyze_pixel(test_histogram)
    
    # 7. 显示结果
    print(f"\n分析结果:")
    print(f"平均寿命: {result['mean_tau']:.3f} ns")
    print(f"显著成分数: {result['num_components']}")
    
    for i, comp in enumerate(result['significant_components']):
        print(f"成分 {i+1}: τ = {comp['tau']:.3f} ns, 权重 = {comp['weight']:.3f}")
    
    # 8. 绘制分析结果
    analyzer.plot_analysis_result(test_histogram, result, save_path='analysis_result.png')
    
    print("\n分析完成！结果已保存到图片文件中。")


if __name__ == "__main__":
    main()
