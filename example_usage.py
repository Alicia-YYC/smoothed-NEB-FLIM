"""
非参数贝叶斯荧光寿命分析使用示例

这个文件展示了如何使用 NonparametricBayesianFLIM 类进行荧光寿命分析
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from nonparametric_bayesian_flim import NonparametricBayesianFLIM
import os

def load_config(config_path: str = "config_nonparametric_flim.json"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_output_directory(output_dir: str):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

def example_basic_analysis():
    """基本分析示例"""
    print("=== 基本分析示例 ===\n")
    
    # 加载配置
    config = load_config()
    
    # 创建输出目录
    output_dir = config['output_settings']['output_directory']
    create_output_directory(output_dir)
    
    # 初始化分析器
    analyzer = NonparametricBayesianFLIM(
        tau_range=tuple(config['analysis_parameters']['tau_range']),
        num_tau_points=config['analysis_parameters']['num_tau_points'],
        time_range=config['analysis_parameters']['time_range'],
        num_channels=config['analysis_parameters']['num_channels']
    )
    
    # 生成训练数据
    training_histograms = analyzer.generate_training_data(
        num_samples=config['simulation_parameters']['training_samples']
    )
    
    # 训练先验分布
    prior = analyzer.train_prior(training_histograms)
    
    # 保存先验分布图
    if config['output_settings']['save_prior_distribution']:
        prior_plot_path = os.path.join(output_dir, 'prior_distribution.png')
        analyzer.plot_prior_distribution(save_path=prior_plot_path)
    
    # 生成测试数据
    test_params = config['test_parameters']
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=test_params['tau1'],
        tau2=test_params['tau2'],
        fraction1=test_params['fraction1'],
        total_photons=test_params['total_photons'],
        background_ratio=test_params['background_ratio']
    )
    
    # 分析测试数据
    result = analyzer.analyze_pixel(test_histogram)
    
    # 显示结果
    print(f"真实参数:")
    print(f"  τ₁ = {test_params['tau1']} ns")
    print(f"  τ₂ = {test_params['tau2']} ns")
    print(f"  f₁ = {test_params['fraction1']}")
    
    print(f"\n分析结果:")
    print(f"  平均寿命: {result['mean_tau']:.3f} ns")
    print(f"  显著成分数: {result['num_components']}")
    
    for i, comp in enumerate(result['significant_components']):
        print(f"  成分 {i+1}: τ = {comp['tau']:.3f} ns, 权重 = {comp['weight']:.3f}")
    
    # 保存分析结果图
    if config['output_settings']['save_analysis_results']:
        analysis_plot_path = os.path.join(output_dir, 'analysis_result.png')
        analyzer.plot_analysis_result(test_histogram, result, save_path=analysis_plot_path)
    
    return analyzer, result

def example_parameter_study():
    """参数研究示例"""
    print("\n=== 参数研究示例 ===\n")
    
    # 初始化分析器
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=30,
        time_range=12.5,
        num_channels=256
    )
    
    # 训练先验分布
    training_histograms = analyzer.generate_training_data(num_samples=200)
    prior = analyzer.train_prior(training_histograms)
    
    # 研究不同寿命参数的影响
    tau1_values = [1.5, 2.0, 2.5, 3.0]
    tau2_values = [0.5, 0.7, 0.9, 1.1]
    
    results = []
    
    for tau1 in tau1_values:
        for tau2 in tau2_values:
            if tau1 > tau2:  # 确保 τ₁ > τ₂
                # 生成测试数据
                test_histogram = analyzer.simulate_dual_exponential_histogram(
                    tau1=tau1, tau2=tau2, fraction1=0.6, total_photons=100000
                )
                
                # 分析
                result = analyzer.analyze_pixel(test_histogram)
                
                results.append({
                    'true_tau1': tau1,
                    'true_tau2': tau2,
                    'estimated_mean_tau': result['mean_tau'],
                    'num_components': result['num_components'],
                    'significant_components': result['significant_components']
                })
    
    # 显示结果
    print("参数研究结果:")
    for r in results:
        print(f"真实: τ₁={r['true_tau1']}, τ₂={r['true_tau2']} | "
              f"估计平均: {r['estimated_mean_tau']:.3f} | "
              f"成分数: {r['num_components']}")
    
    return results

def example_image_analysis():
    """图像分析示例"""
    print("\n=== 图像分析示例 ===\n")
    
    # 初始化分析器
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=40,
        time_range=12.5,
        num_channels=256
    )
    
    # 训练先验分布
    training_histograms = analyzer.generate_training_data(num_samples=300)
    prior = analyzer.train_prior(training_histograms)
    
    # 模拟图像数据（假设 10x10 像素）
    image_size = 10
    image_histograms = []
    
    print(f"生成 {image_size}x{image_size} 图像数据...")
    
    for i in range(image_size):
        for j in range(image_size):
            # 根据像素位置设置不同的参数
            tau1 = 2.0 + 0.5 * np.sin(i * np.pi / image_size)
            tau2 = 0.7 + 0.3 * np.cos(j * np.pi / image_size)
            fraction1 = 0.5 + 0.3 * (i + j) / (2 * image_size)
            
            # 生成像素直方图
            pixel_histogram = analyzer.simulate_dual_exponential_histogram(
                tau1=tau1, tau2=tau2, fraction1=fraction1, total_photons=80000
            )
            
            image_histograms.append(pixel_histogram)
    
    # 分析整个图像
    print("分析图像...")
    image_results = analyzer.analyze_image(image_histograms)
    
    # 创建结果图像
    mean_tau_image = np.array([r['mean_tau'] for r in image_results]).reshape(image_size, image_size)
    num_components_image = np.array([r['num_components'] for r in image_results]).reshape(image_size, image_size)
    
    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(mean_tau_image, cmap='viridis')
    axes[0].set_title('Mean Lifetime (ns)')
    axes[0].set_xlabel('X Pixel')
    axes[0].set_ylabel('Y Pixel')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(num_components_image, cmap='plasma')
    axes[1].set_title('Number of Components')
    axes[1].set_xlabel('X Pixel')
    axes[1].set_ylabel('Y Pixel')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('image_analysis_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图像分析完成，结果已保存为 'image_analysis_result.png'")
    
    return image_results

def example_comparison_with_traditional():
    """与传统方法比较示例"""
    print("\n=== 与传统方法比较示例 ===\n")
    
    # 初始化分析器
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=50,
        time_range=12.5,
        num_channels=256
    )
    
    # 训练先验分布
    training_histograms = analyzer.generate_training_data(num_samples=200)
    prior = analyzer.train_prior(training_histograms)
    
    # 生成测试数据
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
    )
    
    # 非参数贝叶斯方法
    bayesian_result = analyzer.analyze_pixel(test_histogram)
    
    # 传统最小二乘拟合（简化版本）
    def traditional_fit(histogram, time_centers):
        """传统最小二乘拟合"""
        from scipy.optimize import curve_fit
        
        def double_exp(t, a1, tau1, a2, tau2, offset):
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + offset
        
        # 初始猜测
        p0 = [np.max(histogram) * 0.6, 2.0, np.max(histogram) * 0.4, 0.7, np.min(histogram)]
        
        try:
            popt, pcov = curve_fit(double_exp, time_centers, histogram, p0=p0, maxfev=1000)
            return popt
        except:
            return p0
    
    # 执行传统拟合
    traditional_params = traditional_fit(test_histogram, analyzer.time_channel_centers)
    
    # 比较结果
    print("比较结果:")
    print(f"真实参数: τ₁=2.14, τ₂=0.69")
    print(f"非参数贝叶斯方法:")
    print(f"  平均寿命: {bayesian_result['mean_tau']:.3f} ns")
    print(f"  显著成分: {len(bayesian_result['significant_components'])}")
    
    print(f"传统最小二乘法:")
    print(f"  τ₁ = {traditional_params[1]:.3f} ns")
    print(f"  τ₂ = {traditional_params[3]:.3f} ns")
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始数据
    axes[0, 0].plot(analyzer.time_channel_centers, test_histogram, 'ko-', markersize=3)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Photon Counts')
    axes[0, 0].grid(True)
    
    # 贝叶斯后验分布
    axes[0, 1].semilogx(analyzer.tau_space, bayesian_result['posterior_weights'], 'r-', linewidth=2)
    axes[0, 1].set_title('Bayesian Posterior')
    axes[0, 1].set_xlabel('Lifetime (ns)')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].grid(True)
    
    # 传统拟合结果
    def plot_traditional_fit(ax):
        t = analyzer.time_channel_centers
        fit_curve = (traditional_params[0] * np.exp(-t / traditional_params[1]) + 
                    traditional_params[2] * np.exp(-t / traditional_params[3]) + 
                    traditional_params[4])
        ax.plot(t, test_histogram, 'ko-', markersize=3, label='Data')
        ax.plot(t, fit_curve, 'b-', linewidth=2, label='Traditional Fit')
        ax.set_title('Traditional Fit')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Photon Counts')
        ax.legend()
        ax.grid(True)
    
    plot_traditional_fit(axes[1, 0])
    
    # 贝叶斯拟合结果
    def plot_bayesian_fit(ax):
        model = np.zeros(analyzer.num_channels)
        for comp in bayesian_result['significant_components']:
            tau = comp['tau']
            weight = comp['weight']
            exponential = np.exp(-analyzer.time_channel_centers / tau)
            model += weight * exponential
        
        # 归一化和缩放
        model = model / np.sum(model)
        total_photons = np.sum(test_histogram)
        model = model * total_photons
        
        ax.plot(analyzer.time_channel_centers, test_histogram, 'ko-', markersize=3, label='Data')
        ax.plot(analyzer.time_channel_centers, model, 'r-', linewidth=2, label='Bayesian Fit')
        ax.set_title('Bayesian Fit')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Photon Counts')
        ax.legend()
        ax.grid(True)
    
    plot_bayesian_fit(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("比较完成，结果已保存为 'comparison_result.png'")

def main():
    """主函数：运行所有示例"""
    print("非参数贝叶斯荧光寿命分析 - 使用示例\n")
    
    try:
        # 基本分析示例
        analyzer, result = example_basic_analysis()
        
        # 参数研究示例
        param_results = example_parameter_study()
        
        # 图像分析示例
        image_results = example_image_analysis()
        
        # 与传统方法比较
        example_comparison_with_traditional()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
