"""
Nonparametric Bayesian Fluorescence Lifetime Analysis Usage Examples

This file demonstrates how to use the NonparametricBayesianFLIM class for fluorescence lifetime analysis
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from nonparametric_bayesian_flim import NonparametricBayesianFLIM
import os

def load_config(config_path: str = "config_nonparametric_flim.json"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_output_directory(output_dir: str):
    """Create output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def example_basic_analysis():
    """Basic analysis example"""
    print("=== Basic Analysis Example ===\n")
    
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = config['output_settings']['output_directory']
    create_output_directory(output_dir)
    
    # Initialize analyzer
    analyzer = NonparametricBayesianFLIM(
        tau_range=tuple(config['analysis_parameters']['tau_range']),
        num_tau_points=config['analysis_parameters']['num_tau_points'],
        time_range=config['analysis_parameters']['time_range'],
        num_channels=config['analysis_parameters']['num_channels']
    )
    
    # Generate training data
    training_histograms = analyzer.generate_training_data(
        num_samples=config['simulation_parameters']['training_samples']
    )
    
    # Train prior distribution
    prior = analyzer.train_prior(training_histograms)
    
    # Save prior distribution plot
    if config['output_settings']['save_prior_distribution']:
        prior_plot_path = os.path.join(output_dir, 'prior_distribution.png')
        analyzer.plot_prior_distribution(save_path=prior_plot_path)
    
    # Generate test data
    test_params = config['test_parameters']
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=test_params['tau1'],
        tau2=test_params['tau2'],
        fraction1=test_params['fraction1'],
        total_photons=test_params['total_photons'],
        background_ratio=test_params['background_ratio']
    )
    
    # Analyze test data
    result = analyzer.analyze_pixel(test_histogram)
    
    # Display results
    print(f"True parameters:")
    print(f"  τ₁ = {test_params['tau1']} ns")
    print(f"  τ₂ = {test_params['tau2']} ns")
    print(f"  f₁ = {test_params['fraction1']}")
    
    print(f"\nAnalysis results:")
    print(f"  Mean lifetime: {result['mean_tau']:.3f} ns")
    print(f"  Number of significant components: {result['num_components']}")
    
    for i, comp in enumerate(result['significant_components']):
        print(f"  Component {i+1}: τ = {comp['tau']:.3f} ns, weight = {comp['weight']:.3f}")
    
    # Save analysis result plot
    if config['output_settings']['save_analysis_results']:
        analysis_plot_path = os.path.join(output_dir, 'analysis_result.png')
        analyzer.plot_analysis_result(test_histogram, result, save_path=analysis_plot_path)
    
    return analyzer, result

def example_parameter_study():
    """Parameter study example"""
    print("\n=== Parameter Study Example ===\n")
    
    # Initialize analyzer
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=30,
        time_range=12.5,
        num_channels=256
    )
    
    # Train prior distribution
    training_histograms = analyzer.generate_training_data(num_samples=200)
    prior = analyzer.train_prior(training_histograms)
    
    # Study the effect of different lifetime parameters
    tau1_values = [1.5, 2.0, 2.5, 3.0]
    tau2_values = [0.5, 0.7, 0.9, 1.1]
    
    results = []
    
    for tau1 in tau1_values:
        for tau2 in tau2_values:
            if tau1 > tau2:  # Ensure τ₁ > τ₂
                # Generate test data
                test_histogram = analyzer.simulate_dual_exponential_histogram(
                    tau1=tau1, tau2=tau2, fraction1=0.6, total_photons=100000
                )
                
                # Analyze
                result = analyzer.analyze_pixel(test_histogram)
                
                results.append({
                    'true_tau1': tau1,
                    'true_tau2': tau2,
                    'estimated_mean_tau': result['mean_tau'],
                    'num_components': result['num_components'],
                    'significant_components': result['significant_components']
                })
    
    # Display results
    print("Parameter study results:")
    for r in results:
        print(f"True: τ₁={r['true_tau1']}, τ₂={r['true_tau2']} | "
              f"Estimated mean: {r['estimated_mean_tau']:.3f} | "
              f"Components: {r['num_components']}")
    
    return results

def example_image_analysis():
    """Image analysis example"""
    print("\n=== Image Analysis Example ===\n")
    
    # Initialize analyzer
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=40,
        time_range=12.5,
        num_channels=256
    )
    
    # Train prior distribution
    training_histograms = analyzer.generate_training_data(num_samples=300)
    prior = analyzer.train_prior(training_histograms)
    
    # Simulate image data (assume 10x10 pixels)
    image_size = 10
    image_histograms = []
    
    print(f"Generating {image_size}x{image_size} image data...")
    
    for i in range(image_size):
        for j in range(image_size):
            # Set different parameters based on pixel position
            tau1 = 2.0 + 0.5 * np.sin(i * np.pi / image_size)
            tau2 = 0.7 + 0.3 * np.cos(j * np.pi / image_size)
            fraction1 = 0.5 + 0.3 * (i + j) / (2 * image_size)
            
            # Generate pixel histogram
            pixel_histogram = analyzer.simulate_dual_exponential_histogram(
                tau1=tau1, tau2=tau2, fraction1=fraction1, total_photons=80000
            )
            
            image_histograms.append(pixel_histogram)
    
    # Analyze entire image
    print("Analyzing image...")
    image_results = analyzer.analyze_image(image_histograms)
    
    # Create result images
    mean_tau_image = np.array([r['mean_tau'] for r in image_results]).reshape(image_size, image_size)
    num_components_image = np.array([r['num_components'] for r in image_results]).reshape(image_size, image_size)
    
    # Plot results
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
    
    print(f"Image analysis completed, results saved as 'image_analysis_result.png'")
    
    return image_results

def example_comparison_with_traditional():
    """Comparison with traditional methods example"""
    print("\n=== Comparison with Traditional Methods Example ===\n")
    
    # Initialize analyzer
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=50,
        time_range=12.5,
        num_channels=256
    )
    
    # Train prior distribution
    training_histograms = analyzer.generate_training_data(num_samples=200)
    prior = analyzer.train_prior(training_histograms)
    
    # Generate test data
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
    )
    
    # Nonparametric Bayesian method
    bayesian_result = analyzer.analyze_pixel(test_histogram)
    
    # Traditional least squares fitting (simplified version)
    def traditional_fit(histogram, time_centers):
        """Traditional least squares fitting"""
        from scipy.optimize import curve_fit
        
        def double_exp(t, a1, tau1, a2, tau2, offset):
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + offset
        
        # Initial guess
        p0 = [np.max(histogram) * 0.6, 2.0, np.max(histogram) * 0.4, 0.7, np.min(histogram)]
        
        try:
            popt, pcov = curve_fit(double_exp, time_centers, histogram, p0=p0, maxfev=1000)
            return popt
        except:
            return p0
    
    # Perform traditional fitting
    traditional_params = traditional_fit(test_histogram, analyzer.time_channel_centers)
    
    # Compare results
    print("Comparison results:")
    print(f"True parameters: τ₁=2.14, τ₂=0.69")
    print(f"Nonparametric Bayesian method:")
    print(f"  Mean lifetime: {bayesian_result['mean_tau']:.3f} ns")
    print(f"  Significant components: {len(bayesian_result['significant_components'])}")
    
    print(f"Traditional least squares method:")
    print(f"  τ₁ = {traditional_params[1]:.3f} ns")
    print(f"  τ₂ = {traditional_params[3]:.3f} ns")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    axes[0, 0].plot(analyzer.time_channel_centers, test_histogram, 'ko-', markersize=3)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Photon Counts')
    axes[0, 0].grid(True)
    
    # Bayesian posterior distribution
    axes[0, 1].semilogx(analyzer.tau_space, bayesian_result['posterior_weights'], 'r-', linewidth=2)
    axes[0, 1].set_title('Bayesian Posterior')
    axes[0, 1].set_xlabel('Lifetime (ns)')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].grid(True)
    
    # Traditional fitting result
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
    
    # Bayesian fitting result
    def plot_bayesian_fit(ax):
        model = np.zeros(analyzer.num_channels)
        for comp in bayesian_result['significant_components']:
            tau = comp['tau']
            weight = comp['weight']
            exponential = np.exp(-analyzer.time_channel_centers / tau)
            model += weight * exponential
        
        # Normalize and scale
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
    
    print("Comparison completed, results saved as 'comparison_result.png'")

def main():
    """Main function: run all examples"""
    print("Nonparametric Bayesian Fluorescence Lifetime Analysis - Usage Examples\n")
    
    try:
        # Basic analysis example
        analyzer, result = example_basic_analysis()
        
        # Parameter study example
        param_results = example_parameter_study()
        
        # Image analysis example
        image_results = example_image_analysis()
        
        # Comparison with traditional methods
        example_comparison_with_traditional()
        
        print("\nAll examples completed!")
        
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
