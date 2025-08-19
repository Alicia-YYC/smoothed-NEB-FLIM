#!/usr/bin/env python3
"""
Detailed Ground Truth Parameters Analysis
Generate comprehensive 2D visualizations and analysis of tau1, tau2, and a parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.colors import LinearSegmentedColormap

def simulate_ground_truth_image(image_size=32):
    """
    Simulate ground truth image parameters (based on paper Figure 4)
    """
    # Create 32x32 grid
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    
    # Design parameter distribution based on paper Figure 4
    # τ1: varies between 1.5-3.0 ns
    tau1_image = 1.5 + 1.5 * np.sin(np.pi * x / image_size) * np.cos(np.pi * y / image_size)
    
    # τ2: varies between 0.5-1.2 ns
    tau2_image = 0.5 + 0.7 * np.cos(np.pi * x / image_size) * np.sin(np.pi * y / image_size)
    
    # a: varies between 0.3-0.8
    a_image = 0.3 + 0.5 * (x + y) / (2 * image_size)
    
    return tau1_image, tau2_image, a_image

def create_custom_colormaps():
    """Create custom colormaps for better visualization"""
    # Custom colormap for tau1 (blue to green)
    colors1 = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    cmap_tau1 = LinearSegmentedColormap.from_list('tau1_cmap', colors1, N=256)
    
    # Custom colormap for tau2 (red to yellow)
    colors2 = ['#800000', '#FF0000', '#FF8000', '#FFFF00']
    cmap_tau2 = LinearSegmentedColormap.from_list('tau2_cmap', colors2, N=256)
    
    # Custom colormap for a (purple to red)
    colors3 = ['#800080', '#FF00FF', '#FF0080', '#FF0000']
    cmap_a = LinearSegmentedColormap.from_list('a_cmap', colors3, N=256)
    
    return cmap_tau1, cmap_tau2, cmap_a

def plot_comprehensive_visualization(save_path="comprehensive_ground_truth.png"):
    """
    Generate comprehensive 2D visualization with multiple views
    """
    # Generate ground truth parameters
    tau1_image, tau2_image, a_image = simulate_ground_truth_image()
    
    # Create custom colormaps
    cmap_tau1, cmap_tau2, cmap_a = create_custom_colormaps()
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Row 1: Standard visualizations
    # Plot tau1
    im1 = axes[0, 0].imshow(tau1_image, cmap=cmap_tau1, aspect='equal')
    axes[0, 0].set_title('Ground Truth τ1 (ns)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X Position (pixels)', fontsize=12)
    axes[0, 0].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('τ1 (ns)', fontsize=12)
    
    # Plot tau2
    im2 = axes[0, 1].imshow(tau2_image, cmap=cmap_tau2, aspect='equal')
    axes[0, 1].set_title('Ground Truth τ2 (ns)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('X Position (pixels)', fontsize=12)
    axes[0, 1].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('τ2 (ns)', fontsize=12)
    
    # Plot a
    im3 = axes[0, 2].imshow(a_image, cmap=cmap_a, aspect='equal')
    axes[0, 2].set_title('Ground Truth a (amplitude ratio)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('X Position (pixels)', fontsize=12)
    axes[0, 2].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar3 = plt.colorbar(im3, ax=axes[0, 2])
    cbar3.set_label('a', fontsize=12)
    
    # Row 2: Enhanced visualizations
    # Tau1 with contour lines
    im4 = axes[1, 0].imshow(tau1_image, cmap=cmap_tau1, aspect='equal')
    contours = axes[1, 0].contour(tau1_image, levels=10, colors='white', alpha=0.7, linewidths=0.5)
    axes[1, 0].clabel(contours, inline=True, fontsize=8)
    axes[1, 0].set_title('τ1 with Contour Lines', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('X Position (pixels)', fontsize=12)
    axes[1, 0].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar4 = plt.colorbar(im4, ax=axes[1, 0])
    cbar4.set_label('τ1 (ns)', fontsize=12)
    
    # Tau2 with contour lines
    im5 = axes[1, 1].imshow(tau2_image, cmap=cmap_tau2, aspect='equal')
    contours = axes[1, 1].contour(tau2_image, levels=8, colors='white', alpha=0.7, linewidths=0.5)
    axes[1, 1].clabel(contours, inline=True, fontsize=8)
    axes[1, 1].set_title('τ2 with Contour Lines', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('X Position (pixels)', fontsize=12)
    axes[1, 1].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar5 = plt.colorbar(im5, ax=axes[1, 1])
    cbar5.set_label('τ2 (ns)', fontsize=12)
    
    # Parameter correlation (tau1 vs tau2)
    scatter = axes[1, 2].scatter(tau1_image.flatten(), tau2_image.flatten(), 
                                 c=a_image.flatten(), cmap='viridis', alpha=0.7, s=20)
    axes[1, 2].set_xlabel('τ1 (ns)', fontsize=12)
    axes[1, 2].set_ylabel('τ2 (ns)', fontsize=12)
    axes[1, 2].set_title('τ1 vs τ2 Correlation\n(colored by a)', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    cbar6 = plt.colorbar(scatter, ax=axes[1, 2])
    cbar6.set_label('a', fontsize=12)
    
    # Add parameter statistics
    fig.text(0.02, 0.98, f'τ1: {np.min(tau1_image):.2f}-{np.max(tau1_image):.2f} ns, mean={np.mean(tau1_image):.2f} ns', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    fig.text(0.02, 0.95, f'τ2: {np.min(tau2_image):.2f}-{np.max(tau2_image):.2f} ns, mean={np.mean(tau2_image):.2f} ns', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    fig.text(0.02, 0.92, f'a: {np.min(a_image):.2f}-{np.max(a_image):.2f}, mean={np.mean(a_image):.2f}', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {save_path}")
    
    return tau1_image, tau2_image, a_image

def plot_parameter_distributions(save_path="parameter_distributions.png"):
    """
    Generate histogram distributions of the parameters
    """
    # Generate ground truth parameters
    tau1_image, tau2_image, a_image = simulate_ground_truth_image()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Histogram of tau1
    axes[0].hist(tau1_image.flatten(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('τ1 (ns)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('τ1 Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(np.mean(tau1_image), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(tau1_image):.3f} ns')
    axes[0].legend()
    
    # Histogram of tau2
    axes[1].hist(tau2_image.flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('τ2 (ns)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('τ2 Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(np.mean(tau2_image), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(tau2_image):.3f} ns')
    axes[1].legend()
    
    # Histogram of a
    axes[2].hist(a_image.flatten(), bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[2].set_xlabel('a', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('a Distribution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(np.mean(a_image), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(a_image):.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parameter distributions saved to: {save_path}")

def print_detailed_statistics():
    """
    Print detailed statistical analysis of the parameters
    """
    # Generate ground truth parameters
    tau1_image, tau2_image, a_image = simulate_ground_truth_image()
    
    print("=== DETAILED GROUND TRUTH PARAMETERS ANALYSIS ===")
    print(f"\nτ1 (First Lifetime Component):")
    print(f"  Range: {np.min(tau1_image):.3f} - {np.max(tau1_image):.3f} ns")
    print(f"  Mean: {np.mean(tau1_image):.3f} ns")
    print(f"  Std: {np.std(tau1_image):.3f} ns")
    print(f"  Median: {np.median(tau1_image):.3f} ns")
    
    print(f"\nτ2 (Second Lifetime Component):")
    print(f"  Range: {np.min(tau2_image):.3f} - {np.max(tau2_image):.3f} ns")
    print(f"  Mean: {np.mean(tau2_image):.3f} ns")
    print(f"  Std: {np.std(tau2_image):.3f} ns")
    print(f"  Median: {np.median(tau2_image):.3f} ns")
    
    print(f"\na (Amplitude Ratio):")
    print(f"  Range: {np.min(a_image):.3f} - {np.max(a_image):.3f}")
    print(f"  Mean: {np.mean(a_image):.3f}")
    print(f"  Std: {np.std(a_image):.3f}")
    print(f"  Median: {np.median(a_image):.3f}")
    
    print(f"\nParameter Correlations:")
    print(f"  τ1 vs τ2: {np.corrcoef(tau1_image.flatten(), tau2_image.flatten())[0,1]:.3f}")
    print(f"  τ1 vs a: {np.corrcoef(tau1_image.flatten(), a_image.flatten())[0,1]:.3f}")
    print(f"  τ2 vs a: {np.corrcoef(tau2_image.flatten(), a_image.flatten())[0,1]:.3f}")

if __name__ == "__main__":
    print("Generating Comprehensive Ground Truth Parameters Analysis...")
    
    # Generate basic visualization
    print("\n1. Generating basic visualization...")
    plot_ground_truth_parameters()
    
    # Generate comprehensive visualization
    print("\n2. Generating comprehensive visualization...")
    plot_comprehensive_visualization()
    
    # Generate parameter distributions
    print("\n3. Generating parameter distributions...")
    plot_parameter_distributions()
    
    # Print detailed statistics
    print("\n4. Printing detailed statistics...")
    print_detailed_statistics()
    
    print("\nAll visualizations completed!")
    print("Generated files:")
    print("- ground_truth_parameters.png (basic visualization)")
    print("- comprehensive_ground_truth.png (comprehensive analysis)")
    print("- parameter_distributions.png (histogram distributions)")
