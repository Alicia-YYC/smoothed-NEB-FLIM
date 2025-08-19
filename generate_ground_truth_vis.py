#!/usr/bin/env python3
"""
Generate Ground Truth Parameters Visualization
Generate 2D visualization of tau1, tau2, and a parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def simulate_ground_truth_image(image_size=32):
    """
    Simulate ground truth image parameters (based on paper Figure 4)
    
    Returns:
        tau1_image: τ1 image (440-500 ps = 0.44-0.50 ns)
        tau2_image: τ2 image (1850-2150 ps = 1.85-2.15 ns)
        a_image: a image (0.41-0.50)
    """
    H, W = image_size, image_size
    yy, xx = np.mgrid[0:H, 0:W]
    
    # 1) τ1：中心高、四周低（高斯团），范围 0.44–0.50 ns
    tau1_min, tau1_max = 0.44, 0.50  # ns
    amp1 = tau1_max - tau1_min       # 0.06 ns
    cy, cx = (H-1)/2, (W-1)/2        # 中心
    sigma1 = 6.0                     # 可微调团的大小
    r2_center = (yy-cy)**2 + (xx-cx)**2
    tau1_image = tau1_min + amp1 * np.exp(-r2_center/(2*sigma1**2))
    
    # 2) τ2：左下角高、向右上角递减，范围 1.85–2.15 ns
    tau2_min, tau2_max = 1.85, 2.15  # ns
    amp2 = tau2_max - tau2_min       # 0.30 ns
    oy, ox = H-1, 0                  # 左下角
    r_corner = np.sqrt((yy-oy)**2 + (xx-ox)**2)
    r_corner_n = (r_corner - r_corner.min()) / (r_corner.max() - r_corner.min())
    tau2_image = tau2_max - amp2 * r_corner_n  # 角上最大，远离角逐渐减到最小
    
    # 3) a：四分之一同心圆（左上角最小，右下角最大），范围 0.41–0.50
    a_min, a_max = 0.41, 0.50
    r_tl = np.sqrt((yy - 0)**2 + (xx - 0)**2)
    r_tl_n = (r_tl - r_tl.min()) / (r_tl.max() - r_tl.min())
    a_image = a_min + (a_max - a_min) * r_tl_n
    
    return tau1_image, tau2_image, a_image

def plot_ground_truth_parameters(save_path="ground_truth_parameters.png"):
    """
    Generate 2D visualization of ground truth parameters (tau1, tau2, a)
    """
    # Generate ground truth parameters
    tau1_image, tau2_image, a_image = simulate_ground_truth_image()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot tau1
    im1 = axes[0].imshow(tau1_image, cmap='coolwarm', aspect='equal', vmin=0.44, vmax=0.50)
    axes[0].set_title('Ground Truth τ1 (ns)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X Position (pixels)', fontsize=12)
    axes[0].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('τ1 (ns)', fontsize=12)
    
    # Plot tau2
    im2 = axes[1].imshow(tau2_image, cmap='coolwarm', aspect='equal', vmin=1.85, vmax=2.15)
    axes[1].set_title('Ground Truth τ2 (ns)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X Position (pixels)', fontsize=12)
    axes[1].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('τ2 (ns)', fontsize=12)
    
    # Plot a
    im3 = axes[2].imshow(a_image, cmap='coolwarm', aspect='equal', vmin=0.41, vmax=0.50)
    axes[2].set_title('Ground Truth a (amplitude ratio)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('X Position (pixels)', fontsize=12)
    axes[2].set_ylabel('Y Position (pixels)', fontsize=12)
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('a', fontsize=12)
    
    # Add text annotations with parameter ranges
    fig.text(0.02, 0.98, f'τ1 range: {np.min(tau1_image):.2f} - {np.max(tau1_image):.2f} ns', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    fig.text(0.02, 0.95, f'τ2 range: {np.min(tau2_image):.2f} - {np.max(tau2_image):.2f} ns', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    fig.text(0.02, 0.92, f'a range: {np.min(a_image):.2f} - {np.max(a_image):.2f}', 
            fontsize=10, transform=fig.transFigure, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ground truth parameters visualization saved to: {save_path}")
    
    # Also print parameter statistics
    print(f"\n=== Ground Truth Parameters Statistics ===")
    print(f"τ1: min={np.min(tau1_image):.3f}, max={np.max(tau1_image):.3f}, mean={np.mean(tau1_image):.3f} ns")
    print(f"τ2: min={np.min(tau2_image):.3f}, max={np.max(tau2_image):.3f}, mean={np.mean(tau2_image):.3f} ns")
    print(f"a:  min={np.min(a_image):.3f}, max={np.max(a_image):.3f}, mean={np.mean(a_image):.3f}")
    
    return tau1_image, tau2_image, a_image

if __name__ == "__main__":
    print("Generating Ground Truth Parameters Visualization...")
    plot_ground_truth_parameters()
    print("Done!")
