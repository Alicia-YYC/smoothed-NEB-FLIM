#!/usr/bin/env python3
"""Quick script to generate τ1 / τ2 / a maps for a single photon count.

Usage (PowerShell / Bash):
    python demo.py --n 100 --out tau_maps_n100.png --smooth 0.8 --size 128

Arguments
---------
--n         Photon count per pixel (default: 100)
--out       Output PNG path (default: tau_maps.png)
--smooth    λ for Laplacian smoothing (0 → disable, default: 0.8)
--size      Image size H=W (default: 16)

The script:
1. Simulates one FLIM image with given photons.
2. Runs pixel-wise analysis on each pixel.
3. Optionally applies Laplacian smoothing to τ1/τ2 maps.
4. Saves a 1×3 panel figure (τ1, τ2, a).
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
from paper_experiment_reproduction_fixed import PaperExperimentReproductionFixed

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tau maps for a single photon count")
    parser.add_argument("--n", type=int, default=100, help="photons per pixel")
    parser.add_argument("--out", type=str, default="tau_maps.png", help="output PNG path")
    parser.add_argument("--smooth", type=float, default=0.8, help="lambda for Laplacian smoothing (0=off)")
    parser.add_argument("--size", type=int, default=16, help="image size (H = W)")
    parser.add_argument("--bin", type=int, default=3, choices=[1,3,5], help="binning factor (1=no binning)")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs for pixel-wise fitting (-1 = all cores)")
    args = parser.parse_args()

    exp = PaperExperimentReproductionFixed(
        image_size=args.size,
        time_range=10.0,
        num_channels=256,
        irf_mean=1.5,
        irf_std=0.1,
    )

    print(f"Simulating image ({args.size}×{args.size}) with n={args.n} photons…")
    start = time.time()
    histograms, gt_tau1, gt_tau2, gt_a = exp.simulate_flim_image(args.n)

    # Optional binning
    if args.bin == 1:
        H = W = args.size
    elif args.bin == 3:
        histograms, H, W = exp._bin3x3_histograms(histograms)
        print(f"After 3×3 binning -> working size: {H}×{W}")
    else:  # 5×5 binning
        # Generic 5×5 sum binning
        H0 = W0 = args.size
        H5, W5 = H0 // 5, W0 // 5
        if H5 == 0 or W5 == 0:
            H = W = args.size  # fall back no binning
        else:
            C = histograms[0].shape[0]
            arr = np.array(histograms).reshape(H0, W0, C)
            arr = arr[:H5*5, :W5*5, :].reshape(H5, 5, W5, 5, C).sum(axis=(1,3))
            histograms = [arr[i, j].astype(int) for i in range(H5) for j in range(W5)]
            H, W = H5, W5
            print(f"After 5×5 binning -> working size: {H}×{W}")

    from joblib import Parallel, delayed

    print(f"Pixel-wise fitting using {args.jobs} parallel jobs…")
    fit_results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(exp.pixel_wise_analysis)(hist) for hist in histograms)

    tau1_map = np.array([r[0] for r in fit_results]).reshape(H, W)
    tau2_map = np.array([r[1] for r in fit_results]).reshape(H, W)
    a_map   = np.array([r[2] for r in fit_results]).reshape(H, W)

    # Optional Laplacian smoothing for a as well
    if args.smooth > 0:
        a_map_s = exp.laplacian_smooth_2d(a_map, args.smooth)
    else:
        a_map_s = a_map

    if args.smooth > 0:
        tau1_map_s = exp.laplacian_smooth_2d(tau1_map, args.smooth)
        tau2_map_s = exp.laplacian_smooth_2d(tau2_map, args.smooth)
    else:
        tau1_map_s, tau2_map_s = tau1_map, tau2_map

    # plot
    # Use same color scale as main experiments
    ranges = {
        "tau1": (0.44, 0.50),
        "tau2": (1.85, 2.15),
        "a":    (0.41, 0.50),
    }

    # Plot GT (row 0) and estimates (row 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Row 0: Ground truth
    axes[0,0].imshow(gt_tau1, cmap="coolwarm", vmin=ranges["tau1"][0], vmax=ranges["tau1"][1])
    axes[0,0].set_title("GT τ1 (ns)")
    axes[0,1].imshow(gt_tau2, cmap="coolwarm", vmin=ranges["tau2"][0], vmax=ranges["tau2"][1])
    axes[0,1].set_title("GT τ2 (ns)")
    axes[0,2].imshow(gt_a, cmap="coolwarm", vmin=ranges["a"][0], vmax=ranges["a"][1])
    axes[0,2].set_title("GT a")

    # Row 1: Estimates
    axes[1,0].imshow(tau1_map_s, cmap="coolwarm", vmin=ranges["tau1"][0], vmax=ranges["tau1"][1])
    axes[1,0].set_title("Est τ1")
    axes[1,1].imshow(tau2_map_s, cmap="coolwarm", vmin=ranges["tau2"][0], vmax=ranges["tau2"][1])
    axes[1,1].set_title("Est τ2")
    axes[1,2].imshow(a_map_s, cmap="coolwarm", vmin=ranges["a"][0], vmax=ranges["a"][1])
    axes[1,2].set_title("Est a")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {args.out} (elapsed {time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()
