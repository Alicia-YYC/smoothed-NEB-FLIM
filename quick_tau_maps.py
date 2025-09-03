#!/usr/bin/env python3
"""Quick script to generate τ1 / τ2 / a maps for a single photon count.

Usage (PowerShell / Bash):
    python quick_tau_maps.py --n 100 --out tau_maps_n100.png --smooth 0.8 --size 16

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
    histograms, _, _, _ = exp.simulate_flim_image(args.n)

    H = W = args.size
    from joblib import Parallel, delayed

    print(f"Pixel-wise fitting using {args.jobs} parallel jobs…")
    fit_results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(exp.pixel_wise_analysis)(hist) for hist in histograms)

    tau1_map = np.array([r[0] for r in fit_results]).reshape(H, W)
    tau2_map = np.array([r[1] for r in fit_results]).reshape(H, W)
    a_map   = np.array([r[2] for r in fit_results]).reshape(H, W)

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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(tau1_map_s, cmap="coolwarm", vmin=ranges["tau1"][0], vmax=ranges["tau1"][1])
    axes[0].set_title("τ1 (ns)")
    axes[1].imshow(tau2_map_s, cmap="coolwarm", vmin=ranges["tau2"][0], vmax=ranges["tau2"][1])
    axes[1].set_title("τ2 (ns)")
    axes[2].imshow(a_map, cmap="coolwarm", vmin=ranges["a"][0], vmax=ranges["a"][1])
    axes[2].set_title("a")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {args.out} (elapsed {time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()
