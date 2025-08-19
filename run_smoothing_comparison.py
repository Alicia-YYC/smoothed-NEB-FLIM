#!/usr/bin/env python3
from paper_experiment_reproduction_fixed import PaperExperimentReproductionFixed as E
import json

if __name__ == "__main__":
    exp = E()
    metrics = exp.plot_smoothing_comparison(
        n_photons=1000,
        lambda_laplacian=0.2,
        save_path='smoothing_comparison.png'
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


