#!/usr/bin/env python3
import json
from paper_experiment_reproduction_fixed import PaperExperimentReproductionFixed as E

if __name__ == "__main__":
    exp = E(image_size=16)
    res = exp.experiment_a_sensitivity(
        n_photons=316,
        num_repeats=1,
        save_plot=True,
        save_path='a_sensitivity_results.png'
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


