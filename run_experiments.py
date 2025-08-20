#!/usr/bin/env python3
"""
Experiment Runner Script
For quickly executing nonparametric Bayesian FLIM analysis experiments
"""

import os
import sys
import time
from paper_experiment_reproduction_fixed import main

def run_experiments():
    """Run all experiments"""
    print("=" * 60)
    print("Nonparametric Bayesian FLIM Analysis Experiments")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run main experiments
        main()
        
        end_time = time.time()
        print(f"\nExperiments completed! Total time: {end_time - start_time:.2f} seconds")
        print("Results saved to: error_line_charts.png")
        
    except Exception as e:
        print(f"Error running experiments: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_experiments()
