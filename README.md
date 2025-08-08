# Nonparametric Bayesian FLIM Analysis Experiment Package

This experiment package contains complete code and documentation for reproducing all experiments from the paper "Non-parametric Bayesian FLIM Analysis".

## File Description

### Core Code Files
- `paper_experiment_reproduction_fixed.py` - Main experiment reproduction code, containing three experiments:
  - Experiment 1: Prior distribution estimation performance evaluation
  - Experiment 2: Pixel-wise lifetime recovery performance comparison
  - Experiment 3: Computation efficiency evaluation
- `nonparametric_bayesian_flim.py` - Core implementation of nonparametric Bayesian FLIM analysis
- `example_usage.py` - Usage examples and demonstration code

### Configuration Files
- `config_nonparametric_flim.json` - Analysis parameter configuration file
- `requirements.txt` - Python dependency package list

### Documentation and Results
- `README_nonparametric_flim.md` - Detailed technical documentation
- `paper_experiment_results_fixed.png` - Experiment result charts

## Installation and Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
python paper_experiment_reproduction_fixed.py
```

### 3. View Examples
```bash
python example_usage.py
```

## Experiment Parameters

### Experiment Setup
- Image size: 32×32 pixels
- Time range: 10 ns (10000 ps)
- Number of time channels: 256
- Background photon ratio: 0.001
- IRF: Gaussian distribution, mean 1500 ps, standard deviation 100 ps

### Experiment 1: Prior Distribution Estimation
- Photon number range: [10, 32, 100, 316, 1000]
- Interval number range: [400, 600, 800]
- Number of repetitions: 3
- Evaluation metric: L2 distance

### Experiment 2: Pixel-wise Recovery
- Photon number range: [100, 316, 1000, 3162]
- Number of repetitions: 3
- Evaluation metric: Mean Square Error (MSE)

### Experiment 3: Computation Efficiency
- Image sizes: [16×16, 32×32, 64×64]
- Photons: 1000/pixel
- Evaluation metric: Computation time

## Algorithm Implementation

### Nonparametric Bayesian FLIM Analysis
1. **Dual-exponential Decay Simulation**: Generate photon histogram data with background
2. **Probability Library Construction**: Build probability library P(τ) for possible lifetimes
3. **NPMLE Estimation**: Non-parametric Maximum Likelihood Estimation to obtain prior distribution
4. **EM-MAP Estimation**: Expectation-Maximization Maximum A Posteriori estimation to recover pixel parameters

### Comparison Methods
- **Pixel-wise Analysis**: Fit using only photons from individual pixels
- **Global Analysis**: Globally estimate lifetimes of two components, then estimate component contribution for each pixel
- **NEB-FLIM**: Empirical Bayesian analysis, combining local and global information

## Result Description

Experiment results are saved in `paper_experiment_results_fixed.png`, including:
- Prior distribution estimation performance under different photon numbers and interval numbers
- Pixel-wise lifetime recovery performance comparison
- Computation efficiency comparison

## Technical Details

### Numerical Stability Improvements
- Limit signal value range to avoid `lam value too large` errors
- Generate Poisson samples channel by channel to improve numerical stability
- Use non-interactive matplotlib backend to avoid display issues

### Performance Optimization
- Reduce experiment parameter ranges to speed up debugging
- Use try-except blocks to handle exceptional cases
- Optimize memory usage and computation efficiency

## Notes

1. Ensure sufficient computational resources in Python environment
2. Experiments may take a long time to run, recommended to execute in background
3. If numerical issues are encountered, parameter ranges can be adjusted
4. Result charts are automatically saved, no manual display required

## Citation

This experiment package is implemented based on the following paper:
"Non-parametric Bayesian FLIM Analysis" - Biomedical Optics Express, Vol. 10, No. 11
