# Nonparametric Bayesian Fluorescence Lifetime Analysis (Nonparametric Bayesian FLIM Analysis)

## Overview

This project implements a nonparametric Bayesian fluorescence lifetime analysis framework, including the following four main steps:

1. **Simulate dual-exponential decay photon histogram data (with background)**
2. **Build probability library P(τ) for possible lifetimes**
3. **Perform Non-Parametric Maximum Likelihood Estimation (NPMLE) to obtain prior distribution**
4. **Apply Expectation-Maximization (EM) for Maximum A Posteriori (MAP) estimation to recover parameters for each pixel**

## File Structure

```
├── nonparametric_bayesian_flim.py    # Main analysis class
├── config_nonparametric_flim.json    # Configuration file
├── example_usage.py                   # Usage examples
├── README_nonparametric_flim.md      # Documentation
└── nonparametric_flim_results/       # Output directory (auto-created)
```

## Install Dependencies

```bash
pip install numpy matplotlib scipy pandas
```

## Quick Start

### 1. Basic Usage

```python
from nonparametric_bayesian_flim import NonparametricBayesianFLIM

# Initialize analyzer
analyzer = NonparametricBayesianFLIM(
    tau_range=(0.1, 10.0),      # Lifetime range (ns)
    num_tau_points=50,          # Number of points in lifetime space
    time_range=12.5,            # Time range (ns)
    num_channels=256            # Number of time channels
)

# Generate training data
training_histograms = analyzer.generate_training_data(num_samples=500)

# Train prior distribution
prior = analyzer.train_prior(training_histograms)

# Generate test data
test_histogram = analyzer.simulate_dual_exponential_histogram(
    tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
)

# Analyze data
result = analyzer.analyze_pixel(test_histogram)

# Display results
print(f"Mean lifetime: {result['mean_tau']:.3f} ns")
print(f"Number of significant components: {result['num_components']}")
```

### 2. Run Complete Examples

```bash
python example_usage.py
```

This will run the following examples:
- Basic analysis example
- Parameter study example
- Image analysis example
- Comparison with traditional methods example

## Main Features

### NonparametricBayesianFLIM Class

#### Initialization Parameters
- `tau_range`: Lifetime range (ns)
- `num_tau_points`: Number of points in lifetime space
- `time_range`: Time range (ns)
- `num_channels`: Number of time channels

#### Main Methods

##### Data Generation
- `simulate_dual_exponential_histogram()`: Simulate dual-exponential decay histogram
- `generate_training_data()`: Generate training data

##### Prior Training
- `build_probability_library()`: Build lifetime probability library
- `npmle_estimation()`: Non-parametric Maximum Likelihood Estimation
- `train_prior()`: Train prior distribution

##### Data Analysis
- `em_map_estimation()`: EM-MAP estimation
- `analyze_pixel()`: Analyze single pixel
- `analyze_image()`: Analyze entire image

##### Visualization
- `plot_prior_distribution()`: Plot prior distribution
- `plot_analysis_result()`: Plot analysis results

## Configuration

### config_nonparametric_flim.json

```json
{
  "analysis_parameters": {
    "tau_range": [0.1, 10.0],           // Lifetime range
    "num_tau_points": 50,               // Number of points in lifetime space
    "time_range": 12.5,                 // Time range
    "num_channels": 256,                // Number of time channels
    "convergence_tolerance": 1e-6,      // Convergence tolerance
    "max_iterations": 100,              // Maximum iterations
    "significance_threshold": 0.01      // Significance threshold
  },
  
  "simulation_parameters": {
    "training_samples": 500,            // Number of training samples
    "tau1_range": [1.5, 3.0],          // Long lifetime range
    "tau2_range": [0.5, 1.2],          // Short lifetime range
    "fraction1_range": [0.4, 0.8],     // Proportion range
    "total_photons_range": [50000, 200000], // Photon number range
    "background_ratio_range": [0.05, 0.2],  // Background ratio range
    "noise_level": 0.05                // Noise level
  }
}
```

## Algorithm Principles

### 1. Non-parametric Maximum Likelihood Estimation (NPMLE)

NPMLE is used to learn the prior distribution of lifetimes from training data:

```python
# Objective function: maximize log likelihood
def objective_function(weights):
    mixed_distribution = sum(weight_i * exp(-t/tau_i) for i, weight_i in enumerate(weights))
    return -sum(log_likelihood(histogram, mixed_distribution) for histogram in training_data)
```

### 2. Expectation-Maximization Maximum A Posteriori Estimation (EM-MAP)

The EM-MAP algorithm is used to estimate the lifetime distribution of individual pixels:

#### E-step (Expectation Step)
```python
responsibilities[i,j] = weights[i] * exp(-time[j]/tau[i]) / sum(weights[k] * exp(-time[j]/tau[k]))
```

#### M-step (Maximization Step)
```python
new_weights[i] = sum(responsibilities[i,j] * histogram[j]) * prior[i]
```

## Output Results

### Analysis Result Format

```python
result = {
    'posterior_weights': array,           # Posterior weight distribution
    'significant_components': [           # List of significant components
        {
            'tau': 2.14,                  # Lifetime value
            'weight': 0.6,                # Weight
            'index': 15                   # Index
        }
    ],
    'mean_tau': 1.85,                    # Mean lifetime
    'num_components': 2                   # Number of significant components
}
```

### Output Files

- `prior_distribution.png`: Prior distribution plot
- `analysis_result.png`: Analysis result plot
- `image_analysis_result.png`: Image analysis result
- `comparison_result.png`: Comparison with traditional methods

## Advantages

### 1. Nonparametric Nature
- No need to assume specific lifetime distribution forms
- Can adapt to complex multi-component fluorescence systems

### 2. Robustness
- More robust to noise and background
- Can handle low signal-to-noise ratio data

### 3. Adaptability
- Automatically learn prior distribution from training data
- Can adapt to different experimental conditions

### 4. Uncertainty Quantification
- Provide uncertainty in parameter estimation
- Posterior distribution reflects estimation confidence

## Application Scenarios

### 1. Complex Biological Samples
- Samples with multiple fluorescence lifetime components
- Samples with high environmental heterogeneity

### 2. Low Signal-to-Noise Ratio Data
- Experiments with few photons
- Cases with high background noise

### 3. High-Resolution Imaging
- Require pixel-level precise analysis
- Spatial heterogeneity analysis

## Comparison with Traditional Methods

| Aspect | Traditional Least Squares | Nonparametric Bayesian Method |
|--------|---------------------------|-------------------------------|
| **Parameter Assumptions** | Fixed number of components | Adaptive number of components |
| **Noise Handling** | Sensitive to noise | More robust |
| **Uncertainty** | Point estimates | Probability distributions |
| **Prior Knowledge** | Not used | Automatically learned |
| **Computational Complexity** | Low | Medium |

## Notes

### 1. Computation Time
- Training phase requires longer time (depends on number of training samples)
- Analysis phase is relatively fast

### 2. Memory Usage
- Large number of training samples may occupy more memory
- Recommend adjusting parameters based on system configuration

### 3. Parameter Selection
- `num_tau_points`: Affects resolution and computation time
- `significance_threshold`: Affects component detection sensitivity

## Extension Features

### 1. Add New Noise Models
```python
def custom_noise_model(self, histogram, noise_type):
    # Implement custom noise model
    pass
```

### 2. Support More Fluorescence Models
```python
def multi_exponential_model(self, taus, weights):
    # Implement multi-exponential model
    pass
```

### 3. Integrate Instrument Response Function
```python
def add_irf_convolution(self, histogram, irf):
    # Add IRF convolution
    pass
```

## References

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Laird, N. M. (1978). Nonparametric maximum likelihood estimation of a mixing distribution.
3. Bayesian methods in fluorescence lifetime imaging microscopy.

## Contact

For questions or suggestions, please contact the development team.

## License

This project is licensed under the MIT License.
