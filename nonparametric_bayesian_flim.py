"""
Nonparametric Bayesian Fluorescence Lifetime Analysis (Nonparametric Bayesian FLIM Analysis)

Features:
1. Simulate dual-exponential decay photon histogram data (with background)
2. Build probability library P(τ) for possible lifetimes
3. Perform Non-Parametric Maximum Likelihood Estimation (NPMLE) to obtain prior distribution
4. Apply Expectation-Maximization (EM) for Maximum A Posteriori (MAP) estimation to recover parameters for each pixel

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import itertools
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NonparametricBayesianFLIM:
    """
    Nonparametric Bayesian Fluorescence Lifetime Analysis Class
    """
    
    def __init__(self, tau_range: Tuple[float, float] = (0.1, 10.0), 
                 num_tau_points: int = 100, time_range: float = 12.5, 
                 num_channels: int = 256):
        """
        Initialize parameters
        
        Args:
            tau_range: Lifetime range (ns)
            num_tau_points: Number of points in lifetime space
            time_range: Time range (ns)
            num_channels: Number of time channels
        """
        self.tau_range = tau_range
        self.num_tau_points = num_tau_points
        self.time_range = time_range
        self.num_channels = num_channels
        
        # Build lifetime space (logarithmic distribution)
        self.tau_space = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), num_tau_points)
        
        # Build time channels
        self.time_channels = np.linspace(0, time_range, num_channels + 1)
        self.time_channel_centers = np.linspace(0, time_range, num_channels)
        
        # Initialize prior distribution
        self.prior_distribution = None
        
    def simulate_dual_exponential_histogram(self, tau1: float, tau2: float, 
                                          fraction1: float, total_photons: int = 100000,
                                          background_ratio: float = 0.1,
                                          noise_level: float = 0.05) -> np.ndarray:
        """
        Simulate dual-exponential decay photon histogram
        
        Args:
            tau1: First lifetime component (ns)
            tau2: Second lifetime component (ns)
            fraction1: Proportion of first component
            total_photons: Total number of photons
            background_ratio: Background photon ratio
            noise_level: Noise level
            
        Returns:
            Photon histogram array
        """
        # Calculate ideal dual-exponential decay
        ideal_decay = (fraction1 * np.exp(-self.time_channel_centers / tau1) + 
                      (1 - fraction1) * np.exp(-self.time_channel_centers / tau2))
        
        # Normalize
        ideal_decay = ideal_decay / np.sum(ideal_decay)
        
        # Allocate photon numbers
        signal_photons = int(total_photons * (1 - background_ratio))
        background_photons = int(total_photons * background_ratio)
        
        # Generate signal photons
        signal_histogram = np.random.poisson(ideal_decay * signal_photons)
        
        # Generate background photons (uniform distribution)
        background_histogram = np.random.poisson(background_photons / self.num_channels, 
                                               size=self.num_channels)
        
        # Combine signal and background
        total_histogram = signal_histogram + background_histogram
        
        # Add additional noise
        noise = np.random.normal(0, noise_level * np.sqrt(total_histogram))
        total_histogram = np.maximum(0, total_histogram + noise).astype(int)
        
        return total_histogram
    
    def generate_training_data(self, num_samples: int = 1000) -> List[np.ndarray]:
        """
        Generate training data for building prior distribution
        
        Args:
            num_samples: Number of samples
            
        Returns:
            List of training histograms
        """
        print(f"Generating {num_samples} training samples...")
        
        training_histograms = []
        
        for i in range(num_samples):
            # Randomize parameters
            tau1 = np.random.uniform(1.5, 3.0)  # Long lifetime range
            tau2 = np.random.uniform(0.5, 1.2)  # Short lifetime range
            fraction1 = np.random.uniform(0.4, 0.8)  # Proportion range
            total_photons = np.random.uniform(50000, 200000)  # Photon number range
            background_ratio = np.random.uniform(0.05, 0.2)  # Background ratio range
            
            # Generate histogram
            histogram = self.simulate_dual_exponential_histogram(
                tau1, tau2, fraction1, int(total_photons), background_ratio
            )
            
            training_histograms.append(histogram)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples")
        
        return training_histograms
    
    def build_probability_library(self, histograms: List[np.ndarray]) -> np.ndarray:
        """
        Build lifetime probability library
        
        Args:
            histograms: List of training histograms
            
        Returns:
            Lifetime probability array
        """
        print("Building lifetime probability library (fast scoring)...")

        # Precompute decay dictionary over tau grid (L x C)
        t = self.time_channel_centers  # (C,)
        tau_space = self.tau_space     # (L,)
        L = len(tau_space)
        eps = 1e-12

        # E[l, c] = exp(-t[c] / tau_space[l])
        E = np.exp(-np.outer(1.0 / np.clip(tau_space, eps, None), t))  # (L, C)
        # Normalize each tau row to sum 1 (probability shape)
        E_sum = np.clip(E.sum(axis=1, keepdims=True), eps, None)
        E_norm = E / E_sum  # (L, C)

        tau_votes = np.zeros(L, dtype=float)

        for i, y in enumerate(histograms):
            y = np.asarray(y, dtype=float)
            total = max(float(y.sum()), 1.0)
            # Expected counts for each tau: lam = total * E_norm
            lam = E_norm * total + eps  # (L, C)
            # Poisson negative log-likelihood per tau
            # nll = sum(lam - y*log(lam)) along channels
            nll = lam.sum(axis=1) - (y * np.log(lam)).sum(axis=1)
            # Pick the best tau (smallest nll) and vote
            best_idx = int(np.argmin(nll))
            tau_votes[best_idx] += 1.0

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} histograms")

        # Normalize votes to probability
        total_votes = tau_votes.sum()
        if total_votes > 0:
            tau_probabilities = tau_votes / total_votes
        else:
            tau_probabilities = np.ones(L, dtype=float) / L

        return tau_probabilities
    
    def fit_histogram_to_taus(self, histogram: np.ndarray, max_components: int = 3) -> List[float]:
        """
        Fit multiple possible lifetimes to histogram
        
        Args:
            histogram: Photon histogram
            max_components: Maximum number of components
            
        Returns:
            List of fitted lifetimes
        """
        best_taus = []
        
        # Try different numbers of components
        for num_components in range(1, max_components + 1):
            # Sample combinations from lifetime space
            tau_combinations = list(itertools.combinations(self.tau_space, num_components))
            
            best_likelihood = -np.inf
            best_tau_combo = None
            
            # Limit number of combinations to avoid computational overload
            max_combinations = min(100, len(tau_combinations))
            if len(tau_combinations) > max_combinations:
                selected_indices = np.random.choice(len(tau_combinations), max_combinations, replace=False)
                selected_combinations = [tau_combinations[i] for i in selected_indices]
            else:
                selected_combinations = tau_combinations
            
            for tau_combo in selected_combinations:
                # Fit this lifetime combination
                likelihood = self.calculate_likelihood(histogram, tau_combo)
                
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_tau_combo = tau_combo
            
            if best_tau_combo is not None:
                best_taus.extend(best_tau_combo)
        
        return best_taus
    
    def calculate_likelihood(self, histogram: np.ndarray, taus: Tuple[float, ...]) -> float:
        """
        Calculate likelihood for given lifetime combination
        
        Args:
            histogram: Photon histogram
            taus: Lifetime combination
            
        Returns:
            Log likelihood
        """
        # Build model
        model = np.zeros(self.num_channels)
        
        # Assume equal weights
        weight = 1.0 / len(taus)
        
        for tau in taus:
            exponential = np.exp(-self.time_channel_centers / tau)
            model += weight * exponential
        
        # Normalize
        model = model / np.sum(model)
        
        # Scale model to match total photon count
        total_photons = np.sum(histogram)
        model = model * total_photons
        
        # Calculate Poisson likelihood
        log_likelihood = np.sum(poisson.logpmf(histogram, model))
        
        return log_likelihood
    
    def npmle_estimation(self, histograms: List[np.ndarray], 
                        initial_probabilities: np.ndarray) -> np.ndarray:
        """
        Non-parametric Maximum Likelihood Estimation
        
        Args:
            histograms: List of training histograms
            initial_probabilities: Initial probability distribution
            
        Returns:
            Optimized probability distribution
        """
        print("Performing Non-parametric Maximum Likelihood Estimation...")
        
        def objective_function(weights):
            """Objective function: negative log likelihood"""
            # Build mixed distribution
            mixed_distribution = np.zeros(self.num_channels)
            
            for i, weight in enumerate(weights):
                tau = self.tau_space[i]
                exponential = np.exp(-self.time_channel_centers / tau)
                mixed_distribution += weight * exponential
            
            # Normalize
            mixed_distribution = mixed_distribution / np.sum(mixed_distribution)
            
            # Calculate log likelihood
            log_likelihood = 0
            for histogram in histograms:
                # Scale model
                total_photons = np.sum(histogram)
                model = mixed_distribution * total_photons
                
                # Poisson likelihood
                log_likelihood += np.sum(poisson.logpmf(histogram, model))
            
            return -log_likelihood
        
        # Constraints: weights sum to 1, weights non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0, 1)] * self.num_tau_points
        
        # Initial guess
        initial_weights = initial_probabilities.copy()
        
        # Optimization
        result = minimize(objective_function, initial_weights, 
                         constraints=constraints, bounds=bounds,
                         method='SLSQP', options={'maxiter': 1000})
        
        if result.success:
            print("NPMLE optimization successful")
            return result.x
        else:
            print("NPMLE optimization failed, using initial probabilities")
            return initial_probabilities
    
    def em_map_estimation(self, pixel_histogram: np.ndarray, 
                         max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        Expectation-Maximization Maximum A Posteriori Estimation
        
        Args:
            pixel_histogram: Pixel histogram
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Posterior weight distribution
        """
        if self.prior_distribution is None:
            raise ValueError("Please train prior distribution first")
        
        # Initialize weights
        current_weights = np.ones(self.num_tau_points) / self.num_tau_points
        
        for iteration in range(max_iterations):
            # E-step: Calculate expectations
            responsibilities = self.expectation_step(pixel_histogram, current_weights)
            
            # M-step: Maximize posterior
            new_weights = self.maximization_step(responsibilities, pixel_histogram)
            
            # Check convergence
            if np.allclose(current_weights, new_weights, rtol=tolerance):
                break
                
            current_weights = new_weights
        
        return current_weights
    
    def expectation_step(self, histogram: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Expectation step
        
        Args:
            histogram: Photon histogram
            weights: Current weights
            
        Returns:
            Responsibility matrix
        """
        responsibilities = np.zeros((self.num_tau_points, self.num_channels))
        
        for i, tau in enumerate(self.tau_space):
            exponential = np.exp(-self.time_channel_centers / tau)
            responsibilities[i] = weights[i] * exponential
        
        # Normalize
        responsibilities = responsibilities / np.sum(responsibilities, axis=0, keepdims=True)
        
        return responsibilities
    
    def maximization_step(self, responsibilities: np.ndarray, histogram: np.ndarray) -> np.ndarray:
        """
        Maximization step (including prior)
        
        Args:
            responsibilities: Responsibility matrix
            histogram: Photon histogram
            
        Returns:
            New weight distribution
        """
        # Calculate posterior weights
        posterior_weights = np.sum(responsibilities * histogram[np.newaxis, :], axis=1)
        
        # Combine with prior
        posterior_weights = posterior_weights * self.prior_distribution
        
        # Normalize
        posterior_weights = posterior_weights / np.sum(posterior_weights)
        
        return posterior_weights
    
    def extract_significant_components(self, posterior_weights: np.ndarray, 
                                     threshold: float = 0.01) -> List[Dict]:
        """
        Extract significant components
        
        Args:
            posterior_weights: Posterior weights
            threshold: Significance threshold
            
        Returns:
            List of significant components
        """
        significant_components = []
        
        # Find components above threshold
        significant_indices = np.where(posterior_weights > threshold)[0]
        
        for idx in significant_indices:
            component = {
                'tau': self.tau_space[idx],
                'weight': posterior_weights[idx],
                'index': idx
            }
            significant_components.append(component)
        
        # Sort by weight
        significant_components.sort(key=lambda x: x['weight'], reverse=True)
        
        return significant_components
    
    def train_prior(self, training_histograms: List[np.ndarray]) -> np.ndarray:
        """
        Train prior distribution
        
        Args:
            training_histograms: List of training histograms
            
        Returns:
            Prior distribution
        """
        print("Starting prior distribution training...")
        
        # Step 1: Build probability library
        tau_probabilities = self.build_probability_library(training_histograms)
        
        # Step 2: NPMLE estimation
        self.prior_distribution = self.npmle_estimation(training_histograms, tau_probabilities)
        
        print("Prior distribution training completed")
        return self.prior_distribution
    
    def analyze_pixel(self, pixel_histogram: np.ndarray) -> Dict:
        """
        Analyze single pixel
        
        Args:
            pixel_histogram: Pixel histogram
            
        Returns:
            Analysis result
        """
        # Use EM-MAP estimation
        posterior_weights = self.em_map_estimation(pixel_histogram)
        
        # Extract main lifetime components
        significant_components = self.extract_significant_components(posterior_weights)
        
        # Calculate mean lifetime
        mean_tau = np.sum(posterior_weights * self.tau_space)
        
        result = {
            'posterior_weights': posterior_weights,
            'significant_components': significant_components,
            'mean_tau': mean_tau,
            'num_components': len(significant_components)
        }
        
        return result
    
    def analyze_image(self, image_histograms: List[np.ndarray]) -> List[Dict]:
        """
        Analyze entire image
        
        Args:
            image_histograms: List of image histograms
            
        Returns:
            List of analysis results
        """
        print(f"Analyzing {len(image_histograms)} pixels...")
        
        results = []
        for i, pixel_histogram in enumerate(image_histograms):
            pixel_result = self.analyze_pixel(pixel_histogram)
            results.append(pixel_result)
            
            if (i + 1) % 100 == 0:
                print(f"Analyzed {i + 1} pixels")
        
        return results
    
    def plot_prior_distribution(self, save_path: Optional[str] = None):
        """
        Plot prior distribution
        
        Args:
            save_path: Save path
        """
        if self.prior_distribution is None:
            print("Please train prior distribution first")
            return
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.tau_space, self.prior_distribution, 'b-', linewidth=2)
        plt.xlabel('Lifetime (ns)')
        plt.ylabel('Prior Probability')
        plt.title('Nonparametric Prior Distribution')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_analysis_result(self, pixel_histogram: np.ndarray, result: Dict, 
                           save_path: Optional[str] = None):
        """
        Plot analysis result
        
        Args:
            pixel_histogram: Pixel histogram
            result: Analysis result
            save_path: Save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original data
        axes[0, 0].plot(self.time_channel_centers, pixel_histogram, 'ko-', markersize=3)
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('Photon Counts')
        axes[0, 0].set_title('Original Histogram')
        axes[0, 0].grid(True)
        
        # Posterior distribution
        axes[0, 1].semilogx(self.tau_space, result['posterior_weights'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Lifetime (ns)')
        axes[0, 1].set_ylabel('Posterior Weight')
        axes[0, 1].set_title('Posterior Distribution')
        axes[0, 1].grid(True)
        
        # Significant components
        if result['significant_components']:
            taus = [comp['tau'] for comp in result['significant_components']]
            weights = [comp['weight'] for comp in result['significant_components']]
            axes[1, 0].bar(range(len(taus)), weights)
            axes[1, 0].set_xlabel('Component Index')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Significant Components')
            axes[1, 0].set_xticks(range(len(taus)))
            axes[1, 0].set_xticklabels([f'{tau:.2f}ns' for tau in taus])
        
        # Fit result
        model = np.zeros(self.num_channels)
        for comp in result['significant_components']:
            tau = comp['tau']
            weight = comp['weight']
            exponential = np.exp(-self.time_channel_centers / tau)
            model += weight * exponential
        
        # Normalize and scale
        model = model / np.sum(model)
        total_photons = np.sum(pixel_histogram)
        model = model * total_photons
        
        axes[1, 1].plot(self.time_channel_centers, pixel_histogram, 'ko-', markersize=3, label='Data')
        axes[1, 1].plot(self.time_channel_centers, model, 'r-', linewidth=2, label='Fit')
        axes[1, 1].set_xlabel('Time (ns)')
        axes[1, 1].set_ylabel('Photon Counts')
        axes[1, 1].set_title(f'Fit Result (Mean τ = {result["mean_tau"]:.2f}ns)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """
    Main function: Demonstrate nonparametric Bayesian fluorescence lifetime analysis
    """
    print("=== Nonparametric Bayesian Fluorescence Lifetime Analysis Demo ===\n")
    
    # 1. Initialize analyzer
    analyzer = NonparametricBayesianFLIM(
        tau_range=(0.1, 10.0),
        num_tau_points=50,
        time_range=12.5,
        num_channels=256
    )
    
    # 2. Generate training data
    training_histograms = analyzer.generate_training_data(num_samples=500)
    
    # 3. Train prior distribution
    prior = analyzer.train_prior(training_histograms)
    
    # 4. Plot prior distribution
    analyzer.plot_prior_distribution(save_path='prior_distribution.png')
    
    # 5. Generate test data
    print("\nGenerating test data...")
    test_histogram = analyzer.simulate_dual_exponential_histogram(
        tau1=2.14, tau2=0.69, fraction1=0.6, total_photons=100000
    )
    
    # 6. Analyze test data
    print("Analyzing test data...")
    result = analyzer.analyze_pixel(test_histogram)
    
    # 7. Display results
    print(f"\nAnalysis results:")
    print(f"Mean lifetime: {result['mean_tau']:.3f} ns")
    print(f"Number of significant components: {result['num_components']}")
    
    for i, comp in enumerate(result['significant_components']):
        print(f"Component {i+1}: τ = {comp['tau']:.3f} ns, weight = {comp['weight']:.3f}")
    
    # 8. Plot analysis results
    analyzer.plot_analysis_result(test_histogram, result, save_path='analysis_result.png')
    
    print("\nAnalysis completed! Results saved to image files.")


if __name__ == "__main__":
    main()
