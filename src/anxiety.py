"""
Simple anxiety model based on personal SoC threshold and derived patience level.
Formula: Anxiety A = (k × P) / (k_max × P_max)
Where k is patience level, P is sum of penalties in minutes.
"""

import numpy as np

class Anxiety:
    """
    Simple anxiety model with personal threshold and derived patience.
    """
    
    def __init__(self, threshold=None, patience_level=None):
        """
        Initialize anxiety model with personal threshold and patience.
        
        Args:
            threshold: Personal SoC threshold (0.0-1.0). If None, samples from Beta distribution.
            patience_level: Patience level k (0.2-1.0). If None, derived from threshold.
        """
        # Sample threshold from Beta distribution to match histogram
        if threshold is None:
            self.threshold = self._sample_threshold()
        else:
            self.threshold = max(0.20, min(0.70, threshold))
        
        # Derive patience level from threshold (monotonic relationship)
        if patience_level is None:
            self.patience_level = self._derive_patience(self.threshold)
        else:
            self.patience_level = max(0.20, min(1.00, patience_level))
        
        # Constants for anxiety computation
        self.k_max = 1.0
        self.p_max = 120.0  # 120 minutes max penalty
    
    def _sample_threshold(self):
        """
        Sample threshold from Beta distribution to match histogram.
        Target: 20%-70% range, peaking around 45%.
        
        Using Beta(5, 6) which peaks around 0.45 when scaled to [0.2, 0.7].
        
        Returns:
            float: Threshold in range [0.20, 0.70]
        """
        # Beta(5, 6) parameters chosen to peak around 0.45
        # Beta(a, b) has mode at (a-1)/(a+b-2) = 4/9 ≈ 0.44
        beta_sample = np.random.beta(5, 6)
        
        # Scale from [0,1] to [0.20, 0.70]
        min_threshold = 0.20
        max_threshold = 0.70
        
        scaled_threshold = min_threshold + beta_sample * (max_threshold - min_threshold)
        return scaled_threshold
    
    def _derive_patience(self, threshold):
        """
        Derive patience level k from threshold (monotonic relationship).
        Lower threshold → more patient (smaller k)
        Higher threshold → less patient (larger k)
        
        Args:
            threshold: SoC threshold (0.20-0.70)
            
        Returns:
            float: Patience level k in range [0.20, 1.00]
        """
        # Linear mapping: threshold [0.20, 0.70] → patience [0.20, 1.00]
        min_threshold = 0.20
        max_threshold = 0.70
        min_patience = 0.20  # Most patient (anxious drivers)
        max_patience = 1.00  # Least patient (confident drivers)
        
        # Normalize threshold to [0, 1]
        normalized = (threshold - min_threshold) / (max_threshold - min_threshold)
        normalized = max(0.0, min(1.0, normalized))
        
        # Linear mapping to patience range
        patience = min_patience + normalized * (max_patience - min_patience)
        
        return patience
    
    def compute(self, penalties_min):
        """
        Compute anxiety A = (k × P) / (k_max × P_max).
        
        Args:
            penalties_min: Sum of penalties in minutes (queue + detour + slow-charger time)
            
        Returns:
            float: Anxiety A in range [0, 1]
        """
        # Ensure penalties are non-negative
        penalties_min = max(0.0, penalties_min)
        
        # Apply formula: A = (k × P) / (k_max × P_max)
        anxiety = (self.patience_level * penalties_min) / (self.k_max * self.p_max)
        
        # Clamp to [0, 1] range
        anxiety = max(0.0, min(1.0, anxiety))
        
        return anxiety
    
    def should_search(self, soc):
        """
        Determine if driver should search for charging based on SoC threshold only.
        
        Args:
            soc: Current state of charge (0.0-1.0)
            
        Returns:
            bool: True if soc ≤ threshold
        """
        return soc <= self.threshold
    
    def get_threshold(self):
        """Get personal SoC threshold"""
        return self.threshold
    
    def get_patience_level(self):
        """Get patience level k"""
        return self.patience_level
    
    def get_threshold_percentage(self):
        """Get threshold as percentage for display"""
        return self.threshold * 100
    
    def get_profile_description(self):
        """Get human-readable description of this anxiety profile"""
        threshold_pct = self.get_threshold_percentage()
        
        if threshold_pct <= 30:
            anxiety_type = "Very Anxious"
        elif threshold_pct <= 40:
            anxiety_type = "Anxious"
        elif threshold_pct <= 50:
            anxiety_type = "Moderate"
        elif threshold_pct <= 60:
            anxiety_type = "Confident"
        else:
            anxiety_type = "Very Confident"
        
        return f"{anxiety_type} (threshold={threshold_pct:.0f}%, k={self.patience_level:.2f})"
    
    def __str__(self):
        return f"Anxiety(threshold={self.threshold:.3f}, patience_level={self.patience_level:.3f})"
    
    def __repr__(self):
        return self.__str__()


def create_test_profiles(num_profiles):
    """
    Create multiple anxiety profiles for testing.
    
    Args:
        num_profiles: Number of profiles to create
        
    Returns:
        list: List of Anxiety objects
    """
    profiles = []
    for _ in range(num_profiles):
        profiles.append(Anxiety())
    return profiles


def analyze_distribution(profiles):
    """
    Analyze the distribution of anxiety profiles.
    
    Args:
        profiles: List of Anxiety objects
        
    Returns:
        dict: Statistics about the distribution
    """
    thresholds = [p.get_threshold() for p in profiles]
    patience_levels = [p.get_patience_level() for p in profiles]
    
    stats = {
        'threshold_mean': np.mean(thresholds),
        'threshold_std': np.std(thresholds),
        'threshold_min': np.min(thresholds),
        'threshold_max': np.max(thresholds),
        'patience_mean': np.mean(patience_levels),
        'patience_std': np.std(patience_levels),
        'patience_min': np.min(patience_levels),
        'patience_max': np.max(patience_levels)
    }
    
    return stats


if __name__ == "__main__":
    # Test the anxiety module
    print("=== Testing New Anxiety Module ===\n")
    
    # Test Beta distribution parameters
    print("Beta(5, 6) distribution test:")
    test_samples = [np.random.beta(5, 6) for _ in range(1000)]
    scaled_samples = [0.20 + s * 0.50 for s in test_samples]
    
    print(f"Raw Beta samples: mean={np.mean(test_samples):.3f}, should be ~0.44")
    print(f"Scaled samples: mean={np.mean(scaled_samples):.3f}, should be ~0.45")
    print(f"Scaled range: {np.min(scaled_samples):.3f} - {np.max(scaled_samples):.3f}")
    
    # Create test profiles
    print(f"\nCreating 10 test anxiety profiles:")
    profiles = create_test_profiles(10)
    
    for i, profile in enumerate(profiles, 1):
        print(f"{i:2d}. {profile.get_profile_description()}")
    
    # Test anxiety computation
    print(f"\nTesting anxiety computation:")
    test_profile = Anxiety(threshold=0.45, patience_level=0.6)
    print(f"Test profile: {test_profile}")
    
    penalty_tests = [0, 30, 60, 120, 180]
    for penalties in penalty_tests:
        anxiety = test_profile.compute(penalties)
        print(f"  Penalties: {penalties:3d}min - Anxiety: {anxiety:.3f}")
    
    # Test search decision
    print(f"\nTesting search decisions (threshold = {test_profile.threshold:.2f}):")
    soc_tests = [0.8, 0.6, 0.45, 0.4, 0.2]
    for soc in soc_tests:
        should_search = test_profile.should_search(soc)
        print(f"  SoC {soc*100:2.0f}%: {'SEARCH' if should_search else 'continue'}")
    
    # Distribution analysis
    print(f"\n=== Distribution Analysis (1000 profiles) ===")
    large_sample = create_test_profiles(1000)
    stats = analyze_distribution(large_sample)
    
    print(f"Threshold Distribution:")
    print(f"  Mean: {stats['threshold_mean']:.3f} ({stats['threshold_mean']*100:.1f}%) - target ~45%")
    print(f"  Range: {stats['threshold_min']:.3f} - {stats['threshold_max']:.3f}")
    
    print(f"Patience Distribution:")
    print(f"  Mean: {stats['patience_mean']:.3f}")
    print(f"  Range: {stats['patience_min']:.3f} - {stats['patience_max']:.3f}")
    
    # Verify monotonic relationship
    import matplotlib.pyplot as plt
    thresholds_sorted = sorted([p.get_threshold() for p in large_sample[:100]])
    patience_sorted = [Anxiety(t).get_patience_level() for t in thresholds_sorted]
    correlation = np.corrcoef(thresholds_sorted, patience_sorted)[0, 1]
    print(f"\nMonotonic relationship verification:")
    print(f"  Correlation (threshold, patience): {correlation:.3f} (should be positive)")