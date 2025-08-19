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
        # Linear mapping: threshold [0.20, 0.70] -> patience [0.20, 1.00]
        min_threshold = 0.20
        max_threshold = 0.70
        min_patience = 0.20  # Most patient 
        max_patience = 1.00  # Least patient 
        
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

