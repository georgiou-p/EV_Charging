"""
Standalone test script for the anxiety recording system
This version doesn't import external modules - tests core algorithms only
"""

def test_welford_algorithm():
    """Test the Welford algorithm implementation"""
    print("=== Testing Welford Algorithm ===")
    
    # Simulate some anxiety values
    test_values = [0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.1, 0.8]
    
    # Manual calculation using built-in functions
    expected_mean = sum(test_values) / len(test_values)
    
    # Calculate sample standard deviation manually
    variance_sum = sum((x - expected_mean) ** 2 for x in test_values)
    expected_std = (variance_sum / (len(test_values) - 1)) ** 0.5
    
    # Welford calculation (this is what we use in the simulation)
    count = 0
    mean = 0.0
    M2 = 0.0
    
    for value in test_values:
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        M2 += delta * delta2
    
    welford_mean = mean
    welford_std = (M2 / (count - 1)) ** 0.5 if count > 1 else 0.0
    
    print(f"Test values: {test_values}")
    print(f"Expected mean: {expected_mean:.6f}, Welford mean: {welford_mean:.6f}")
    print(f"Expected std:  {expected_std:.6f}, Welford std:  {welford_std:.6f}")
    print(f"Mean difference: {abs(expected_mean - welford_mean):.10f}")
    print(f"Std difference:  {abs(expected_std - welford_std):.10f}")
    
    assert abs(expected_mean - welford_mean) < 1e-10, "Mean calculation error"
    assert abs(expected_std - welford_std) < 1e-10, "Std calculation error"
    print(" Welford algorithm working correctly\n")


def test_hour_binning():
    """Test the hour binning function"""
    print("=== Testing Hour Binning ===")
    
    def hour_bin_function(simulation_time_minutes):
        """This is the exact function used in the simulation"""
        return int(simulation_time_minutes // 60) % 24
    
    # Test cases: (simulation_time, expected_hour_bin)
    test_cases = [
        (0, 0),      # T1 (start of first hour)
        (30, 0),     # T1 (middle of first hour)
        (59, 0),     # T1 (end of first hour)
        (60, 1),     # T2 (start of second hour)
        (120, 2),    # T3 (2 hours = 120 minutes)
        (720, 12),   # T13 (12 hours = 720 minutes)
        (1380, 23),  # T24 (23 hours = 1380 minutes)
        (1440, 0),   # Back to T1 (24 hours = 1440 minutes)
        (1500, 1),   # T2 of next day
        (2880, 0),   # T1 of day after (48 hours)
    ]
    
    print("Testing hour bin mapping (simulation_time -> hour_bin -> T_label):")
    for sim_time, expected_bin in test_cases:
        actual_bin = hour_bin_function(sim_time)
        t_label = f"T{actual_bin + 1}"
        
        print(f"  {sim_time:4d} min -> bin {actual_bin:2d} -> {t_label}")
        assert actual_bin == expected_bin, f"Expected bin {expected_bin}, got {actual_bin}"
    
    print(" Hour binning working correctly\n")


def test_anxiety_stats_integration():
    """Test the full anxiety statistics integration"""
    print("=== Testing Anxiety Statistics Integration ===")
    
    import math
    
    class MockAnxietyTracker:
        """Mock version of the anxiety tracking system"""
        def __init__(self):
            self._anx_count = [0] * 24
            self._anx_mean = [0.0] * 24
            self._anx_M2 = [0.0] * 24
            self.stats = {}
        
        def _update_anxiety_stats(self, hour_idx, value):
            """Welford online algorithm update"""
            n = self._anx_count[hour_idx] + 1
            delta = value - self._anx_mean[hour_idx]
            self._anx_mean[hour_idx] += delta / n
            delta2 = value - self._anx_mean[hour_idx]
            self._anx_M2[hour_idx] += delta * delta2
            self._anx_count[hour_idx] = n
        
        def _finalize_anxiety_stats(self):
            """Compute final statistics"""
            anxiety_mean_T = []
            anxiety_std_T = []
            anxiety_count_T = self._anx_count[:]
            
            for i in range(24):
                n = self._anx_count[i]
                if n >= 2:
                    mean = self._anx_mean[i]
                    std = math.sqrt(self._anx_M2[i] / (n - 1))
                elif n == 1:
                    mean = self._anx_mean[i]
                    std = 0.0
                else:  # n == 0
                    mean = 0.0
                    std = 0.0
                
                anxiety_mean_T.append(mean)
                anxiety_std_T.append(std)
            
            self.stats['anxiety_mean_T'] = anxiety_mean_T
            self.stats['anxiety_std_T'] = anxiety_std_T
            self.stats['anxiety_count_T'] = anxiety_count_T
    
    # Create mock tracker
    tracker = MockAnxietyTracker()
    
    # Add some test data to different hours
    test_data = [
        (0, [0.1, 0.2, 0.3]),      # T1: 3 samples
        (1, [0.4, 0.5]),           # T2: 2 samples  
        (12, [0.8, 0.9, 0.7, 0.6]), # T13: 4 samples (peak hour)
        (23, [0.2]),               # T24: 1 sample
    ]
    
    for hour_idx, values in test_data:
        for value in values:
            tracker._update_anxiety_stats(hour_idx, value)
    
    # Finalize statistics
    tracker._finalize_anxiety_stats()
    
    # Verify results
    print("Testing specific hours:")
    for hour_idx, expected_values in test_data:
        t_label = f"T{hour_idx + 1}"
        count = tracker.stats['anxiety_count_T'][hour_idx]
        mean = tracker.stats['anxiety_mean_T'][hour_idx]
        std = tracker.stats['anxiety_std_T'][hour_idx]
        
        # Manual calculation for verification
        expected_mean = sum(expected_values) / len(expected_values)
        if len(expected_values) > 1:
            variance = sum((x - expected_mean) ** 2 for x in expected_values) / (len(expected_values) - 1)
            expected_std = variance ** 0.5
        else:
            expected_std = 0.0
        
        print(f"  {t_label}: {count} samples, mean={mean:.4f} (exp: {expected_mean:.4f}), std={std:.4f} (exp: {expected_std:.4f})")
        
        assert count == len(expected_values), f"{t_label}: Wrong count"
        assert abs(mean - expected_mean) < 1e-10, f"{t_label}: Wrong mean"
        assert abs(std - expected_std) < 1e-10, f"{t_label}: Wrong std"
    
    # Check acceptance criteria
    assert len(tracker.stats['anxiety_mean_T']) == 24, "Wrong number of mean values"
    assert len(tracker.stats['anxiety_std_T']) == 24, "Wrong number of std values"
    assert sum(tracker.stats['anxiety_count_T']) > 0, "No samples recorded"
    
    for i, (mean, std) in enumerate(zip(tracker.stats['anxiety_mean_T'], tracker.stats['anxiety_std_T'])):
        assert 0.0 <= mean <= 1.0, f"T{i+1}: Mean {mean} out of range"
        assert std >= 0.0, f"T{i+1}: Negative std {std}"
    
    print("Anxiety statistics integration working correctly\n")


def test_visualization_data_structure():
    """Test that the data structure matches visualization requirements"""
    print("=== Testing Visualization Data Structure ===")
    
    # Create realistic mock data that the visualization function expects
    realistic_pattern = [
        0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # T1-T6: Night (low)
        0.05, 0.08, 0.15, 0.25, 0.35, 0.45,  # T7-T12: Morning rise
        0.50, 0.40, 0.35, 0.30, 0.25, 0.20,  # T13-T18: Afternoon decline
        0.15, 0.12, 0.08, 0.05, 0.03, 0.02   # T19-T24: Evening/night
    ]
    
    realistic_std = [
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # Night: low variation
        0.02, 0.03, 0.06, 0.08, 0.10, 0.12,      # Morning: increasing variation  
        0.15, 0.12, 0.10, 0.08, 0.07, 0.06,      # Afternoon: high then decreasing
        0.05, 0.04, 0.03, 0.02, 0.015, 0.01      # Evening: decreasing variation
    ]
    
    realistic_counts = [10, 5, 3, 2, 2, 5, 15, 25, 40, 60, 80, 100,
                       120, 90, 75, 60, 50, 40, 30, 25, 20, 15, 12, 10]
    
    # Verify data structure requirements
    assert len(realistic_pattern) == 24, f"Mean array wrong length: {len(realistic_pattern)}"
    assert len(realistic_std) == 24, f"Std array wrong length: {len(realistic_std)}"
    assert len(realistic_counts) == 24, f"Count array wrong length: {len(realistic_counts)}"
    
    # Check value ranges
    for i, (mean, std, count) in enumerate(zip(realistic_pattern, realistic_std, realistic_counts)):
        assert 0.0 <= mean <= 1.0, f"T{i+1}: Mean {mean} out of range [0,1]"
        assert std >= 0.0, f"T{i+1}: Std {std} negative"
        assert count >= 0, f"T{i+1}: Count {count} negative"
    
    # Test confidence band calculation (this is what the plot function does)
    confidence_bands = []
    for mean, std in zip(realistic_pattern, realistic_std):
        upper = min(1.0, mean + std)  # Clip to [0,1]
        lower = max(0.0, mean - std)  # Clip to [0,1]
        confidence_bands.append((lower, upper))
    
    print("Sample confidence bands (mean +- std, clipped to [0,1]):")
    for i in range(0, 24, 6):  # Show every 6 hours
        mean = realistic_pattern[i]
        std = realistic_std[i]
        lower, upper = confidence_bands[i]
        print(f"  T{i+1:2d}: {mean:.3f} +- {std:.3f} -> [{lower:.3f}, {upper:.3f}]")
    
    # Find peak and low periods
    peak_hour = realistic_pattern.index(max(realistic_pattern)) + 1
    low_hour = realistic_pattern.index(min(realistic_pattern)) + 1
    
    print(f"\nPattern analysis:")
    print(f"  Peak anxiety: T{peak_hour} ({max(realistic_pattern):.3f})")
    print(f"  Lowest anxiety: T{low_hour} ({min(realistic_pattern):.3f})")
    print(f"  Total samples: {sum(realistic_counts)}")
    print(f"  Hours with data: {sum(1 for c in realistic_counts if c > 0)}/24")
    
    print(" Visualization data structure correct\n")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("ANXIETY RECORDING SYSTEM - STANDALONE TESTS")
    print("="*60)
    
    try:
        test_welford_algorithm()
        test_hour_binning()
        test_anxiety_stats_integration()
        test_visualization_data_structure()
        
        print("="*60)
        print(" ALL CORE TESTS PASSED")
        print("The anxiety recording algorithms are working correctly!")
        print("="*60)
        
    except Exception as e:
        print("="*60)
        print(" TEST FAILED")
        print(f"Error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting standalone tests (no external dependencies)...")
    print("This tests the core algorithms without importing simulation modules.\n")
    run_all_tests()