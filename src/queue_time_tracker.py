import numpy as np
import matplotlib.pyplot as plt

class QueueTimeTracker:
    """
    Tracks queue times by hour with quartile statistics
    """
    def __init__(self):
        # Store queue times for each hour T1-T24 (indices 0-23)
        self.hourly_queue_times = [[] for _ in range(24)]
        
        # Store driver counts for each hour
        self.hourly_driver_counts = [[] for _ in range(24)]
        
        # Track when we sample driver counts
        self.last_driver_count_time = 0
        self.driver_count_interval = 10  # Sample every 10 minutes
    
    def _get_hour_index(self, simulation_time):
        """Convert simulation time (minutes) to hour index 0-23"""
        return int(simulation_time // 60) % 24
    
    def record_queue_time(self, simulation_time, queue_time_minutes):
        """Record a queue time for the appropriate hour"""
        if simulation_time >= 1440:
            print(f"DEBUG: Ignoring queue event at time {simulation_time:.1f} (after 1440 min cutoff)")
            return
        hour_idx = self._get_hour_index(simulation_time)
        self.hourly_queue_times[hour_idx].append(queue_time_minutes)
    
    def record_driver_count(self, simulation_time, active_drivers):
        """Record active driver count for the appropriate hour"""
        if simulation_time >= 1440:
            return
        hour_idx = self._get_hour_index(simulation_time)
        self.hourly_driver_counts[hour_idx].append(active_drivers)
    
    def should_sample_driver_count(self, simulation_time):
        """Check if it's time to sample driver count"""
        if simulation_time >= 1440:
            return
        if simulation_time - self.last_driver_count_time >= self.driver_count_interval:
            self.last_driver_count_time = simulation_time
            return True
        return False
    
    def calculate_hourly_statistics(self):
        """
        Calculate statistics for each hour
        Returns: (hours, avg_queue_times, q1_queue_times, q3_queue_times, avg_driver_counts, sample_counts)
        """
        hours = list(range(1, 25))  # T1 to T24
        avg_queue_times = []
        q1_queue_times = []
        q3_queue_times = []
        avg_driver_counts = []
        sample_counts = []
        
        for hour_idx in range(24):
            queue_times = self.hourly_queue_times[hour_idx]
            driver_counts = self.hourly_driver_counts[hour_idx]
            
            if queue_times:
                # Queue time statistics
                queue_array = np.array(queue_times)
                avg_queue = np.mean(queue_array)
                q1_queue = np.percentile(queue_array, 25)
                q3_queue = np.percentile(queue_array, 75)
                
                avg_queue_times.append(avg_queue)
                q1_queue_times.append(q1_queue)
                q3_queue_times.append(q3_queue)
                sample_counts.append(len(queue_times))
            else:
                avg_queue_times.append(0)
                q1_queue_times.append(0)
                q3_queue_times.append(0)
                sample_counts.append(0)
            
            if driver_counts:
                # Driver count statistics
                avg_drivers = np.mean(driver_counts)
                avg_driver_counts.append(avg_drivers)
            else:
                avg_driver_counts.append(0)
        
        return hours, avg_queue_times, q1_queue_times, q3_queue_times, avg_driver_counts, sample_counts
    
    def print_hourly_summary(self):
        """Print a detailed hourly summary table"""
        hours, avg_queue, q1_queue, q3_queue, avg_drivers, counts = self.calculate_hourly_statistics()
        
        print(f"\n{'=' * 100}")
        print(f"{'HOURLY QUEUE TIME SUMMARY':^100}")
        print(f"{'=' * 100}")
        print(f"{'Hour':<6} {'Time Range':<12} {'Samples':<8} {'Avg Queue':<12} {'Q1 (25%)':<10} {'Q3 (75%)':<10} {'IQR':<10} {'Avg Drivers':<12}")
        print(f"{'-' * 100}")
        
        for i in range(24):
            hour_label = f"T{i+1}"
            time_range = f"{i:02d}:00-{(i+1)%24:02d}:00"
            samples = counts[i]
            avg_q = avg_queue[i]
            q1_q = q1_queue[i]
            q3_q = q3_queue[i]
            iqr = q3_q - q1_q if samples > 0 else 0
            avg_d = avg_drivers[i]
            
            if samples > 0:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {avg_q:<12.2f} {q1_q:<10.2f} {q3_q:<10.2f} {iqr:<10.2f} {avg_d:<12.1f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {avg_d:<12.1f}")
        
        print(f"{'-' * 100}")
        total_samples = sum(counts)
        hours_with_data = sum(1 for c in counts if c > 0)
        print(f"Total queue events: {total_samples}, Hours with queue data: {hours_with_data}/24")
        
        if total_samples > 0:
            overall_avg = sum(avg_queue[i] * counts[i] for i in range(24) if counts[i] > 0) / total_samples
            print(f"Overall average queue time: {overall_avg:.2f} minutes")
        
        print(f"{'=' * 100}")
    
    def plot_hourly_queue_times(self, title="Queue Times Throughout the Day"):
        """
        Create a comprehensive plot showing queue times and driver counts
        """
        hours, avg_queue, q1_queue, q3_queue, avg_drivers, counts = self.calculate_hourly_statistics()
        
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(16, 10))
        ax2 = ax1.twinx()
        
        # Convert hours to x-axis values
        x = np.array(hours)
        
        # Plot queue times with quartiles
        avg_queue = np.array(avg_queue)
        q1_queue = np.array(q1_queue)
        q3_queue = np.array(q3_queue)
        
        # Fill between Q1 and Q3 (interquartile range)
        ax1.fill_between(x, q1_queue, q3_queue, alpha=0.3, color='lightcoral', 
                        label='Interquartile Range (Q1-Q3)')
        
        # Plot average queue time line
        line1 = ax1.plot(x, avg_queue, color='darkred', linewidth=3, 
                        marker='o', markersize=6, label='Average Queue Time')
        
        # Plot Q1 and Q3 lines
        ax1.plot(x, q1_queue, color='red', linewidth=1, linestyle='--', 
                alpha=0.7, label='Q1 (25th percentile)')
        ax1.plot(x, q3_queue, color='red', linewidth=1, linestyle='--', 
                alpha=0.7, label='Q3 (75th percentile)')
        
        # Plot driver count
        avg_drivers = np.array(avg_drivers)
        line2 = ax2.plot(x, avg_drivers, color='steelblue', linewidth=2.5,
                        marker='s', markersize=5, linestyle='-', alpha=0.8,
                        label='Active Drivers')
        
        # Customize primary y-axis (queue times)
        ax1.set_xlabel('Time of Day (T1–T24)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Queue Time (minutes)', fontsize=14, fontweight='bold', color='darkred')
        ax1.tick_params(axis='y', labelcolor='darkred')
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'T{i}' for i in x], rotation=45, ha='right')
        
        # Set y-axis limits for queue times
        max_queue = np.max(q3_queue) if np.max(q3_queue) > 0 else 10
        ax1.set_ylim(0, max_queue * 1.1)
        
        # Add horizontal reference lines for queue times
        if max_queue > 30:
            ax1.axhline(y=30, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='30 min threshold')
        if max_queue > 60:
            ax1.axhline(y=60, color='red', linestyle=':', alpha=0.7, linewidth=2, label='60 min tolerance')
        
        # Customize secondary y-axis (driver count)
        ax2.set_ylabel('Number of Active Drivers', fontsize=14, fontweight='bold', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        
        # Set reasonable limits for driver count
        if np.max(avg_drivers) > 0:
            ax2.set_ylim(0, np.max(avg_drivers) * 1.1)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  fontsize=12, framealpha=0.9)
        
        # Add time labels on secondary x-axis
        time_labels = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                      '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                      '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                      '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
        
        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xticks(x[::3])  # Every 3 hours
        ax3.set_xticklabels([time_labels[i-1] for i in x[::3]], fontsize=10, alpha=0.7)
        ax3.set_xlabel('Hour of Day', fontsize=12, alpha=0.7)
        
        # Add annotations for peak periods
        if len(avg_queue) > 0 and np.max(avg_queue) > 0:
            # Find peak queue time
            peak_hour_idx = np.argmax(avg_queue)
            peak_hour = peak_hour_idx + 1
            peak_value = avg_queue[peak_hour_idx]
            
            if peak_value > 0:
                ax1.annotate(f'Peak Queue: T{peak_hour}\n({peak_value:.1f} min)',
                           xy=(peak_hour, peak_value),
                           xytext=(peak_hour + 2, peak_value + max_queue * 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
                           fontsize=11, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add driver count peak annotation
        if np.max(avg_drivers) > 0:
            peak_driver_hour_idx = np.argmax(avg_drivers)
            peak_driver_hour = peak_driver_hour_idx + 1
            peak_driver_count = avg_drivers[peak_driver_hour_idx]
            
            ax2.annotate(f'Peak Drivers: T{peak_driver_hour}\n({peak_driver_count:.0f} active)',
                        xy=(peak_driver_hour, peak_driver_count),
                        xytext=(peak_driver_hour - 3, peak_driver_count * 0.9),
                        arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.7),
                        fontsize=10, ha='right', color='steelblue',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)


# Integration with simulation
def integrate_queue_tracking_with_simulation():
    """
    Example of how to integrate the queue tracker with your simulation
    """
    
    class EnhancedSimulation:
        def __init__(self, graph, simulation_time=None):
            # ... existing initialization ...
            self.queue_tracker = QueueTimeTracker()
            # ... rest of existing code ...
        
        def record_queue_event(self, queue_time_minutes):
            """Call this whenever a car finishes queuing"""
            self.queue_tracker.record_queue_time(self.env.now, queue_time_minutes)
        
        def update_driver_count(self):
            """Call this periodically to track active drivers"""
            if self.queue_tracker.should_sample_driver_count(self.env.now):
                active_drivers = self.stats['cars_spawned'] - self.stats['cars_completed'] - self.stats['cars_stranded']
                self.queue_tracker.record_driver_count(self.env.now, active_drivers)
        
        def finalize_and_plot(self):
            """Call this at the end of simulation"""
            # Print summary
            self.queue_tracker.print_hourly_summary()
            
            # Create plot
            self.queue_tracker.plot_hourly_queue_times("EV Charging Queue Times Throughout the Day")
            
            return self.queue_tracker
    
    return EnhancedSimulation