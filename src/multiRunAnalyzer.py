import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from station_assignment import assign_charging_stations_to_nodes
from simPySimulation import SimpleEVSimulation

class MultiRunAnalyzer:
    """
    Runs multiple simulations with different seeds and combines results
    """
    def __init__(self, num_runs=4):
        self.num_runs = num_runs
        self.all_run_results = []
        self.combined_stats = {}
        
    def run_multiple_simulations(self, graph, total_cars=10000, base_seed=1):
        """
        Run simulation multiple times with different seeds
        """
        print(f"="*80)
        print(f"RUNNING {self.num_runs} SIMULATION RUNS")
        print(f"="*80)
        
        for run_num in range(1, self.num_runs + 1):
            # Set unique seed for this run
            current_seed = base_seed + (run_num - 1) * 1000
            random.seed(current_seed)
            np.random.seed(current_seed)
            
            print(f"\n{'='*60}")
            print(f"SIMULATION RUN {run_num}/{self.num_runs} (Seed: {current_seed})")
            print(f"{'='*60}")
            
            try:
                # Create and run simulation
                simulation = SimpleEVSimulation(graph, simulation_time=None)
                queue_tracker = simulation.run_simulation()
                
                # Collect results from this run
                run_results = {
                    'run_number': run_num,
                    'seed': current_seed,
                    'stats': simulation.stats.copy(),
                    'queue_tracker': queue_tracker,
                    'hourly_data': queue_tracker.calculate_hourly_statistics()
                }
                
                self.all_run_results.append(run_results)
                
                print(f"\n Run {run_num} completed successfully")
                print(f"  Cars completed: {simulation.stats['cars_completed']}")
                print(f"  Cars stranded: {simulation.stats['cars_stranded']}")
                print(f"  Total queue events: {len(simulation.stats['queue_times'])}")
                
            except Exception as e:
                print(f" Run {run_num} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"ALL RUNS COMPLETED: {len(self.all_run_results)}/{self.num_runs} successful")
        print(f"{'='*60}")
        
        if len(self.all_run_results) > 0:
            self._combine_results()
            return True
        else:
            print("No successful runs to analyze!")
            return False
    
    def _combine_results(self):
        """
        Combine results from all runs into aggregated statistics
        """
        print("\nCombining results from all runs...")
        
        # Initialize combined hourly data structures
        combined_hourly_queue_times = [[] for _ in range(24)]  # T1-T24
        combined_hourly_driver_counts = [[] for _ in range(24)]
        
        # Aggregate basic stats
        total_cars_spawned = 0
        total_cars_completed = 0
        total_cars_stranded = 0
        total_charging_events = 0
        total_queue_abandonments = 0
        all_queue_times = []
        
        # Process each run
        for run_result in self.all_run_results:
            stats = run_result['stats']
            queue_tracker = run_result['queue_tracker']
            
            # Aggregate basic statistics
            total_cars_spawned += stats['cars_spawned']
            total_cars_completed += stats['cars_completed'] 
            total_cars_stranded += stats['cars_stranded']
            total_charging_events += stats['total_charging_events']
            total_queue_abandonments += stats.get('queue_abandonments', 0)
            all_queue_times.extend(stats['queue_times'])
            
            # Aggregate hourly queue times
            for hour_idx in range(24):
                hourly_queue_times = queue_tracker.hourly_queue_times[hour_idx]
                hourly_driver_counts = queue_tracker.hourly_driver_counts[hour_idx]
                
                combined_hourly_queue_times[hour_idx].extend(hourly_queue_times)
                combined_hourly_driver_counts[hour_idx].extend(hourly_driver_counts)
        
        # Calculate combined statistics
        self.combined_stats = {
            'total_runs': len(self.all_run_results),
            'total_cars_spawned': total_cars_spawned,
            'total_cars_completed': total_cars_completed,
            'total_cars_stranded': total_cars_stranded,
            'total_charging_events': total_charging_events,
            'total_queue_abandonments': total_queue_abandonments,
            'all_queue_times': all_queue_times,
            'combined_hourly_queue_times': combined_hourly_queue_times,
            'combined_hourly_driver_counts': combined_hourly_driver_counts
        }
        
        print(f" Combined data from {len(self.all_run_results)} runs")
        print(f"  Total cars processed: {total_cars_spawned}")
        print(f"  Total queue events: {len(all_queue_times)}")
    
    def calculate_combined_hourly_statistics(self):
        """
        Calculate comprehensive statistics for combined data
        """
        hours = list(range(1, 25))  # T1 to T24
        
        # Queue time statistics
        avg_queue_times = []
        std_queue_times = []
        q1_queue_times = []
        q3_queue_times = []
        median_queue_times = []
        sample_counts = []
        
        # Driver count statistics  
        avg_driver_counts = []
        std_driver_counts = []
        
        for hour_idx in range(24):
            queue_times = self.combined_stats['combined_hourly_queue_times'][hour_idx]
            driver_counts = self.combined_stats['combined_hourly_driver_counts'][hour_idx]
            
            # Queue time statistics
            if queue_times:
                queue_array = np.array(queue_times)
                avg_queue = np.mean(queue_array)
                std_queue = np.std(queue_array)
                q1_queue = np.percentile(queue_array, 25)
                q3_queue = np.percentile(queue_array, 75)
                median_queue = np.median(queue_array)
                
                avg_queue_times.append(avg_queue)
                std_queue_times.append(std_queue)
                q1_queue_times.append(q1_queue)
                q3_queue_times.append(q3_queue)
                median_queue_times.append(median_queue)
                sample_counts.append(len(queue_times))
            else:
                avg_queue_times.append(0)
                std_queue_times.append(0)
                q1_queue_times.append(0)
                q3_queue_times.append(0)
                median_queue_times.append(0)
                sample_counts.append(0)
            
            # Driver count statistics
            if driver_counts:
                driver_array = np.array(driver_counts)
                avg_drivers = np.mean(driver_array)
                std_drivers = np.std(driver_array)
                
                avg_driver_counts.append(avg_drivers)
                std_driver_counts.append(std_drivers)
            else:
                avg_driver_counts.append(0)
                std_driver_counts.append(0)
        
        return {
            'hours': hours,
            'avg_queue_times': avg_queue_times,
            'std_queue_times': std_queue_times,
            'q1_queue_times': q1_queue_times,
            'q3_queue_times': q3_queue_times,
            'median_queue_times': median_queue_times,
            'sample_counts': sample_counts,
            'avg_driver_counts': avg_driver_counts,
            'std_driver_counts': std_driver_counts
        }
    
    def print_combined_summary(self):
        """
        Print comprehensive summary of combined results
        """
        stats = self.combined_stats
        hourly_stats = self.calculate_combined_hourly_statistics()
        
        print(f"\n{'='*100}")
        print(f"{'COMBINED RESULTS SUMMARY (5 RUNS)':^100}")
        print(f"{'='*100}")
        
        # Overall statistics
        print(f"OVERALL STATISTICS:")
        print(f"  Total runs: {stats['total_runs']}")
        print(f"  Total cars spawned: {stats['total_cars_spawned']}")
        print(f"  Total cars completed: {stats['total_cars_completed']}")
        print(f"  Total cars stranded: {stats['total_cars_stranded']}")
        print(f"  Total charging events: {stats['total_charging_events']}")
        print(f"  Total queue abandonments: {stats['total_queue_abandonments']}")
        
        # Success rates
        if stats['total_cars_spawned'] > 0:
            completion_rate = (stats['total_cars_completed'] / stats['total_cars_spawned']) * 100
            stranded_rate = (stats['total_cars_stranded'] / stats['total_cars_spawned']) * 100
            abandonment_rate = (stats['total_queue_abandonments'] / stats['total_cars_spawned']) * 100
            
            print(f"  Journey completion rate: {completion_rate:.2f}%")
            print(f"  Stranded rate: {stranded_rate:.2f}%")
            print(f"  Queue abandonment rate: {abandonment_rate:.2f}%")
        
        # Queue statistics
        if stats['all_queue_times']:
            queue_array = np.array(stats['all_queue_times'])
            print(f"\nQUEUE TIME STATISTICS:")
            print(f"  Total queue events: {len(queue_array)}")
            print(f"  Average queue time: {np.mean(queue_array):.2f} minutes")
            print(f"  Median queue time: {np.median(queue_array):.2f} minutes")
            print(f"  Standard deviation: {np.std(queue_array):.2f} minutes")
            print(f"  Min queue time: {np.min(queue_array):.2f} minutes")
            print(f"  Max queue time: {np.max(queue_array):.2f} minutes")
            print(f"  95th percentile: {np.percentile(queue_array, 95):.2f} minutes")
        
        # Hourly breakdown
        print(f"\nHOURLY QUEUE TIME BREAKDOWN:")
        print(f"{'Hour':<6} {'Time Range':<12} {'Samples':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Q1':<8} {'Q3':<8} {'Drivers':<8}")
        print(f"{'-'*80}")
        
        for i in range(24):
            hour_label = f"T{i+1}"
            time_range = f"{i:02d}:00-{(i+1)%24:02d}:00"
            samples = hourly_stats['sample_counts'][i]
            mean_q = hourly_stats['avg_queue_times'][i]
            median_q = hourly_stats['median_queue_times'][i]
            std_q = hourly_stats['std_queue_times'][i]
            q1_q = hourly_stats['q1_queue_times'][i]
            q3_q = hourly_stats['q3_queue_times'][i]
            drivers = hourly_stats['avg_driver_counts'][i]
            
            if samples > 0:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {mean_q:<8.1f} {median_q:<8.1f} {std_q:<8.1f} {q1_q:<8.1f} {q3_q:<8.1f} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {drivers:<8.0f}")
        
        print(f"{'='*100}")
        
        # Individual run comparison
        print(f"\nINDIVIDUAL RUN COMPARISON:")
        print(f"{'Run':<4} {'Seed':<8} {'Completed':<10} {'Stranded':<9} {'Queue Events':<12} {'Avg Queue':<10}")
        print(f"{'-'*60}")
        
        for run_result in self.all_run_results:
            run_num = run_result['run_number']
            seed = run_result['seed']
            completed = run_result['stats']['cars_completed']
            stranded = run_result['stats']['cars_stranded']
            queue_events = len(run_result['stats']['queue_times'])
            avg_queue = np.mean(run_result['stats']['queue_times']) if run_result['stats']['queue_times'] else 0
            
            print(f"{run_num:<4} {seed:<8} {completed:<10} {stranded:<9} {queue_events:<12} {avg_queue:<10.1f}")
        
        print(f"{'='*100}")
    
    def plot_combined_results(self):
        """
        Create queue time plot showing MEDIAN only (not mean)
        """
        hourly_stats = self.calculate_combined_hourly_statistics()

        # Create single figure for main plot only
        fig, ax1 = plt.subplots(figsize=(16, 10))
        ax2 = ax1.twinx()

        x = np.array(hourly_stats['hours'])
        # CHANGED: Use median instead of mean
        median_queue = np.array(hourly_stats['median_queue_times'])  # Use median instead of avg
        q1_queue = np.array(hourly_stats['q1_queue_times'])
        q3_queue = np.array(hourly_stats['q3_queue_times'])
        avg_drivers = np.array(hourly_stats['avg_driver_counts'])
        std_drivers = np.array(hourly_stats['std_driver_counts'])

        # Plot queue times - CHANGED: median instead of mean
        ax1.fill_between(x, q1_queue, q3_queue, alpha=0.3, color='lightcoral', 
                        label='Interquartile Range (Q1-Q3)')
        # CHANGED: Plot median as the main line
        ax1.plot(x, median_queue, color='darkred', linewidth=3, marker='o', 
                markersize=6, label='Median Queue Time')  # Changed from 'Mean' to 'Median'

        # Optional: Remove the redundant median line since it's now the main line
        # ax1.plot(x, median_queue, color='red', linewidth=2, linestyle='--', 
        #         alpha=0.8, label='Median Queue Time')

        # Plot driver counts with error bars (keep as is)
        ax2.errorbar(x, avg_drivers, yerr=std_drivers, color='steelblue', 
                    linewidth=2, marker='s', markersize=4, alpha=0.8, 
                    capsize=3, label='Active Drivers (±1σ)')

        # Customize main plot
        ax1.set_xlabel('Time of Day (T1–T24)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Queue Time (minutes)', fontsize=14, fontweight='bold', color='darkred')
        # CHANGED: Update title to reflect median
        ax1.set_title('Combined Results: Median Queue Times & Active Drivers (Multiple Runs)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.tick_params(axis='y', labelcolor='darkred')

        ax2.set_ylabel('Number of Active Drivers', fontsize=14, fontweight='bold', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')

        # FIXED: Set x-axis ticks properly
        ax1.set_xticks(x[::2])  # Every 2 hours to avoid crowding
        ax1.set_xticklabels([f'{i-1:02d}:00' for i in x[::2]], rotation=45)

        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig
    
    def export_results(self, filename_base="simulation_results"):
        """
        Export combined results to files
        """
        import json
        import pandas as pd
        
        # Export summary statistics
        summary_data = {
            'combined_stats': {k: v for k, v in self.combined_stats.items() 
                             if k not in ['combined_hourly_queue_times', 'combined_hourly_driver_counts']},
            'hourly_statistics': self.calculate_combined_hourly_statistics(),
            'run_details': [
                {
                    'run_number': r['run_number'],
                    'seed': r['seed'],
                    'cars_completed': r['stats']['cars_completed'],
                    'cars_stranded': r['stats']['cars_stranded'],
                    'total_queue_events': len(r['stats']['queue_times']),
                    'avg_queue_time': np.mean(r['stats']['queue_times']) if r['stats']['queue_times'] else 0
                }
                for r in self.all_run_results
            ]
        }
        
        # Save as JSON
        with open(f"{filename_base}_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        hourly_stats = self.calculate_combined_hourly_statistics()
        df = pd.DataFrame({
            'Hour': [f"T{i}" for i in hourly_stats['hours']],
            'Hour_Number': hourly_stats['hours'],  
            'Time_Range': [f"{i-1:02d}:00-{i:02d}:00" for i in hourly_stats['hours']],
            'Sample_Count': hourly_stats['sample_counts'],
            'Mean_Queue_Time': hourly_stats['avg_queue_times'],
            'Median_Queue_Time': hourly_stats['median_queue_times'],  
            'Std_Queue_Time': hourly_stats['std_queue_times'],
            'Q1_Queue_Time': hourly_stats['q1_queue_times'],           
            'Q3_Queue_Time': hourly_stats['q3_queue_times'],          
            'Mean_Active_Drivers': hourly_stats['avg_driver_counts'], 
            'Std_Active_Drivers': hourly_stats['std_driver_counts']  
        })
        
        df.to_csv(f"{filename_base}_hourly.csv", index=False)
        
        print(f" Results exported:")
        print(f"  Summary: {filename_base}_summary.json")
        print(f"  Hourly data: {filename_base}_hourly.csv")

        combined_soc_data = []
    
        for run_result in self.all_run_results:
            queue_tracker = run_result['queue_tracker']
            run_number = run_result['run_number']
        
            # Extract SoC data from this run
            for hour_idx in range(24):
                hour_label = f"T{hour_idx + 1}"
                soc_data = queue_tracker.hourly_soc_data[hour_idx]

                for entry in soc_data:
                    combined_soc_data.append({
                        'Run': run_number,
                        'Hour': hour_label,
                        'Hour_Number': hour_idx + 1,
                        'Simulation_Time': entry['time'],
                        'Average_SoC': entry['avg_soc'],
                        'Active_Drivers': entry['active_drivers'],
                        'Total_SoC': entry['total_soc']
                    })

        # Save combined SoC data
        soc_df = pd.DataFrame(combined_soc_data)
        soc_df.to_csv(f"{filename_base}_soc_data.csv", index=False)
        print(f"Combined SoC data exported: {filename_base}_soc_data.csv")

        # Calculate hourly SoC averages across all runs
        hourly_soc_summary = []
        for hour_idx in range(24):
            hour_data = soc_df[soc_df['Hour_Number'] == hour_idx + 1]
            if not hour_data.empty:
                hourly_soc_summary.append({
                    'Hour': f"T{hour_idx + 1}",
                    'Hour_Number': hour_idx + 1,
                    'Median_SoC': hour_data['Average_SoC'].median(),
                    'Mean_SoC': hour_data['Average_SoC'].mean(),
                    'Std_SoC': hour_data['Average_SoC'].std(),
                    'Min_SoC': hour_data['Average_SoC'].min(),
                    'Max_SoC': hour_data['Average_SoC'].max(),
                    'Sample_Count': len(hour_data)
                })
            else:
                hourly_soc_summary.append({
                    'Hour': f"T{hour_idx + 1}",
                    'Hour_Number': hour_idx + 1,
                    'Mean_SoC': 0,
                    'Median_SoC': 0,
                    'Std_SoC': 0,
                    'Min_SoC': 0,
                    'Max_SoC': 0,
                    'Sample_Count': 0
                    })

        # Save hourly SoC summary
        soc_summary_df = pd.DataFrame(hourly_soc_summary)
        soc_summary_df.to_csv(f"{filename_base}_soc_hourly_summary.csv", index=False)
        print(f"Hourly SoC summary exported: {filename_base}_soc_hourly_summary.csv")

        print(f"\nResults exported:")
        print(f"  Summary: {filename_base}_summary.json")
        print(f"  Hourly data: {filename_base}_hourly.csv")
        print(f"  SoC data: {filename_base}_soc_data.csv")
        print(f"  SoC summary: {filename_base}_soc_hourly_summary.csv")


def run_multi_simulation_analysis():
    """
    Main function to run multiple simulations and analyze results
    """
    print("Loading graph and charging stations...")
    
    # Load data
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/TargetedWeightedFailures.json"
    
    try:
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is None:
            print("Failed to load graph!")
            return
        
        print(f"Graph loaded successfully with {len(graph.nodes)} nodes")
        
        # Create analyzer and run multiple simulations
        analyzer = MultiRunAnalyzer(num_runs=4)
        success = analyzer.run_multiple_simulations(graph, total_cars=10000, base_seed=1)
        
        if success:
            # Print combined summary
            analyzer.print_combined_summary()
            
            # Create combined plots
            analyzer.plot_combined_results()
            
            # Export results
            analyzer.export_results("ev_simulation_5runs")
            
            print("\n" + "="*80)
            print("MULTI-RUN ANALYSIS COMPLETE")
            print("="*80)
            print(" 5 simulation runs completed")
            print(" Combined statistics calculated")
            print(" Comprehensive plots generated")
            print(" Results exported to files")
            
        else:
            print("Failed to complete multi-run analysis!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_multi_simulation_analysis()