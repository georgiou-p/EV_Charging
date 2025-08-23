import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_queue_times(queue_csv_path):
    """
    Plot queue times in a separate window
    """
    
    # Load the data
    queue_df = pd.read_csv(queue_csv_path)
    
    # Create queue plot window
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    
    # Extract queue data
    hours = queue_df['Hour_Number'].values
    median_queue = queue_df['Median_Queue_Time'].values
    q1_queue = queue_df['Q1_Queue_Time'].values
    q3_queue = queue_df['Q3_Queue_Time'].values
    avg_drivers = queue_df['Mean_Active_Drivers'].values
    std_drivers = queue_df['Std_Active_Drivers'].values
    
    # Create secondary y-axis for driver count
    ax1_secondary = ax1.twinx()
    
    # Plot queue times
    ax1.fill_between(hours, q1_queue, q3_queue, alpha=0.3, color='lightcoral', 
                    label='Interquartile Range (Q1-Q3)')
    ax1.plot(hours, median_queue, color='darkred', linewidth=3, marker='o', 
            markersize=6, label='Median Queue Time')
    
    # Plot driver counts with error bars
    ax1_secondary.errorbar(hours, avg_drivers, yerr=std_drivers, color='steelblue', 
                          linewidth=2, marker='s', markersize=4, alpha=0.8, 
                          capsize=3, label='Active Drivers (±1σ)')
    
    # Customize queue plot
    ax1.set_xlabel('Time of Day', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Queue Time (minutes)', fontsize=14, fontweight='bold', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    
    ax1_secondary.set_ylabel('Number of Active Drivers', fontsize=14, fontweight='bold', color='steelblue')
    ax1_secondary.tick_params(axis='y', labelcolor='steelblue')
    
    # Set x-axis to show actual hours (0-23)
    ax1.set_xticks(hours[::2])  # Every 2 hours
    ax1.set_xticklabels([f'{int(h-1):02d}:00' for h in hours[::2]], rotation=45)
    
    # Set reasonable limits for driver count
    if np.max(avg_drivers) > 0:
        ax1_secondary.set_ylim(0, np.max(avg_drivers + std_drivers) * 1.1)
    
    # Add grid and legends
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_secondary.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    # Set window title
    fig1.canvas.manager.set_window_title('Queue Times Throughout the Day')
    
    plt.tight_layout()
    plt.show(block=False)  # Don't block so we can show second window
    
    return fig1, ax1

def plot_anxiety_levels_with_drivers(soc_csv_path, queue_csv_path):
    """
    Plot anxiety levels (1 - SoC) with active drivers on secondary axis
    """
    
    # Load both datasets
    soc_df = pd.read_csv(soc_csv_path)
    queue_df = pd.read_csv(queue_csv_path)
    
    # Create anxiety plot window
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    
    # Create secondary y-axis for driver count
    ax2_secondary = ax2.twinx()
    
    # Extract SoC data
    soc_hours = soc_df['Hour_Number'].values
    sample_counts = soc_df['Sample_Count'].values
    
    # Extract driver data from queue CSV
    driver_hours = queue_df['Hour_Number'].values
    avg_drivers = queue_df['Mean_Active_Drivers'].values
    std_drivers = queue_df['Std_Active_Drivers'].values
    
    # Check if we have quartile data (new format) or just mean/std (old format)
    if 'Q1_SoC' in soc_df.columns and 'Q3_SoC' in soc_df.columns:
        print("Using quartile-based confidence bands for SoC")
        # Use quartiles for confidence bands (more robust)
        mean_soc = soc_df['Mean_SoC'].values
        q1_soc = soc_df['Q1_SoC'].values
        q3_soc = soc_df['Q3_SoC'].values
        
        # Calculate anxiety levels
        anxiety_level = 1 - mean_soc
        anxiety_q1 = 1 - q3_soc  # Note: inverted because 1-SoC
        anxiety_q3 = 1 - q1_soc  # Note: inverted because 1-SoC
        
        # Plot with quartile bands
        ax2.fill_between(soc_hours, anxiety_q1, anxiety_q3, alpha=0.3, color='orange', 
                        label='Interquartile Range (Q1-Q3)')
        
    else:
        print("Using standard deviation confidence bands for SoC")
        # Fall back to mean ± std
        mean_soc = soc_df['Mean_SoC'].values
        std_soc = soc_df['Std_SoC'].values
        
        # Calculate anxiety level: 1 - SoC
        anxiety_level = 1 - mean_soc
        
        # Use standard error for better confidence bands
        std_error = std_soc / np.sqrt(np.maximum(sample_counts, 1))
        upper_band = np.clip(anxiety_level + std_error, 0, 1)
        lower_band = np.clip(anxiety_level - std_error, 0, 1)
        
        # Plot with standard error bands
        ax2.fill_between(soc_hours, lower_band, upper_band, alpha=0.3, color='orange', 
                        label='±1 Standard Error')
    
    # Plot main anxiety line
    ax2.plot(soc_hours, anxiety_level, color='darkorange', linewidth=3, marker='o', 
            markersize=6, label='Anxiety Level (1 - Mean SoC)')
    
    # Plot driver counts with error bars on secondary axis
    ax2_secondary.errorbar(driver_hours, avg_drivers, yerr=std_drivers, color='steelblue', 
                          linewidth=2, marker='s', markersize=4, alpha=0.8, 
                          capsize=3, label='Active Drivers (±1σ)')
    
    # Customize anxiety plot (primary axis)
    ax2.set_xlabel('Time of Day', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Anxiety Level (1 - SoC)', fontsize=14, fontweight='bold', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # Customize driver count axis (secondary axis)
    ax2_secondary.set_ylabel('Number of Active Drivers', fontsize=14, fontweight='bold', color='steelblue')
    ax2_secondary.tick_params(axis='y', labelcolor='steelblue')
    
    # Set x-axis to show actual hours (0-23)
    ax2.set_xticks(soc_hours[::2])  # Every 2 hours
    ax2.set_xticklabels([f'{int(h-1):02d}:00' for h in soc_hours[::2]], rotation=45)
    
    # Set y-axis limits
    ax2.set_ylim(0, 1)  # Anxiety level from 0 to 1
    
    # Set reasonable limits for driver count
    if np.max(avg_drivers) > 0:
        ax2_secondary.set_ylim(0, np.max(avg_drivers + std_drivers) * 1.1)
    
    # Add horizontal reference lines for anxiety
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='50% anxiety')
    ax2.axhline(y=0.25, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax2.axhline(y=0.75, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_secondary.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    # Set window title
    fig2.canvas.manager.set_window_title('Driver Anxiety Levels & Active Drivers Throughout the Day')
    
    plt.tight_layout()
    plt.show(block=False)  # Don't block so both windows show
    
    return fig2, ax2

def plot_both_separate_windows(queue_csv_path="Queue1_1.csv", 
                              soc_csv_path="SoC1_1.csv"):
    """
    Plot both graphs in separate windows - anxiety graph now includes active drivers
    """
    
    print("Creating separate windows for queue times and anxiety levels with drivers...")
    
    # Create first window - Queue Times
    print("Opening Queue Times window...")
    fig1, ax1 = plot_queue_times(queue_csv_path)
    
    # Create second window - Anxiety Levels with Active Drivers
    print("Opening Anxiety Levels with Active Drivers window...")
    fig2, ax2 = plot_anxiety_levels_with_drivers(soc_csv_path, queue_csv_path)
    
    # Keep both windows open
    print("Both windows are now open. Close windows manually when done.")
    plt.show()  # This will block and keep both windows open
    
    return fig1, fig2


if __name__ == "__main__":
   
    
    # Create both plots in separate windows
    fig1, fig2 = plot_both_separate_windows()
    
    print("Plots generated successfully in separate windows!")