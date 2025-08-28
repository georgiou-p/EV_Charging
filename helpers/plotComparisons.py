import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme for matplotlib
plt.style.use('dark_background')

def create_strategy_comparison_graphs():
    """
    Creates 6 graphs comparing EV charging strategies in dark mode:
    - 3 SoC graphs (all working, 10% random failures, 10% targeted failures)
    - 3 Queue time graphs (all working, 10% random failures, 10% targeted failures)
    """
    
    # Define file mappings
    # x = {1: all working, 2: 10% random, 3: 10% targeted}
    # y = {1: forward planning, 2: nearest first, 3: performance only}
    
    soc_files = {
        'all_working': {
            'forward_planning': 'SoC1_1.csv',
            'nearest_first': 'SoC1_2.csv', 
            'performance_only': 'SoC1_3.csv'
        },
        'random_failures': {
            'forward_planning': 'SoC2_1.csv',
            'nearest_first': 'SoC2_2.csv',
            'performance_only': 'SoC2_3.csv'
        },
        'targeted_failures': {
            'forward_planning': 'SoC3_1.csv',
            'nearest_first': 'SoC3_2.csv',
            'performance_only': 'SoC3_3.csv'
        }
    }
    
    queue_files = {
        'all_working': {
            'forward_planning': 'Queue1_1.csv',
            'nearest_first': 'Queue1_2.csv',
            'performance_only': 'Queue1_3.csv'
        },
        'random_failures': {
            'forward_planning': 'Queue2_1.csv',
            'nearest_first': 'Queue2_2.csv',
            'performance_only': 'Queue2_3.csv'
        },
        'targeted_failures': {
            'forward_planning': 'Queue3_1.csv',
            'nearest_first': 'Queue3_2.csv',
            'performance_only': 'Queue3_3.csv'
        }
    }
    
    # Dark mode compatible strategy colors - brighter colors for dark backgrounds
    strategies = {
        'forward_planning': ('Strategic Forward Planning', '#00D4FF'),  # Bright cyan
        'nearest_first': ('Nearest First', '#FF6B6B'),  # Bright red
        'performance_only': ('Performance Only', '#ADFF2F')  # GreenYellow
    }
    
    # Create combined SoC/Anxiety graph
    plt.figure(figsize=(14, 8), facecolor='#1a1a1a')  # Dark figure background
    
    # Strategy labels and colors for dark mode
    strategy_colors = {
        'forward_planning': '#00D4FF',  # Bright cyan
        'nearest_first': '#FF6B6B',    # Bright red
        'performance_only': '#ADFF2F'  # GreenYellow
    }
    
    strategy_labels = {
        'forward_planning': 'Strategic Forward Planning',
        'nearest_first': 'Nearest First',
        'performance_only': 'Performance Only'
    }
    
    # Line styles for different scenarios with larger, more distinguishable markers
    scenario_styles = {
        'all_working': {'linestyle': '-', 'marker': 'o', 'alpha': 1.0, 'linewidth': 2.5, 'markersize': 7},
        'random_failures': {'linestyle': '--', 'marker': 's', 'alpha': 0.9, 'linewidth': 2.0, 'markersize': 6},
        'targeted_failures': {'linestyle': '-.', 'marker': '^', 'alpha': 0.95, 'linewidth': 2.0, 'markersize': 8}
    }
    
    scenario_names = {
        'all_working': 'All Working',
        'random_failures': '10% Random Failures',
        'targeted_failures': '10% Targeted Failures'
    }
    
    # Plot all combinations
    for scenario in ['all_working', 'random_failures', 'targeted_failures']:
        for strategy_key in ['forward_planning', 'nearest_first', 'performance_only']:
            try:
                # Load data
                df = pd.read_csv(soc_files[scenario][strategy_key])
                
                # Calculate anxiety level (1 - Mean SoC)
                anxiety_level = 1 - df['Mean_SoC']
                
                # Create combined label
                label = f"{strategy_labels[strategy_key]} - {scenario_names[scenario]}"
                
                # Plot anxiety level
                plt.plot(df['Hour_Number'], anxiety_level,
                        color=strategy_colors[strategy_key],
                        linestyle=scenario_styles[scenario]['linestyle'],
                        marker=scenario_styles[scenario]['marker'],
                        linewidth=scenario_styles[scenario]['linewidth'],
                        alpha=scenario_styles[scenario]['alpha'],
                        markersize=scenario_styles[scenario]['markersize'],
                        label=label,
                        markevery=3,  # Show markers every 3 points to reduce clutter
                        markeredgecolor='white',  # White edge for dark mode
                        markeredgewidth=0.8)  # Slightly thicker for visibility
                
            except FileNotFoundError:
                print(f"Warning: File {soc_files[scenario][strategy_key]} not found")
                continue
    
    # Dark mode styling
    ax = plt.gca()
    ax.set_facecolor('#2d2d2d')  # Dark grey plot background
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')
    
    plt.xlabel('Hour of Day', fontsize=16, fontweight='bold', color='white')
    plt.ylabel('Anxiety Level (1 - Mean SoC)', fontsize=16, fontweight='bold', color='white')
    plt.title('EV Driver Anxiety Levels Across All Scenarios', fontsize=18, fontweight='bold', pad=20, color='white')
    
    # Grid with dark mode styling
    plt.grid(True, alpha=0.3, color='gray')
    plt.xlim(1, 24)
    plt.ylim(0.2, 0.6)  # Start from 0.2 for more compact view
    
    # Set x-axis ticks to show every 2 hours in HH:MM format
    hours = range(1, 25, 2)  # Every 2 hours: 1, 3, 5, 7, ...
    hour_labels = [f"{h-1:02d}:00" for h in hours]
    plt.xticks(hours, hour_labels, fontsize=14, color='white')
    plt.yticks(fontsize=14, color='white')
    
    # Store handles and labels for separate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Adjust layout for cleaner appearance
    plt.tight_layout()
    
    # Save plot with higher DPI for better quality
    filename = "anxiety_combined_scenarios_dark.png"
    plt.savefig(filename, dpi=1000, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")
    
    # Store legend data for later use
    global legend_handles, legend_labels
    legend_handles = handles
    legend_labels = labels
    
    # Create individual SoC graphs (keep originals as backup)
    soc_scenarios = [
        ('all_working', 'Anxiety Level - All Working Stations'),
        ('random_failures', 'Anxiety Level - 10% Random Failures'),
        ('targeted_failures', 'Anxiety Level - 10% Targeted Failures')
    ]
    
    for scenario, title in soc_scenarios:
        plt.figure(figsize=(12, 8), facecolor='#1a1a1a')
        
        for strategy_key, (strategy_label, color) in strategies.items():
            try:
                # Load data
                df = pd.read_csv(soc_files[scenario][strategy_key])
                
                # Calculate anxiety level (1 - Mean SoC)
                anxiety_level = 1 - df['Mean_SoC']
                
                # Plot anxiety level
                plt.plot(df['Hour_Number'], anxiety_level, 
                        color=color, linewidth=2.5, label=strategy_label,
                        marker='o', markersize=5, markeredgecolor='white', 
                        markeredgewidth=0.8, alpha=0.9)
                
            except FileNotFoundError:
                print(f"Warning: File {soc_files[scenario][strategy_key]} not found")
                continue
        
        # Dark mode styling for individual graphs
        ax = plt.gca()
        ax.set_facecolor('#2d2d2d')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        
        plt.xlabel('Hour of Day', fontsize=16, fontweight='bold', color='white')
        plt.ylabel('Anxiety Level (1 - Mean SoC)', fontsize=16, fontweight='bold', color='white')
        plt.title(title, fontsize=18, fontweight='bold', color='white')
        plt.grid(True, alpha=0.3, color='gray')
        plt.xlim(1, 24)
        plt.ylim(0, 1)
        
        # Set x-axis ticks to show every 2 hours in HH:MM format
        hours = range(1, 25, 2)  # Every 2 hours: 1, 3, 5, 7, ...
        hour_labels = [f"{h-1:02d}:00" for h in hours]
        plt.xticks(hours, hour_labels, fontsize=14, color='white')
        plt.yticks(fontsize=14, color='white')
        
        # Save plot with higher DPI
        filename = f"soc_{scenario}_dark.png"
        plt.savefig(filename, dpi=1000, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        plt.close()
        print(f"Saved: {filename}")

def create_legend_png():
    """
    Create two separate visual legend PNGs matching the dark mode style
    """
    print("=== CREATING DARK MODE LEGEND PNGs ===")
    
    # Strategy definitions for dark mode
    strategy_colors = {
        'forward_planning': '#00D4FF', 
        'nearest_first': '#FF6B6B', 
        'performance_only': '#ADFF2F'
    }
    strategy_labels = {
        'forward_planning': 'Strategic Forward Planning', 
        'nearest_first': 'Nearest First', 
        'performance_only': 'Performance Only'
    }
    
    # Create ANXIETY LEGEND
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a1a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')
    
    # Draw legend box border in white
    legend_box = plt.Rectangle((0.5, 2), 9, 6, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(legend_box)
    
    y_start = 7.5
    line_spacing = 0.6
    
    # Anxiety legend entries
    scenarios = [
        ('all_working', '-', 'o', 'All Working'),
        ('random_failures', '--', 's', '10% Random Failures'),
        ('targeted_failures', '-.', '^', '10% Targeted Failures')
    ]
    
    for strategy_key, strategy_name in [('forward_planning', 'Strategic Forward Planning'), 
                                       ('nearest_first', 'Nearest First'), 
                                       ('performance_only', 'Performance Only')]:
        color = strategy_colors[strategy_key]
        
        for scenario_key, linestyle, marker, scenario_name in scenarios:
            # Draw line sample
            ax.plot([1, 2.5], [y_start, y_start], color=color, linestyle=linestyle, 
                   linewidth=3, marker=marker, markersize=8, markeredgecolor='white', 
                   markeredgewidth=0.8)
            
            # Add text label in white
            ax.text(3, y_start, f'{strategy_name} - {scenario_name}', 
                   va='center', fontsize=12, fontweight='bold', color='white')
            
            y_start -= line_spacing
    
    plt.tight_layout()
    plt.savefig('legend_anxiety_dark.png', dpi=1000, bbox_inches='tight', 
               facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("SAVED: legend_anxiety_dark.png")
    
    # Create QUEUE LEGEND
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')
    
    # Draw legend box border in white
    legend_box = plt.Rectangle((0.5, 1), 9, 6, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(legend_box)
    
    y_start = 6
    line_spacing = 0.8
    
    # Queue legend entries
    for strategy_key, strategy_name in [('forward_planning', 'Strategic Forward Planning'), 
                                       ('nearest_first', 'Nearest First'), 
                                       ('performance_only', 'Performance Only')]:
        color = strategy_colors[strategy_key]
        
        # Median line
        ax.plot([1, 2.5], [y_start, y_start], color=color, linestyle='-', 
               linewidth=3, marker='o', markersize=8, markeredgecolor='white', 
               markeredgewidth=0.8)
        ax.text(3, y_start, f'{strategy_name} (Median)', 
               va='center', fontsize=12, fontweight='bold', color='white')
        y_start -= line_spacing/2
        
        # Q3 line
        ax.plot([1, 2.5], [y_start, y_start], color=color, linestyle='--', 
               linewidth=3, marker='^', markersize=8, markeredgecolor='white', 
               markeredgewidth=0.8, alpha=0.9)
        ax.text(3, y_start, f'{strategy_name} (Q3)', 
               va='center', fontsize=12, fontweight='bold', color='white')
        y_start -= line_spacing
    
    plt.tight_layout()
    plt.savefig('legend_queue_dark.png', dpi=1000, bbox_inches='tight', 
               facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print("SAVED: legend_queue_dark.png")
    print("=== LEGEND CREATION COMPLETE ===")

def create_queue_time_graphs():
    """Create queue time graphs separately"""
    
    # Define queue files mapping
    queue_files = {
        'all_working': {
            'forward_planning': 'Queue1_1.csv',
            'nearest_first': 'Queue1_2.csv',
            'performance_only': 'Queue1_3.csv'
        },
        'random_failures': {
            'forward_planning': 'Queue2_1.csv',
            'nearest_first': 'Queue2_2.csv',
            'performance_only': 'Queue2_3.csv'
        },
        'targeted_failures': {
            'forward_planning': 'Queue3_1.csv',
            'nearest_first': 'Queue3_2.csv',
            'performance_only': 'Queue3_3.csv'
        }
    }
    
    # Strategy colors for queue graphs
    strategies = {
        'forward_planning': ('Strategic Forward Planning', '#00D4FF'),  # Bright cyan
        'nearest_first': ('Nearest First', '#FF6B6B'),  # Bright red
        'performance_only': ('Performance Only', '#ADFF2F')  # GreenYellow
    }
    
    # Create Queue time graphs
    queue_scenarios = [
        ('all_working', 'Queue Times - All Working Stations'),
        ('random_failures', 'Queue Times - 10% Random Failures'),
        ('targeted_failures', 'Queue Times - 10% Targeted Failures')
    ]
    
    for scenario, title in queue_scenarios:
        plt.figure(figsize=(16, 8), facecolor='#1a1a1a')  # Dark background
        
        # Check if this is the targeted failures scenario for special handling
        is_targeted_failures = scenario == 'targeted_failures'
        
        for strategy_key, (strategy_label, color) in strategies.items():
            try:
                # Load data
                df = pd.read_csv(queue_files[scenario][strategy_key])
                
                # Plot median queue time (solid line)
                plt.plot(df['Hour_Number'], df['Median_Queue_Time'], 
                        color=color, linewidth=2.5, label=f'{strategy_label} (Median)',
                        marker='o', markersize=6, markeredgecolor='white', 
                        markeredgewidth=0.8, alpha=0.9)
                
                # Plot Q3 (75th percentile) queue time (dashed line)
                plt.plot(df['Hour_Number'], df['Q3_Queue_Time'], 
                        color=color, linewidth=2.0, linestyle='--', 
                        label=f'{strategy_label} (Q3)', alpha=0.85,
                        marker='^', markersize=5, markeredgecolor='white', 
                        markeredgewidth=0.8)
                
            except FileNotFoundError:
                print(f"Warning: File {queue_files[scenario][strategy_key]} not found")
                continue
        
        # Dark mode styling
        ax = plt.gca()
        ax.set_facecolor('#2d2d2d')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        
        plt.xlabel('Hour of Day', fontsize=16, fontweight='bold', color='white')
        plt.ylabel('Queue Time (minutes)', fontsize=16, fontweight='bold', color='white')
        plt.title(title, fontsize=18, fontweight='bold', color='white')
        plt.grid(True, alpha=0.3, color='gray')
        plt.xlim(1, 24)
        
        # Set y-axis: 50 for most graphs, 90 ONLY for targeted failures
        if is_targeted_failures:
            plt.ylim(0, 90)  # Only targeted failures gets 90
        else:
            plt.ylim(0, 50)  # All other queue graphs stay at 50
        
        # Set x-axis ticks to show every 2 hours in HH:MM format
        hours = range(1, 25, 2)  # Every 2 hours
        hour_labels = [f"{h-1:02d}:00" for h in hours]
        plt.xticks(hours, hour_labels, fontsize=14, color='white')
        plt.yticks(fontsize=14, color='white')
        
        # Save plot with higher DPI
        filename = f"queue_{scenario}_dark.png"
        plt.savefig(filename, dpi=1000, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        plt.close()
        print(f"Saved: {filename}")

def create_all_graphs():
    """Main function to create all 6 comparison graphs in dark mode"""
    print("Creating EV charging strategy comparison graphs in DARK MODE...")
    print("=" * 60)
    
    # Create the graphs first
    create_strategy_comparison_graphs()
    
    # Create legend PNG
    create_legend_png()
    
    # Create queue time graphs
    create_queue_time_graphs()
    
    print("=" * 60)
    print("All DARK MODE graphs created successfully!")
    print("\nGenerated files:")
    print("SoC Graphs:")
    print("  - soc_all_working_dark.png")
    print("  - soc_random_failures_dark.png") 
    print("  - soc_targeted_failures_dark.png")
    print("  - anxiety_combined_scenarios_dark.png")
    print("\nQueue Time Graphs:")
    print("  - queue_all_working_dark.png")
    print("  - queue_random_failures_dark.png")
    print("  - queue_targeted_failures_dark.png")
    print("\nLegends:")
    print("  - legend_anxiety_dark.png")
    print("  - legend_queue_dark.png")

if __name__ == "__main__":
    create_all_graphs()