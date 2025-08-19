import matplotlib.pyplot as plt
import networkx as nx
import geopandas
import numpy as np

def visualize_stations_on_map(graph, map_regions_path=None, figsize=(15, 10)):
    """
    Visualize charging stations assigned to graph nodes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load basemap if available
    if map_regions_path:
        try:
            map_regions = geopandas.read_file(map_regions_path)
            map_regions.plot(ax=ax, linewidth=0.5, edgecolor="grey", 
                           facecolor="lightblue", alpha=0.3)
        except:
            print("Could not load basemap")
    
    # Get positions
    positions = {node: graph.nodes[node]['position'] for node in graph.nodes}
    
    # Separate nodes with and without stations
    nodes_with_stations = []
    nodes_without_stations = []
    station_counts = []
    
    for node in graph.nodes:
        num_stations = len(graph.nodes[node]['charging_stations'])
        if num_stations > 0:
            nodes_with_stations.append(node)
            station_counts.append(num_stations)
        else:
            nodes_without_stations.append(node)
    
    # Draw edges
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color='lightgray', 
                          alpha=0.3, width=0.5)
    
    # Draw nodes without stations
    if nodes_without_stations:
        nx.draw_networkx_nodes(graph, positions, nodelist=nodes_without_stations,
                              node_color='lightgray', node_size=20, alpha=0.5, ax=ax)
    
    # Draw nodes with stations
    if nodes_with_stations:
        max_stations = max(station_counts)
        node_sizes = [50 + (count / max_stations) * 200 for count in station_counts]
        
        scatter = nx.draw_networkx_nodes(graph, positions, nodelist=nodes_with_stations,
                                        node_color=station_counts, node_size=node_sizes,
                                        cmap='Reds', alpha=0.8, ax=ax)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Number of Charging Stations', rotation=270, labelpad=20)
    
    # Set appearance
    ax.set_xlim(-8, 2)
    ax.set_ylim(50, 61)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('UK Charging Stations Assignment', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def visualize_complete_graph(graph, map_regions_path=None, figsize=(20, 15)):
    """
    Complete visualization showing both charging stations AND edge weights in one graph
    With interactive hover to show node IDs
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load basemap if available
    if map_regions_path:
        try:
            map_regions = geopandas.read_file(map_regions_path)
            map_regions.plot(ax=ax, linewidth=0.5, edgecolor="grey", 
                           facecolor="lightblue", alpha=0.3)
            print("Loaded UK map background successfully")
        except Exception as e:
            print(f"Could not load basemap: {e}")
    
    # Get positions
    positions = {node: graph.nodes[node]['position'] for node in graph.nodes}
    
    # Get ALL edge weights
    all_edges_with_weights = []
    for edge in graph.edges(data=True):
        if 'weight' in edge[2]:
            weight = edge[2]['weight']
            all_edges_with_weights.append((edge[0], edge[1], weight))
    
    if not all_edges_with_weights:
        print("No weighted edges found!")
        return fig, ax
    
    print(f"Total edges to display: {len(all_edges_with_weights)}")
    
    # Get weight range for color coding edges
    all_weights = [w[2] for w in all_edges_with_weights]
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    weight_range = max_weight - min_weight
    
    # Draw all edges with color coding by weight
    for node1, node2, weight in all_edges_with_weights:
        normalized_weight = (weight - min_weight) / weight_range if weight_range > 0 else 0
        edge_color = plt.cm.RdYlBu_r(normalized_weight)
        edge_width = 0.5 + (1 - normalized_weight) * 2  # Thicker for shorter distances
        
        nx.draw_networkx_edges(graph, positions, edgelist=[(node1, node2)], 
                             edge_color=[edge_color], width=edge_width, 
                             alpha=0.6, ax=ax)
    
    # Separate nodes with and without charging stations
    nodes_with_stations = []
    nodes_without_stations = []
    station_counts = []
    
    for node in graph.nodes:
        num_stations = len(graph.nodes[node]['charging_stations'])
        if num_stations > 0:
            nodes_with_stations.append(node)
            station_counts.append(num_stations)
        else:
            nodes_without_stations.append(node)
    
    # Store node collections for hover functionality
    node_artists = {}
    
    # Draw nodes WITHOUT charging stations (smaller, blue)
    if nodes_without_stations:
        scatter_no_stations = ax.scatter([positions[node][0] for node in nodes_without_stations],
                                        [positions[node][1] for node in nodes_without_stations],
                                        c='lightblue', s=20, alpha=0.7, 
                                        edgecolors='navy', linewidths=0.5, zorder=5)
        # Store for hover detection
        for i, node in enumerate(nodes_without_stations):
            node_artists[node] = {'artist': scatter_no_stations, 'index': i, 'has_stations': False}
    
    # Draw nodes WITH charging stations (larger, red, sized by station count)
    if nodes_with_stations:
        max_stations = max(station_counts) if station_counts else 1
        # Scale node sizes based on number of stations
        node_sizes = [50 + (count / max_stations) * 200 for count in station_counts]
        
        # Use red color scale for charging stations
        scatter_stations = ax.scatter([positions[node][0] for node in nodes_with_stations],
                                     [positions[node][1] for node in nodes_with_stations],
                                     c=station_counts, s=node_sizes, cmap='Reds', alpha=0.8, 
                                     edgecolors='darkred', linewidths=1, zorder=6)
        
        # Store for hover detection
        for i, node in enumerate(nodes_with_stations):
            node_artists[node] = {'artist': scatter_stations, 'index': i, 'has_stations': True, 'station_count': station_counts[i]}
        
        # Add colorbar for charging stations
        cbar_stations = plt.colorbar(scatter_stations, ax=ax, shrink=0.6, pad=0.12)
        cbar_stations.set_label('Number of Charging Stations', rotation=270, labelpad=20)
    
    # LABEL ALL POSSIBLE EDGES with weights
    print("Attempting to label ALL edges where space permits...")
    
    successful_labels = 0
    total_attempts = 0
    
    # Sort edges by weight for better label priority
    sorted_edges = sorted(all_edges_with_weights, key=lambda x: x[2])
    
    label_positions = {}
    min_label_distance = 0.06  # Minimum distance between labels
    
    for node1, node2, weight in sorted_edges:
        total_attempts += 1
        pos1 = np.array(positions[node1])
        pos2 = np.array(positions[node2])
        
        edge_vector = pos2 - pos1
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length == 0:
            continue
        
        # Try multiple positions for this edge
        found_position = False
        
        # Try different positions along the edge
        for t in [0.5, 0.4, 0.6, 0.3, 0.7]:  # Start with center
            edge_point = pos1 + t * edge_vector
            
            # Try perpendicular offsets
            perp_vector = np.array([-edge_vector[1], edge_vector[0]]) / edge_length
            
            for offset in [0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12]:
                label_pos = edge_point + offset * perp_vector
                
                # Check if this position conflicts with existing labels
                too_close = False
                for existing_pos in label_positions.values():
                    if np.linalg.norm(label_pos - existing_pos) < min_label_distance:
                        too_close = True
                        break
                
                if not too_close:
                    label_positions[(node1, node2)] = label_pos
                    found_position = True
                    successful_labels += 1
                    break
            
            if found_position:
                break
    
    print(f"Successfully positioned {successful_labels} edge labels out of {total_attempts} edges ({successful_labels/total_attempts*100:.1f}%)")
    
    # Draw all the edge weight labels
    for (node1, node2, weight) in all_edges_with_weights:
        if (node1, node2) in label_positions:
            label_pos = label_positions[(node1, node2)]
            
            # Clean label text
            label_text = f'{weight:.0f}'
            
            # High contrast label for edge weights
            ax.text(label_pos[0], label_pos[1], label_text,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=7,
                   fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.15", 
                            facecolor="white", 
                            edgecolor="black",
                            linewidth=1,
                            alpha=0.95),
                   zorder=10)
    
    # Create colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                              norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    cbar_edges = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar_edges.set_label('Edge Weight (km)', rotation=270, labelpad=20)
    
    # Create hover annotation
    hover_annotation = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                                  bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                                  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                  fontsize=10, fontweight='bold', zorder=20)
    hover_annotation.set_visible(False)
    
    # Hover event handler
    def on_hover(event):
        if event.inaxes != ax:
            hover_annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        
        # Check if mouse is over any node
        for node_id, node_info in node_artists.items():
            node_pos = positions[node_id]
            # Calculate distance from mouse to node
            if event.xdata is not None and event.ydata is not None:
                distance = np.sqrt((event.xdata - node_pos[0])**2 + (event.ydata - node_pos[1])**2)
                
                # If close enough to node (adjust threshold as needed)
                threshold = 0.1 if node_info['has_stations'] else 0.05
                if distance < threshold:
                    # Show node information
                    if node_info['has_stations']:
                        text = f"Node {node_id}\n{node_info['station_count']} charging stations"
                    else:
                        text = f"Node {node_id}\nNo charging stations"
                    
                    hover_annotation.xy = (node_pos[0], node_pos[1])
                    hover_annotation.set_text(text)
                    hover_annotation.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        
        # If not hovering over any node, hide annotation
        hover_annotation.set_visible(False)
        fig.canvas.draw_idle()
    
    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    # Set appearance
    ax.set_xlim(-8, 2)
    ax.set_ylim(50, 61)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Create comprehensive title
    title = f'UK EV Charging Network: Stations & Weighted Graph (Interactive)\n'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add comprehensive legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
               markersize=12, label='Charging stations (size = # stations)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=8, label='No charging stations'),
        Line2D([0], [0], color='red', linewidth=3, label='Long distance edges'),
        Line2D([0], [0], color='blue', linewidth=3, label='Short distance edges'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    print("Interactive features enabled:")
    print("- Hover over any node to see its ID and charging station info")
    print("- Keep the plot window open to use hover functionality")
    
    return fig, ax

def print_assignment_summary(graph):
    """Print summary statistics of the assignment"""
    total_nodes = len(graph.nodes)
    nodes_with_stations = 0
    total_stations = 0
    
    # Edge weight statistics
    weights = [edge[2]['weight'] for edge in graph.edges(data=True) if 'weight' in edge[2]]
    
    for node in graph.nodes:
        stations = graph.nodes[node]['charging_stations']
        if stations:
            nodes_with_stations += 1
            total_stations += len(stations)
    
    print("\n" + "="*50)
    print("ASSIGNMENT SUMMARY")
    print("="*50)
    print(f"Total graph nodes: {total_nodes}")
    print(f"Nodes with stations: {nodes_with_stations}")
    print(f"Coverage: {(nodes_with_stations/total_nodes)*100:.1f}%")
    print(f"Total stations assigned: {total_stations}")
    
    if weights:
        print("\nEDGE WEIGHT STATISTICS")
        print("-" * 25)
        print(f"Total edges: {len(weights)}")
        print(f"Min distance: {min(weights):.2f}km")
        print(f"Max distance: {max(weights):.2f}km")
        print(f"Average distance: {np.mean(weights):.2f}km")
        print(f"Median distance: {np.median(weights):.2f}km")
        
    print("="*50)


import matplotlib.pyplot as plt
import numpy as np

def plot_anxiety_T_profile(mean_T, std_T, driver_count_T=None, title="EV Driver Anxiety Throughout the Day"):
    """
    Plot shaded line graph of driver anxiety throughout the day (T1-T24) with optional driver count
    
    Args:
        mean_T: List of 24 mean anxiety values (one per hour T1-T24)
        std_T: List of 24 standard deviation values (one per hour T1-T24)
        driver_count_T: Optional list of 24 average driver counts (one per hour T1-T24)
        title: Plot title
    """
    # Validate input
    assert len(mean_T) == 24, f"Expected 24 mean values, got {len(mean_T)}"
    assert len(std_T) == 24, f"Expected 24 std values, got {len(std_T)}"
    if driver_count_T is not None:
        assert len(driver_count_T) == 24, f"Expected 24 driver count values, got {len(driver_count_T)}"
    
    # Convert to numpy arrays for easier manipulation
    mean_T = np.array(mean_T)
    std_T = np.array(std_T)
    
    # Create x-axis values (T1 to T24)
    x = np.arange(1, 25)  # 1, 2, 3, ..., 24
    
    # Calculate confidence bands (mean ± 1σ), clipped to [0, 1]
    upper_band = np.clip(mean_T + std_T, 0, 1)
    lower_band = np.clip(mean_T - std_T, 0, 1)
    
    # Create the plot with dual y-axes if driver count is provided
    if driver_count_T is not None:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx()  # Create second y-axis
        driver_count_T = np.array(driver_count_T)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = None
    
    # Plot the shaded confidence band first (so it appears behind the line)
    ax1.fill_between(x, lower_band, upper_band, 
                    alpha=0.3, color='lightcoral', 
                    label='±1σ confidence band')
    
    # Plot the mean anxiety line
    line1 = ax1.plot(x, mean_T, 
                    color='darkred', linewidth=2.5, 
                    marker='o', markersize=4,
                    label='Mean anxiety')
    
    # Plot driver count if provided
    if driver_count_T is not None and ax2 is not None:
        line2 = ax2.plot(x, driver_count_T,
                        color='steelblue', linewidth=2.0,
                        marker='s', markersize=3,
                        linestyle='--', alpha=0.8,
                        label='Active drivers')
        
        # Customize second y-axis
        ax2.set_ylabel('Number of Active Drivers', fontsize=12, fontweight='bold', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        
        # Set reasonable limits for driver count
        if np.max(driver_count_T) > 0:
            ax2.set_ylim(0, np.max(driver_count_T) * 1.1)
    
    # Customize the primary plot (anxiety)
    ax1.set_xlabel('Time of Day (T1–T24)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Driver Anxiety (0–1)', fontsize=12, fontweight='bold', color='darkred')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='y', labelcolor='darkred')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i}' for i in x], rotation=45, ha='right')
    
    # Set y-axis limits and add horizontal reference lines
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.25, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax1.axhline(y=0.75, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)
    else:
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add annotations for peak and low periods
    if len(mean_T) > 0:
        # Find peak anxiety hour
        peak_hour = np.argmax(mean_T) + 1  # +1 because x starts at 1
        peak_value = mean_T[peak_hour - 1]
        
        # Find lowest anxiety hour (excluding zeros)
        non_zero_indices = mean_T > 0
        if np.any(non_zero_indices):
            low_hour = np.argmin(np.where(non_zero_indices, mean_T, np.inf)) + 1
            low_value = mean_T[low_hour - 1]
            
            # Add annotations
            if peak_value > 0:
                ax1.annotate(f'Peak Anxiety: T{peak_hour}\n({peak_value:.3f})',
                           xy=(peak_hour, peak_value),
                           xytext=(peak_hour + 2, peak_value + 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if low_value > 0 and low_hour != peak_hour:
                ax1.annotate(f'Low Anxiety: T{low_hour}\n({low_value:.3f})',
                           xy=(low_hour, low_value),
                           xytext=(low_hour - 2, low_value + 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7),
                           fontsize=10, ha='right',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Add driver count peak annotation if available
    if driver_count_T is not None and np.max(driver_count_T) > 0:
        peak_driver_hour = np.argmax(driver_count_T) + 1
        peak_driver_count = driver_count_T[peak_driver_hour - 1]
        
        if ax2 is not None:
            ax2.annotate(f'Peak Drivers: T{peak_driver_hour}\n({peak_driver_count:.0f} active)',
                        xy=(peak_driver_hour, peak_driver_count),
                        xytext=(peak_driver_hour - 3, peak_driver_count * 0.9),
                        arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.7),
                        fontsize=9, ha='right', color='steelblue',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))
    
    # Add time period labels on x-axis
    time_labels = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                   '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                   '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                   '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
    
    # Add secondary x-axis with hour labels
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(x[::3])  # Every 3 hours
    ax3.set_xticklabels([time_labels[i-1] for i in x[::3]], fontsize=9, alpha=0.7)
    ax3.set_xlabel('Hour of Day', fontsize=10, alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Print some statistics
    total_samples = np.sum([1 for m in mean_T if m > 0])  # Count non-zero hours
    print(f"\nAnxiety Profile Statistics:")
    print(f"Hours with data: {total_samples}/24")
    if total_samples > 0:
        non_zero_mean = mean_T[mean_T > 0]
        print(f"Overall average anxiety: {np.mean(non_zero_mean):.4f}")
        print(f"Peak anxiety: {np.max(mean_T):.4f} at T{np.argmax(mean_T) + 1}")
        if np.any(mean_T > 0):
            min_non_zero = np.min(mean_T[mean_T > 0])
            min_hour = np.argmin(np.where(mean_T > 0, mean_T, np.inf)) + 1
            print(f"Lowest anxiety: {min_non_zero:.4f} at T{min_hour}")
    
    # Print driver count statistics if available
    if driver_count_T is not None:
        print(f"\nDriver Count Statistics:")
        print(f"Peak active drivers: {np.max(driver_count_T):.0f} at T{np.argmax(driver_count_T) + 1}")
        print(f"Average active drivers: {np.mean(driver_count_T):.1f}")
        print(f"Driver count range: {np.min(driver_count_T):.0f} - {np.max(driver_count_T):.0f}")
    
    # Show the plot
    plt.show()
    
    return fig, (ax1, ax2) if ax2 is not None else (ax1,)


def print_anxiety_summary_table(mean_T, std_T, count_T, driver_count_T=None):
    """
    Print a formatted table of anxiety statistics by hour with optional driver count
    
    Args:
        mean_T: List of 24 mean anxiety values
        std_T: List of 24 standard deviation values  
        count_T: List of 24 sample counts
        driver_count_T: Optional list of 24 average driver counts
    """
    print(f"\n{'=' * 85}")
    print(f"{'HOURLY ANXIETY SUMMARY TABLE':^85}")
    print(f"{'=' * 85}")
    
    if driver_count_T is not None:
        print(f"{'Hour':<6} {'Time':<12} {'Samples':<8} {'Mean':<10} {'Std Dev':<10} {'Range':<12} {'Drivers':<8}")
        print(f"{'-' * 85}")
    else:
        print(f"{'Hour':<6} {'Time':<12} {'Samples':<8} {'Mean':<10} {'Std Dev':<10} {'Range':<12}")
        print(f"{'-' * 70}")
    
    for i in range(24):
        hour_label = f"T{i+1}"
        time_range = f"{i:02d}:00-{(i+1)%24:02d}:00"
        samples = count_T[i]
        mean = mean_T[i]
        std = std_T[i]
        
        if samples > 0:
            range_str = f"[{max(0, mean-std):.3f}, {min(1, mean+std):.3f}]"
            
            if driver_count_T is not None:
                drivers = driver_count_T[i]
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {mean:<10.4f} {std:<10.4f} {range_str:<12} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {mean:<10.4f} {std:<10.4f} {range_str:<12}")
        else:
            if driver_count_T is not None:
                drivers = driver_count_T[i] if driver_count_T else 0
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    
    print(f"{'-' * 85}" if driver_count_T is not None else f"{'-' * 70}")
    total_samples = sum(count_T)
    hours_with_data = sum(1 for c in count_T if c > 0)
    print(f"Total anxiety samples: {total_samples}, Hours with data: {hours_with_data}/24")
    
    if driver_count_T is not None:
        avg_drivers = sum(driver_count_T) / 24
        peak_drivers = max(driver_count_T)
        peak_hour = driver_count_T.index(peak_drivers) + 1
        print(f"Average active drivers: {avg_drivers:.1f}, Peak: {peak_drivers:.0f} drivers at T{peak_hour}")
    
    print(f"{'=' * 85}" if driver_count_T is not None else f"{'=' * 70}")



def plot_queue_time_T_profile(queue_mean_T, queue_std_T, driver_count_T=None, title="EV Charging Queue Times Throughout the Day"):
    """
    Plot queue time throughout the day (T1-T24) with optional driver count
    
    Args:
        queue_mean_T: List of 24 mean queue time values in minutes (one per hour T1-T24)
        queue_std_T: List of 24 standard deviation values in minutes (one per hour T1-T24)
        driver_count_T: Optional list of 24 average driver counts (one per hour T1-T24)
        title: Plot title
    """
    # Validate input
    assert len(queue_mean_T) == 24, f"Expected 24 queue mean values, got {len(queue_mean_T)}"
    assert len(queue_std_T) == 24, f"Expected 24 queue std values, got {len(queue_std_T)}"
    if driver_count_T is not None:
        assert len(driver_count_T) == 24, f"Expected 24 driver count values, got {len(driver_count_T)}"
    
    # Convert to numpy arrays for easier manipulation
    queue_mean_T = np.array(queue_mean_T)
    queue_std_T = np.array(queue_std_T)
    
    # Create x-axis values (T1 to T24)
    x = np.arange(1, 25)  # 1, 2, 3, ..., 24
    
    # Calculate confidence bands (mean ± 1σ), clipped to non-negative values
    upper_band = np.clip(queue_mean_T + queue_std_T, 0, None)
    lower_band = np.clip(queue_mean_T - queue_std_T, 0, None)
    
    # Create the plot with dual y-axes if driver count is provided
    if driver_count_T is not None:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx()  # Create second y-axis
        driver_count_T = np.array(driver_count_T)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = None
    
    # Plot the shaded confidence band first (so it appears behind the line)
    ax1.fill_between(x, lower_band, upper_band, 
                    alpha=0.3, color='lightblue', 
                    label='±1σ confidence band')
    
    # Plot the mean queue time line
    line1 = ax1.plot(x, queue_mean_T, 
                    color='navy', linewidth=2.5, 
                    marker='o', markersize=4,
                    label='Mean queue time')
    
    # Plot driver count if provided
    if driver_count_T is not None and ax2 is not None:
        line2 = ax2.plot(x, driver_count_T,
                        color='darkorange', linewidth=2.0,
                        marker='s', markersize=3,
                        linestyle='--', alpha=0.8,
                        label='Active drivers')
        
        # Customize second y-axis
        ax2.set_ylabel('Number of Active Drivers', fontsize=12, fontweight='bold', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        # Set reasonable limits for driver count
        if np.max(driver_count_T) > 0:
            ax2.set_ylim(0, np.max(driver_count_T) * 1.1)
    
    # Customize the primary plot (queue time)
    ax1.set_xlabel('Time of Day (T1–T24)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Queue Time (minutes)', fontsize=12, fontweight='bold', color='navy')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='y', labelcolor='navy')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i}' for i in x], rotation=45, ha='right')
    
    # Set y-axis limits for queue time (start from 0)
    max_queue_time = np.max(upper_band) if np.max(upper_band) > 0 else 10
    ax1.set_ylim(0, max_queue_time * 1.1)
    
    # Add horizontal reference lines for queue time
    if max_queue_time > 5:
        ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='5 min threshold')
    if max_queue_time > 10:
        ax1.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='10 min threshold')
    if max_queue_time > 20:
        ax1.axhline(y=20, color='red', linestyle=':', alpha=0.7, linewidth=1, label='20 min tolerance limit')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)
    else:
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add annotations for peak and low periods
    if len(queue_mean_T) > 0 and np.max(queue_mean_T) > 0:
        # Find peak queue time hour
        peak_hour = np.argmax(queue_mean_T) + 1  # +1 because x starts at 1
        peak_value = queue_mean_T[peak_hour - 1]
        
        # Find lowest queue time hour (excluding zeros)
        non_zero_indices = queue_mean_T > 0
        if np.any(non_zero_indices):
            low_hour = np.argmin(np.where(non_zero_indices, queue_mean_T, np.inf)) + 1
            low_value = queue_mean_T[low_hour - 1]
            
            # Add annotations
            if peak_value > 0:
                ax1.annotate(f'Peak Queue: T{peak_hour}\n({peak_value:.1f} min)',
                           xy=(peak_hour, peak_value),
                           xytext=(peak_hour + 2, peak_value + max_queue_time * 0.1),
                           arrowprops=dict(arrowstyle='->', color='navy', alpha=0.7),
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
            if low_value > 0 and low_hour != peak_hour:
                ax1.annotate(f'Low Queue: T{low_hour}\n({low_value:.1f} min)',
                           xy=(low_hour, low_value),
                           xytext=(low_hour - 2, low_value + max_queue_time * 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7),
                           fontsize=10, ha='right',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))
    
    # Add driver count peak annotation if available
    if driver_count_T is not None and np.max(driver_count_T) > 0:
        peak_driver_hour = np.argmax(driver_count_T) + 1
        peak_driver_count = driver_count_T[peak_driver_hour - 1]
        
        if ax2 is not None:
            ax2.annotate(f'Peak Drivers: T{peak_driver_hour}\n({peak_driver_count:.0f} active)',
                        xy=(peak_driver_hour, peak_driver_count),
                        xytext=(peak_driver_hour - 3, peak_driver_count * 0.9),
                        arrowprops=dict(arrowstyle='->', color='darkorange', alpha=0.7),
                        fontsize=9, ha='right', color='darkorange',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="moccasin", alpha=0.8))
    
    # Add time period labels on x-axis
    time_labels = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                   '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                   '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                   '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
    
    # Add secondary x-axis with hour labels
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(x[::3])  # Every 3 hours
    ax3.set_xticklabels([time_labels[i-1] for i in x[::3]], fontsize=9, alpha=0.7)
    ax3.set_xlabel('Hour of Day', fontsize=10, alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Print some statistics
    total_samples = np.sum([1 for m in queue_mean_T if m > 0])  # Count non-zero hours
    print(f"\nQueue Time Profile Statistics:")
    print(f"Hours with queue data: {total_samples}/24")
    if total_samples > 0:
        non_zero_queue = queue_mean_T[queue_mean_T > 0]
        print(f"Overall average queue time: {np.mean(non_zero_queue):.2f} minutes")
        print(f"Peak queue time: {np.max(queue_mean_T):.2f} minutes at T{np.argmax(queue_mean_T) + 1}")
        if np.any(queue_mean_T > 0):
            min_non_zero = np.min(queue_mean_T[queue_mean_T > 0])
            min_hour = np.argmin(np.where(queue_mean_T > 0, queue_mean_T, np.inf)) + 1
            print(f"Lowest queue time: {min_non_zero:.2f} minutes at T{min_hour}")
    
    # Print driver count statistics if available
    if driver_count_T is not None:
        print(f"\nDriver Count Statistics:")
        print(f"Peak active drivers: {np.max(driver_count_T):.0f} at T{np.argmax(driver_count_T) + 1}")
        print(f"Average active drivers: {np.mean(driver_count_T):.1f}")
        print(f"Driver count range: {np.min(driver_count_T):.0f} - {np.max(driver_count_T):.0f}")
    
    # Show the plot
    plt.show()
    
    return fig, (ax1, ax2) if ax2 is not None else (ax1,)


def print_queue_summary_table(queue_mean_T, queue_std_T, queue_count_T, driver_count_T=None):
    """
    Print a formatted table of queue time statistics by hour with optional driver count
    
    Args:
        queue_mean_T: List of 24 mean queue time values in minutes
        queue_std_T: List of 24 standard deviation values in minutes
        queue_count_T: List of 24 sample counts for queue events
        driver_count_T: Optional list of 24 average driver counts
    """
    print(f"\n{'=' * 90}")
    print(f"{'HOURLY QUEUE TIME SUMMARY TABLE':^90}")
    print(f"{'=' * 90}")
    
    if driver_count_T is not None:
        print(f"{'Hour':<6} {'Time':<12} {'Queue Events':<12} {'Mean (min)':<12} {'Std Dev':<10} {'Range':<15} {'Drivers':<8}")
        print(f"{'-' * 90}")
    else:
        print(f"{'Hour':<6} {'Time':<12} {'Queue Events':<12} {'Mean (min)':<12} {'Std Dev':<10} {'Range':<15}")
        print(f"{'-' * 75}")
    
    for i in range(24):
        hour_label = f"T{i+1}"
        time_range = f"{i:02d}:00-{(i+1)%24:02d}:00"
        events = queue_count_T[i]
        mean = queue_mean_T[i]
        std = queue_std_T[i]
        
        if events > 0:
            range_str = f"[{max(0, mean-std):.1f}, {mean+std:.1f}]"
            
            if driver_count_T is not None:
                drivers = driver_count_T[i]
                print(f"{hour_label:<6} {time_range:<12} {events:<12} {mean:<12.2f} {std:<10.2f} {range_str:<15} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {events:<12} {mean:<12.2f} {std:<10.2f} {range_str:<15}")
        else:
            if driver_count_T is not None:
                drivers = driver_count_T[i] if driver_count_T else 0
                print(f"{hour_label:<6} {time_range:<12} {events:<12} {'N/A':<12} {'N/A':<10} {'N/A':<15} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {events:<12} {'N/A':<12} {'N/A':<10} {'N/A':<15}")
    
    print(f"{'-' * 90}" if driver_count_T is not None else f"{'-' * 75}")
    total_events = sum(queue_count_T)
    hours_with_data = sum(1 for c in queue_count_T if c > 0)
    print(f"Total queue events: {total_events}, Hours with queue data: {hours_with_data}/24")
    
    if total_events > 0:
        # Calculate overall statistics
        total_queue_time = sum(queue_mean_T[i] * queue_count_T[i] for i in range(24) if queue_count_T[i] > 0)
        avg_queue_time = total_queue_time / total_events if total_events > 0 else 0
        print(f"Average queue time across all events: {avg_queue_time:.2f} minutes")
        
        # Find peak periods
        if any(queue_mean_T):
            peak_hour = queue_mean_T.index(max(queue_mean_T)) + 1
            peak_time = max(queue_mean_T)
            print(f"Peak queue time: {peak_time:.2f} minutes at T{peak_hour}")
    
    if driver_count_T is not None:
        avg_drivers = sum(driver_count_T) / 24
        peak_drivers = max(driver_count_T)
        peak_hour = driver_count_T.index(peak_drivers) + 1
        print(f"Average active drivers: {avg_drivers:.1f}, Peak: {peak_drivers:.0f} drivers at T{peak_hour}")
    
    print(f"{'=' * 90}" if driver_count_T is not None else f"{'=' * 75}")

def plot_anxiety_T_profile(mean_T, std_T, driver_count_T=None, title="EV Driver Anxiety Throughout the Day (Public Charging)"):
    """
    Plot shaded line graph of driver anxiety throughout the day (T1-T24) with optional driver count
    
    Args:
        mean_T: List of 24 mean anxiety values (one per hour T1-T24)
        std_T: List of 24 standard deviation values (one per hour T1-T24)
        driver_count_T: Optional list of 24 average driver counts (one per hour T1-T24)
        title: Plot title
    """
    # Validate input
    assert len(mean_T) == 24, f"Expected 24 mean values, got {len(mean_T)}"
    assert len(std_T) == 24, f"Expected 24 std values, got {len(std_T)}"
    if driver_count_T is not None:
        assert len(driver_count_T) == 24, f"Expected 24 driver count values, got {len(driver_count_T)}"
    
    # Convert to numpy arrays for easier manipulation
    mean_T = np.array(mean_T)
    std_T = np.array(std_T)
    
    # Create x-axis values (T1 to T24)
    x = np.arange(1, 25)  # 1, 2, 3, ..., 24
    
    # Calculate confidence bands (mean ± 1σ), clipped to [0, 1]
    upper_band = np.clip(mean_T + std_T, 0, 1)
    lower_band = np.clip(mean_T - std_T, 0, 1)
    
    # Create the plot with dual y-axes if driver count is provided
    if driver_count_T is not None:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx()  # Create second y-axis
        driver_count_T = np.array(driver_count_T)
    else:
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = None
    
    # Plot the shaded confidence band first (so it appears behind the line)
    ax1.fill_between(x, lower_band, upper_band, 
                    alpha=0.3, color='lightcoral', 
                    label='±1σ confidence band')
    
    # Plot the mean anxiety line
    line1 = ax1.plot(x, mean_T, 
                    color='darkred', linewidth=2.5, 
                    marker='o', markersize=4,
                    label='Mean anxiety')
    
    # Plot driver count if provided
    if driver_count_T is not None and ax2 is not None:
        line2 = ax2.plot(x, driver_count_T,
                        color='steelblue', linewidth=2.0,
                        marker='s', markersize=3,
                        linestyle='--', alpha=0.8,
                        label='Active drivers')
        
        # Customize second y-axis
        ax2.set_ylabel('Number of Active Drivers', fontsize=12, fontweight='bold', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        
        # Set reasonable limits for driver count
        if np.max(driver_count_T) > 0:
            ax2.set_ylim(0, np.max(driver_count_T) * 1.1)
    
    # Customize the primary plot (anxiety)
    ax1.set_xlabel('Time of Day (T1–T24)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Driver Anxiety (0–1)', fontsize=12, fontweight='bold', color='darkred')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='y', labelcolor='darkred')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i}' for i in x], rotation=45, ha='right')
    
    # Set y-axis limits and add horizontal reference lines
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.25, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax1.axhline(y=0.75, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)
    else:
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add annotations for peak and low periods
    if len(mean_T) > 0:
        # Find peak anxiety hour
        peak_hour = np.argmax(mean_T) + 1  # +1 because x starts at 1
        peak_value = mean_T[peak_hour - 1]
        
        # Find lowest anxiety hour (excluding zeros)
        non_zero_indices = mean_T > 0
        if np.any(non_zero_indices):
            low_hour = np.argmin(np.where(non_zero_indices, mean_T, np.inf)) + 1
            low_value = mean_T[low_hour - 1]
            
            # Add annotations
            if peak_value > 0:
                ax1.annotate(f'Peak Anxiety: T{peak_hour}\n({peak_value:.3f})',
                           xy=(peak_hour, peak_value),
                           xytext=(peak_hour + 2, peak_value + 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if low_value > 0 and low_hour != peak_hour:
                ax1.annotate(f'Low Anxiety: T{low_hour}\n({low_value:.3f})',
                           xy=(low_hour, low_value),
                           xytext=(low_hour - 2, low_value + 0.1),
                           arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7),
                           fontsize=10, ha='right',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Add driver count peak annotation if available
    if driver_count_T is not None and np.max(driver_count_T) > 0:
        peak_driver_hour = np.argmax(driver_count_T) + 1
        peak_driver_count = driver_count_T[peak_driver_hour - 1]
        
        if ax2 is not None:
            ax2.annotate(f'Peak Drivers: T{peak_driver_hour}\n({peak_driver_count:.0f} active)',
                        xy=(peak_driver_hour, peak_driver_count),
                        xytext=(peak_driver_hour - 3, peak_driver_count * 0.9),
                        arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.7),
                        fontsize=9, ha='right', color='steelblue',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.8))
    
    # Add time period labels on x-axis
    time_labels = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                   '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                   '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                   '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
    
    # Add secondary x-axis with hour labels
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(x[::3])  # Every 3 hours
    ax3.set_xticklabels([time_labels[i-1] for i in x[::3]], fontsize=9, alpha=0.7)
    ax3.set_xlabel('Hour of Day', fontsize=10, alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Print some statistics
    total_samples = np.sum([1 for m in mean_T if m > 0])  # Count non-zero hours
    print(f"\nAnxiety Profile Statistics:")
    print(f"Hours with data: {total_samples}/24")
    if total_samples > 0:
        non_zero_mean = mean_T[mean_T > 0]
        print(f"Overall average anxiety: {np.mean(non_zero_mean):.4f}")
        print(f"Peak anxiety: {np.max(mean_T):.4f} at T{np.argmax(mean_T) + 1}")
        if np.any(mean_T > 0):
            min_non_zero = np.min(mean_T[mean_T > 0])
            min_hour = np.argmin(np.where(mean_T > 0, mean_T, np.inf)) + 1
            print(f"Lowest anxiety: {min_non_zero:.4f} at T{min_hour}")
    
    # Print driver count statistics if available
    if driver_count_T is not None:
        print(f"\nDriver Count Statistics:")
        print(f"Peak active drivers: {np.max(driver_count_T):.0f} at T{np.argmax(driver_count_T) + 1}")
        print(f"Average active drivers: {np.mean(driver_count_T):.1f}")
        print(f"Driver count range: {np.min(driver_count_T):.0f} - {np.max(driver_count_T):.0f}")
    
    # Show the plot
    plt.show()
    
    return fig, (ax1, ax2) if ax2 is not None else (ax1,)


def print_anxiety_summary_table(mean_T, std_T, count_T, driver_count_T=None):
    """
    Print a formatted table of anxiety statistics by hour with optional driver count
    
    Args:
        mean_T: List of 24 mean anxiety values
        std_T: List of 24 standard deviation values  
        count_T: List of 24 sample counts
        driver_count_T: Optional list of 24 average driver counts
    """
    print(f"\n{'=' * 85}")
    print(f"{'HOURLY ANXIETY SUMMARY TABLE':^85}")
    print(f"{'=' * 85}")
    
    if driver_count_T is not None:
        print(f"{'Hour':<6} {'Time':<12} {'Samples':<8} {'Mean':<10} {'Std Dev':<10} {'Range':<12} {'Drivers':<8}")
        print(f"{'-' * 85}")
    else:
        print(f"{'Hour':<6} {'Time':<12} {'Samples':<8} {'Mean':<10} {'Std Dev':<10} {'Range':<12}")
        print(f"{'-' * 70}")
    
    for i in range(24):
        hour_label = f"T{i+1}"
        time_range = f"{i:02d}:00-{(i+1)%24:02d}:00"
        samples = count_T[i]
        mean = mean_T[i]
        std = std_T[i]
        
        if samples > 0:
            range_str = f"[{max(0, mean-std):.3f}, {min(1, mean+std):.3f}]"
            
            if driver_count_T is not None:
                drivers = driver_count_T[i]
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {mean:<10.4f} {std:<10.4f} {range_str:<12} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {mean:<10.4f} {std:<10.4f} {range_str:<12}")
        else:
            if driver_count_T is not None:
                drivers = driver_count_T[i] if driver_count_T else 0
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {drivers:<8.0f}")
            else:
                print(f"{hour_label:<6} {time_range:<12} {samples:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    
    print(f"{'-' * 85}" if driver_count_T is not None else f"{'-' * 70}")
    total_samples = sum(count_T)
    hours_with_data = sum(1 for c in count_T if c > 0)
    print(f"Total anxiety samples: {total_samples}, Hours with data: {hours_with_data}/24")
    
    if driver_count_T is not None:
        avg_drivers = sum(driver_count_T) / 24
        peak_drivers = max(driver_count_T)
        peak_hour = driver_count_T.index(peak_drivers) + 1
        print(f"Average active drivers: {avg_drivers:.1f}, Peak: {peak_drivers:.0f} drivers at T{peak_hour}")
    
    print(f"{'=' * 85}" if driver_count_T is not None else f"{'=' * 70}")