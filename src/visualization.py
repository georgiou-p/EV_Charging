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