import matplotlib.pyplot as plt
import networkx as nx
import geopandas
import numpy as np

def visualize_stations_on_map(graph, map_regions_path=None, figsize=(15, 10)):
    """
    Visualize charging stations assigned to graph nodes
    
    Args:
        graph: NetworkX graph with charging stations assigned
        map_regions_path: Path to GeoJSON file (optional)
        figsize: Figure size
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

def print_assignment_summary(graph):
    """Print summary statistics of the assignment"""
    total_nodes = len(graph.nodes)
    nodes_with_stations = 0
    total_stations = 0
    
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
    print("="*50)