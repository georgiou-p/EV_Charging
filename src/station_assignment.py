import json
import numpy as np
from scipy.spatial import cKDTree
from libpysal import weights
import networkx as nx
import geopandas
from collections import defaultdict
from charging_station import EVChargingStation
from haversine import haversine, Unit

def assign_charging_stations_to_nodes(geojson_path, charging_data_path):
    """
    Assign charging stations to the nearest nodes in a graph created from UK districts.
    Uses EVChargingStation class objects instead of dictionaries.
    Graph edges are weighted by distance between centroids.
    
    Args:
        geojson_path (str): Path to the UK districts GeoJSON file
        charging_data_path (str): Path to the charging stations JSON file
    
    Returns:
        networkx.Graph: Graph with charging station objects assigned to nodes and weighted edges
        dict: Dictionary mapping node IDs to lists of EVChargingStation objects
    """
    
    # Load and create the graph from GeoJSON
    map_regions = geopandas.read_file(geojson_path)
    
    # Extract centroids (these become our graph nodes)
    centroids = np.column_stack((map_regions.centroid.x, map_regions.centroid.y))
    
    # Create Queen adjacency graph
    queen = weights.Queen.from_dataframe(map_regions)
    graph = queen.to_networkx()
    
    # Create positions dictionary for nodes
    positions = dict(zip(graph.nodes, centroids))
    
    # Add weights to edges based on Haversine distance in kilometers
    for edge in graph.edges():
        node1, node2 = edge
        pos1 = positions[node1]  # (longitude, latitude)
        pos2 = positions[node2]  # (longitude, latitude)
        
        # Calculate Haversine distance using library - expects (lat, lon) format
        distance_km = haversine((pos1[1], pos1[0]), (pos2[1], pos2[0]), unit=Unit.KILOMETERS)
        
        # Add distance as weight to the edge
        graph.edges[node1, node2]['weight'] = distance_km
    
    # Load charging station data
    with open(charging_data_path, 'r', encoding='utf-8') as f:
        charging_stations_data = json.load(f)
    
    # Handle both list and single object formats
    if isinstance(charging_stations_data, list):
        stations_list = charging_stations_data
    else:
        stations_list = [charging_stations_data]
    
    # Create KDTree for efficient nearest neighbor search
    node_coords = np.array([positions[node] for node in graph.nodes])
    kdtree = cKDTree(node_coords)
    
    # Assign each charging station to nearest node
    node_charging_stations = defaultdict(list)
    assigned_count = 0
    
    for station_data in stations_list:
        # Skip stations with invalid coordinates
        lat, lon = station_data.get('Latitude'), station_data.get('Longitude')
        if lat is None or lon is None or lat == 0 or lon == 0:
            continue
        
        try:
            # Create EVChargingStation object from JSON data
            station = EVChargingStation.from_json(station_data)
            
            # Find nearest node using KDTree
            station_coord = np.array([lon, lat])  # Note: lon, lat order for consistency
            distance, nearest_node_idx = kdtree.query(station_coord)
            
            # Get the actual node ID
            nearest_node = list(graph.nodes)[nearest_node_idx]
            
            # Assign station object to node
            node_charging_stations[nearest_node].append(station)
            assigned_count += 1
            
        except Exception as e:
            print(f"Error creating station object for ID {station_data.get('ID', 'unknown')}: {e}")
            continue
    
    # Add charging station objects as node attributes
    for node in graph.nodes:
        graph.nodes[node]['charging_stations'] = node_charging_stations.get(node, [])
        graph.nodes[node]['position'] = positions[node]
    
    # Manual edge addition: 322 <-> 323
    pos1 = positions[322]
    pos2 = positions[323]
    distance_km = haversine((pos1[1], pos1[0]), (pos2[1], pos2[0]), unit=Unit.KILOMETERS)
    graph.add_edge(322, 323, weight=distance_km)
    
    print(f"Successfully assigned {assigned_count} charging stations to {len([n for n in node_charging_stations])} nodes")
    print(f"Graph has weighted edges in kilometers. Sample weights: {[(edge[0], edge[1], f'{edge[2]['weight']:.2f}km') for edge in list(graph.edges(data=True))[:3]]}")
    
    return graph, dict(node_charging_stations)