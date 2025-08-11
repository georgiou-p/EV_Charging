from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
import networkx as nx

def test_distance_between_nodes():
    """Test the shortest distance between nodes 235 and 139"""
    
    print("=== Testing Distance Between Nodes 235 and 139 ===\n")
    
    # Load the graph
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleanedM_charging_stations.json"
    
    try:
        print("Loading graph and charging stations...")
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is None:
            print("Failed to load graph!")
            return
        
        print(f"Graph loaded successfully with {len(graph.nodes)} nodes\n")
        
        # Test nodes
        source_node = 235
        destination_node = 129
        
        print(f"Testing route from node {source_node} to node {destination_node}")
        print("-" * 50)
        
        # Check if nodes exist in graph
        if source_node not in graph.nodes:
            print(f"ERROR: Node {source_node} not found in graph!")
            return
        if destination_node not in graph.nodes:
            print(f"ERROR: Node {destination_node} not found in graph!")
            return
        
        print(f" Both nodes exist in the graph")
        
        # Method 1: Using EVDriver
        print(f"\nMethod 1: Using EVDriver class")
        driver = EVDriver(source_node, destination_node, 1.0, 20)
        path = driver.find_shortest_path(graph)
        
        if path:
            # Calculate total distance using EVDriver method
            total_distance_km = driver._calculate_path_distance(graph, path)
            print(f"EVDriver result:")
            print(f"  Path length: {len(path)} nodes")
            print(f"  Total distance: {total_distance_km:.2f} km")
            print(f"  Full path: {path}")
        else:
            print("EVDriver: No path found!")
        
        # Method 2: Direct NetworkX calculation
        print(f"\nMethod 2: Direct NetworkX calculation")
        try:
            # Get shortest path using NetworkX directly
            nx_path = nx.shortest_path(graph, source_node, destination_node, weight='weight')
            nx_distance = nx.shortest_path_length(graph, source_node, destination_node, weight='weight')
            
            print(f"NetworkX result:")
            print(f"  Path length: {len(nx_path)} nodes")
            print(f"  Total distance: {nx_distance:.2f} km")
            print(f"  Full path: {nx_path}")
            
        except nx.NetworkXNoPath:
            print("NetworkX: No path found!")
        except nx.NodeNotFound as e:
            print(f"NetworkX: Node error - {e}")
        
        # Method 3: Check direct connection
        print(f"\nMethod 3: Direct connection check")
        if graph.has_edge(source_node, destination_node):
            direct_distance = graph.edges[source_node, destination_node]['weight']
            print(f"Direct edge exists: {direct_distance:.2f} km")
        else:
            print("No direct edge between these nodes")
        
        # Method 4: Show node positions
        print(f"\nMethod 4: Node positions")
        if 'position' in graph.nodes[source_node]:
            pos1 = graph.nodes[source_node]['position']
            print(f"Node {source_node} position: ({pos1[0]:.6f}, {pos1[1]:.6f})")
        
        if 'position' in graph.nodes[destination_node]:
            pos2 = graph.nodes[destination_node]['position']
            print(f"Node {destination_node} position: ({pos2[0]:.6f}, {pos2[1]:.6f})")
        
        # Calculate straight-line distance using Haversine
        if 'position' in graph.nodes[source_node] and 'position' in graph.nodes[destination_node]:
            from haversine import haversine, Unit
            pos1 = graph.nodes[source_node]['position']  # (lon, lat)
            pos2 = graph.nodes[destination_node]['position']  # (lon, lat)
            
            # haversine expects (lat, lon)
            straight_distance = haversine((pos1[1], pos1[0]), (pos2[1], pos2[0]), unit=Unit.KILOMETERS)
            print(f"Straight-line distance: {straight_distance:.2f} km")
        
        # Method 5: Check charging stations at both nodes
        print(f"\nMethod 5: Charging station info")
        stations_235 = graph.nodes[source_node]['charging_stations']
        stations_139 = graph.nodes[destination_node]['charging_stations']
        
        print(f"Node {source_node} has {len(stations_235)} charging stations")
        print(f"Node {destination_node} has {len(stations_139)} charging stations")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distance_between_nodes()