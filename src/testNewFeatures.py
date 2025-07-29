from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import get_shortest_path, find_nearest_nodes_with_stations
import random

def test_new_features():
    """Test the new queue and pathfinding features"""
    
    print("=== Testing New Features ===\n")
    
    # Load the graph with charging stations
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleanedM_charging_stations.json"
    
    try:
        graph, node_stations = assign_charging_stations_to_nodes(geojson_path, stations_json_path)
        
        if graph is None:
            print("Failed to load graph")
            return
        
        print(f"Graph loaded with {len(graph.nodes)} nodes")
        
        # Test 1: Check charging station queue functionality
        print("\n1. Testing Charging Station Queue:")
        
        # Find a node with charging stations
        nodes_with_stations = [node for node in graph.nodes 
                             if len(graph.nodes[node]['charging_stations']) > 0]
        
        if nodes_with_stations:
            test_node = nodes_with_stations[0]
            stations = graph.nodes[test_node]['charging_stations']
            station = stations[0]
            
            print(f"Testing station: {station.get_station_id()}")
            print(f"Initial queue length: {station.get_queue_length()}")
            
            # Create test drivers
            driver1 = EVDriver(test_node, test_node + 10, 0.5, 20)
            driver2 = EVDriver(test_node, test_node + 20, 0.3, 30)
            
            # Add to queue
            station.add_to_queue(driver1)
            station.add_to_queue(driver2)
            
            print(f"Queue length after adding 2 drivers: {station.get_queue_length()}")
            
            # Remove from queue
            removed_driver = station.remove_from_queue()
            print(f"Removed driver: {removed_driver}")
            print(f"Queue length after removal: {station.get_queue_length()}")
        
        # Test 2: Check pathfinding functionality
        print("\n2. Testing Driver Pathfinding:")
        
        # Get random nodes for testing
        all_nodes = list(graph.nodes)
        if len(all_nodes) >= 2:
            source = random.choice(all_nodes)
            destination = random.choice([n for n in all_nodes if n != source])
            
            # Create a test driver
            test_driver = EVDriver(source, destination, 0.8, 20)
            print(f"Created driver: {test_driver}")
            
            # Find shortest path
            path = test_driver.find_shortest_path(graph)
            if path:
                print(f"Shortest path found with {len(path)} nodes")
                print(f"Path: {path[:5]}..." if len(path) > 5 else f"Path: {path}")
                
                # Test movement
                print(f"Current node: {test_driver.get_current_node()}")
                next_node = test_driver.move_to_next_node()
                print(f"After moving: {test_driver.get_current_node()}")
                
                # Check remaining distance
                remaining = test_driver.get_remaining_distance(graph)
                print(f"Remaining distance: {remaining}")
                
            else:
                print("No path found")
        
        # Test 3: Find nearby charging stations
        print("\n3. Testing Nearby Station Search:")
        
        if nodes_with_stations:
            test_node = random.choice(all_nodes)
            nearby = find_nearest_nodes_with_stations(graph, test_node, max_distance=10)
            print(f"From node {test_node}, found {len(nearby)} nearby stations")
            if nearby:
                print(f"Closest station at node {nearby[0][0]}, distance: {nearby[0][1]}")
        
        print("\n=== All Tests Completed ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_features()