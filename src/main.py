from station_assignment import assign_charging_stations_to_nodes
from visualization import visualize_stations_on_map, print_assignment_summary

def main():
    """
    Main function to run the entire charging station assignment process
    """
    
    print("Starting EV Charging Station Assignment Process...")
    
    # Configuration - paths relative to project root
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleanedM_charging_stations.json"
    
    try:
        # Assign charging stations to graph nodes using cKDTree
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is not None:
            # Print summary
            print_assignment_summary(graph)
            
            # Check assignment results
            nodes_with_stations = [node for node in graph.nodes 
                                 if len(graph.nodes[node]['charging_stations']) > 0]
            total_assigned = sum(len(graph.nodes[node]['charging_stations']) 
                               for node in graph.nodes)
            
            print(f"Nodes with charging stations: {len(nodes_with_stations)} out of {len(graph.nodes)}")
            print(f"Total charging stations assigned: {total_assigned}")
            
            # Test 
            node = 300
            stations = graph.nodes[node]['charging_stations']
            station = stations[0]
            print(station.get_station_id())
            connections = station.get_connections()
            print(connections[0])
            
            # Create visualization
            print("\nCreating visualization...")
            visualize_stations_on_map(graph, geojson_path)
            
            print("\n Process completed successfully!")
            return graph, node_stations
        else:
            print(" Assignment failed!")
            return None, None
            
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
        return None, None
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None, None

if __name__ == "__main__":
    graph, node_stations = main()