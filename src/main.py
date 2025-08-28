from station_assignment import assign_charging_stations_to_nodes
from visualization import print_assignment_summary, visualize_complete_graph

def main():
    """
    Main function to run the charging station assignment and visualization
    """
    
    print("Starting EV Charging Station Assignment Process...")
    
    # Configuration
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleanedM_charging_stations.json"
    
    try:
        # Assign charging stations to graph nodes
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is not None:
            # Print summary with edge weight statistics
            print_assignment_summary(graph)
            
            # Check assignment results
            nodes_with_stations = [node for node in graph.nodes 
                                 if len(graph.nodes[node]['charging_stations']) > 0]
            total_assigned = sum(len(graph.nodes[node]['charging_stations']) 
                               for node in graph.nodes)
            
            print(f"Nodes with charging stations: {len(nodes_with_stations)} out of {len(graph.nodes)}")
            print(f"Total charging stations assigned: {total_assigned}")
            
            # Test existing functionality
            if nodes_with_stations:
                node = nodes_with_stations[0]
                stations = graph.nodes[node]['charging_stations']
                station = stations[0]
                print(f"Sample station ID: {station.get_station_id()}")
                connections = station.get_connections()
                if connections:
                    print(f"Sample connection: {connections[0]}")
            
            # === VISUALIZATION ===
            print("\n" + "="*60)
            print("CREATING COMPLETE VISUALIZATION")
            print("="*60)
            
            # Complete visualization showing both charging stations AND edge weights
            print("Creating complete graph with charging stations and edge weights...")
            visualize_complete_graph(graph, geojson_path)
            
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
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    graph, node_stations = main()