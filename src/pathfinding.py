import networkx as nx
from typing import List, Optional, Tuple

def get_shortest_path(graph, source_node, destination_node):
    """
    Get the shortest path between two nodes in the graph using weights
    
    Args:
        graph: NetworkX graph with weighted edges
        source_node: Starting node
        destination_node: Target node
        
    Returns:
        list: Shortest path as list of nodes, or None if no path exists
    """
    try:
        path = nx.shortest_path(graph, source_node, destination_node, weight='weight')
        if path:
            print(f"Path calculated from {source_node} to {destination_node}: {path[:5]}{'...' if len(path) > 5 else ''} (total: {len(path)} nodes, {len(path)-1} hops)")
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print(f"    No path found from {source_node} to {destination_node}")
        return None

def find_nearest_charging_station(graph, current_node, planned_route, max_range):
    """
    Find the best charging station considering the planned route
    Prioritizes stations on or near the planned route to minimize detours
    Uses weighted distances instead of hop count
    
    Args:
        graph: NetworkX graph with charging stations and weighted edges
        current_node: Current node position
        planned_route: List of nodes representing the planned route
        max_range: Maximum distance to search for stations
        
    Returns:
        int or None: Node ID of best charging station, or None if none found
    """
    print(f"    Looking for charging stations on route within {max_range}km of node {current_node}")
    
    # Get all stations within range using weighted distances
    all_nearby_stations = []
    
    try:
        distances = nx.single_source_dijkstra_path_length(graph, current_node, cutoff=max_range, weight='weight')
        
        for node, distance in distances.items():
            if node != current_node and 'charging_stations' in graph.nodes[node]:
                stations = graph.nodes[node]['charging_stations']
                if stations:
                    all_nearby_stations.append((node, distance))
        
    except nx.NodeNotFound:
        pass
    
    if not all_nearby_stations:
        print(f"    No charging stations found within {max_range}km!")
        return None
    
    print(f"    Found {len(all_nearby_stations)} charging stations within range")
    
    # Score stations based on route convenience
    scored_stations = []
    
    for station_node, distance in all_nearby_stations:
        # Check if station is on the planned route
        if station_node in planned_route:
            route_position = planned_route.index(station_node)
            current_position = planned_route.index(current_node) if current_node in planned_route else 0
            
            # Prefer stations ahead on the route
            if route_position > current_position:
                score = 1000 - distance  # High score for stations on route ahead
                priority = "ON_ROUTE_AHEAD"
            else:
                score = 500 - distance   # Medium score for stations on route behind
                priority = "ON_ROUTE_BEHIND"
        else:
            # For off-route stations, calculate detour cost using weighted distances
            detour_cost = distance
            
            # Find closest point on route to this station
            min_return_distance = float('inf')
            closest_route_node = None
            
            for route_node in planned_route:
                try:
                    return_distance = nx.shortest_path_length(graph, station_node, route_node, weight='weight')
                    if return_distance < min_return_distance:
                        min_return_distance = return_distance
                        closest_route_node = route_node
                except:
                    continue
            
            if closest_route_node:
                total_detour = detour_cost + min_return_distance
                score = 100 - total_detour  # Lower score for stations requiring detours
                priority = f"DETOUR_{total_detour:.1f}"
            else:
                score = -distance  # Lowest score if can't return to route
                priority = "NO_RETURN_PATH"
        
        scored_stations.append((station_node, distance, score, priority))
    
    # Sort by score (highest first)
    scored_stations.sort(key=lambda x: x[2], reverse=True)
    
    # Show top options
    print("    Top charging station options:")
    for i, (station_node, distance, score, priority) in enumerate(scored_stations[:5]):
        print(f"    {i+1}. Node {station_node}: distance={distance:.2f}km, score={score:.0f}, type={priority}")
    
    # Return best option
    best_station = scored_stations[0][0]
    best_priority = scored_stations[0][3]
    print(f"    Selected best station: node {best_station} ({best_priority})")
    
    return best_station

def calculate_charging_needed(current_soc, battery_range, remaining_distance):
    """
    Calculate if charging is needed to complete the journey
    
    Args:
        current_soc: Current state of charge (0.0 to 1.0)
        battery_range: Maximum distance possible with full battery (in km)
        remaining_distance: Remaining distance to destination (in km)
        
    Returns:
        tuple: (needs_charging: bool, current_range: float, deficit: float)
    """
    current_range = battery_range * current_soc
    deficit = remaining_distance - current_range
    needs_charging = deficit > 0
    
    return needs_charging, current_range, max(0, deficit)

def travel_to_charging_station(env, car_id, current_node, charging_node, graph, travel_time_per_unit=1.0):
    """
    Handle travel from current location to a charging station using weighted distances
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        current_node: Starting node
        charging_node: Destination charging station node
        graph: NetworkX graph for pathfinding with weighted edges
        travel_time_per_unit: Time per distance unit in simulation units
        
    Yields:
        SimPy timeout events for travel time
        
    Returns:
        bool: True if travel successful, False if no path found
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Detouring to charging station at node {charging_node}")
    
    # Get path to charging station using weights
    path_to_station = get_shortest_path(graph, current_node, charging_node)
    
    if not path_to_station:
        print(f"[T={env.now:.1f}] Car {car_id}: No path to charging station {charging_node}!")
        return False
    
    print(f"[T={env.now:.1f}] Car {car_id}: Taking detour path: {path_to_station}")
    
    # Calculate total weighted distance for the path
    total_distance = 0
    for i in range(len(path_to_station) - 1):
        node1, node2 = path_to_station[i], path_to_station[i + 1]
        total_distance += graph.edges[node1, node2]['weight']
    
    if total_distance > 0:
        total_travel_time = total_distance * travel_time_per_unit
        yield env.timeout(total_travel_time)
        print(f"[T={env.now:.1f}] Car {car_id}: Arrived at charging station {charging_node} (traveled distance: {total_distance:.2f}km)")
    else:
        print(f"[T={env.now:.1f}] Car {car_id}: Already at charging station {charging_node}")
    
    return True

def travel_to_next_node(env, car_id, driver, graph, travel_time_per_km=0.01, battery_range_km=300):
    """
    Handle travel to the next node in the path using weighted distances
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        driver: EVDriver object
        graph: NetworkX graph with weighted edges
        travel_time_per_km: Time per kilometer in simulation units
        battery_range_km: Maximum distance possible with full battery (in km)
        
    Yields:
        SimPy timeout events for travel time
    """
    current_node = driver.get_current_node()
    next_node = driver.move_to_next_node()
    
    if next_node is None:
        return
    
    # Get the actual distance in km between these nodes
    distance_km = 0.0
    if graph.has_edge(current_node, next_node):
        distance_km = graph.edges[current_node, next_node]['weight']
    
    # Simulate travel time based on actual distance
    travel_time = distance_km * travel_time_per_km
    yield env.timeout(travel_time)
    
    # Update battery based on actual distance traveled
    consumption_per_km = 1.0 / battery_range_km
    battery_consumption = distance_km * consumption_per_km
    driver.consume_battery(battery_consumption)
    
    print(f"[T={env.now:.1f}] Car {car_id}: Moved to node {next_node} (traveled {distance_km:.2f}km), SoC: {driver.get_state_of_charge():.2f} ({driver.battery_percentage:.0f}%)")

def find_nearest_nodes_with_stations(graph, current_node, max_distance=50): #ONLY USED FOR TESTING
    """
    Find nearest nodes with charging stations using weighted distances
    
    Args:
        graph: NetworkX graph with charging stations and weighted edges
        current_node: Current node position
        max_distance: Maximum distance to search (in km)
        
    Returns:
        list: List of tuples (node_id, distance_km) sorted by distance
    """
    nearby_stations = []
    
    try:
        distances = nx.single_source_dijkstra_path_length(graph, current_node, cutoff=max_distance, weight='weight')
        
        for node, distance in distances.items():
            if node != current_node and 'charging_stations' in graph.nodes[node]:
                stations = graph.nodes[node]['charging_stations']
                if stations:
                    nearby_stations.append((node, distance))
        
        # Sort by distance
        nearby_stations.sort(key=lambda x: x[1])
        
    except nx.NodeNotFound:
        pass
    
    return nearby_stations