import networkx as nx
from typing import List, Optional, Tuple
from charging_utils import has_compatible_connector

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

def find_nearest_charging_station_simplified(graph, current_node, planned_route, max_range, connector_type, driver=None):
    """
    Find the best charging station considering route optimization with correct position tracking
    Returns specific station ID to go to directly
    
    Args:
        graph: NetworkX graph with charging stations
        current_node: Current node position
        planned_route: List of nodes in planned route
        max_range: Maximum search distance in km
        connector_type: Required connector type
        driver: EVDriver object to get accurate position information
        
    Returns:
        tuple: (best_node_id, best_station_id) or (None, None) if none found
    """
    print(f"    Looking for best charging station within {max_range}km")
    
    # Get current position in the planned route for accurate route scoring
    current_route_position = 0
    if driver and planned_route:
        current_route_position = driver.get_current_position_index()
        print(f"    Car is at position {current_route_position} in route with {len(planned_route)} nodes")
    elif current_node in planned_route:
        current_route_position = planned_route.index(current_node)
        print(f"    Car position determined by node lookup: position {current_route_position}")
    else:
        print(f"    Warning: Current node {current_node} not found in planned route")
    
    # Get all stations within range using weighted distances
    all_station_options = []
    
    try:
        distances = nx.single_source_dijkstra_path_length(graph, current_node, cutoff=max_range, weight='weight')
        
        for node, distance in distances.items():
            if node != current_node and 'charging_stations' in graph.nodes[node]:
                stations = graph.nodes[node]['charging_stations']
                
                # Score each individual station at this node
                for station in stations:
                    if has_compatible_connector([station], connector_type):
                        score = score_individual_station(
                            station, node, distance, planned_route, connector_type, current_route_position
                        )
                        all_station_options.append((node, station, score, distance))
        
    except nx.NodeNotFound:
        print(f"    Error: Node {current_node} not found in graph")
        pass
    
    if not all_station_options:
        print(f"    No compatible charging stations found within {max_range}km!")
        return None, None
    
    # Sort by score (highest first)
    all_station_options.sort(key=lambda x: x[2], reverse=True)
    
    print(f"    Found {len(all_station_options)} compatible charging stations within range")
    
    # Show top options with detailed route context
    print("    Top charging station options (with route analysis):")
    for i, (node, station, score, distance) in enumerate(all_station_options[:5]):
        queue_len = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
        estimated_wait = queue_len * 10  # minutes
        max_power = get_station_max_power(station, connector_type)
        
        # Determine route context for display
        route_context = "OFF_ROUTE"
        if node in planned_route:
            node_position = planned_route.index(node)
            if node_position > current_route_position:
                route_context = f"ON_ROUTE_AHEAD (pos {node_position}/{len(planned_route)-1})"
            elif node_position < current_route_position:
                route_context = f"ON_ROUTE_BEHIND (pos {node_position}/{len(planned_route)-1})"
            else:
                route_context = f"CURRENT_LOCATION (pos {node_position})"
        else:
            route_context = f"DETOUR ({distance:.1f}km off route)"
        
        print(f"    {i+1}. Station {station.get_station_id()} at Node {node}: score={score:.0f}")
        print(f"       {route_context} | Distance: {distance:.1f}km | Queue: {queue_len} cars ({estimated_wait}min) | Power: {max_power}kW")
    
    # Return best station
    best_node, best_station, best_score, best_distance = all_station_options[0]
    best_station_id = best_station.get_station_id()
    
    # Log the selection decision
    route_context = "off-route detour"
    if best_node in planned_route:
        best_node_position = planned_route.index(best_node)
        if best_node_position > current_route_position:
            route_context = f"on-route ahead (pos {best_node_position})"
        elif best_node_position < current_route_position:
            route_context = f"on-route behind (pos {best_node_position})"
        else:
            route_context = f"current location (pos {best_node_position})"
    
    print(f"     Selected: {best_station_id} at node {best_node} ({route_context}, score={best_score:.0f})")
    return best_node, best_station_id


def score_individual_station(station, node, distance, planned_route, connector_type, current_route_position=0):
    """
    Score an individual charging station with proper route-aware scoring
    
    Args:
        station: EVChargingStation object
        node: Node ID where station is located
        distance: Distance from current location in km
        planned_route: List of nodes in planned route
        connector_type: Required connector type
        current_route_position: Current position index in planned route
        
    Returns:
        float: Station score (higher is better)
    """
    score = 1000  # Base score
    
    # 1. Queue penalty (most important) - predictive based on current state
    queue_length = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
    estimated_wait_time = queue_length * 10  # Estimate 10 minutes per car in queue
    queue_penalty = estimated_wait_time * 10  # Heavy penalty for predicted wait time
    score -= queue_penalty
    
    # 2. Charging speed bonus (second most important)
    max_power = get_station_max_power(station, connector_type)
    if max_power >= 150:  # Ultra-rapid charging (150kW+)
        speed_bonus = 200
    elif max_power >= 50:  # Rapid charging (50-149kW)
        speed_bonus = 100
    elif max_power >= 22:  # Fast AC charging (22-49kW)
        speed_bonus = 50
    elif max_power >= 7:   # Standard charging (7-21kW)
        speed_bonus = 20
    else:  # Slow charging (<7kW) or unknown
        speed_bonus = -50  # Actually penalize very slow charging
    
    score += speed_bonus
    
    # 3. Route optimization (major factor) - FIXED WITH CORRECT POSITION LOGIC
    route_factor = 1.0
    route_description = "OFF_ROUTE"
    
    if planned_route and node in planned_route:
        try:
            route_position = planned_route.index(node)
            
            if route_position > current_route_position:
                # Station is AHEAD on our planned route - STRONG PREFERENCE
                positions_ahead = route_position - current_route_position
                # Give bigger bonus for stations further ahead (more strategic)
                if positions_ahead >= 3:
                    route_factor = 2.2  # Big bonus for strategic forward planning
                else:
                    route_factor = 2.0  # Good bonus for immediate forward progress
                route_description = f"AHEAD_{positions_ahead}_NODES"
                
            elif route_position < current_route_position:
                # Station is BEHIND us on route - DISCOURAGE BACKTRACKING
                positions_behind = current_route_position - route_position
                if positions_behind >= 3:
                    route_factor = 0.4  # Heavy penalty for going way back
                else:
                    route_factor = 0.6  # Moderate penalty for minor backtracking
                route_description = f"BEHIND_{positions_behind}_NODES"
                
            else:
                # Station is at current position - moderate bonus
                route_factor = 1.3
                route_description = "CURRENT_POSITION"
                
        except ValueError:
            # This shouldn't happen if node is in planned_route, but handle gracefully
            route_factor = 1.0
            route_description = "ROUTE_INDEX_ERROR"
    else:
        # Station requires detour from planned route
        if distance <= 15:  # Minor detour (under 15km)
            route_factor = 0.9
            route_description = f"MINOR_DETOUR_{distance:.1f}KM"
        elif distance <= 35:  # Medium detour (15-35km)
            route_factor = 0.7
            route_description = f"MEDIUM_DETOUR_{distance:.1f}KM"
        else:  # Major detour (over 35km)
            route_factor = 0.5
            route_description = f"MAJOR_DETOUR_{distance:.1f}KM"
    
    # 4. Distance penalty (minor factor - already considered in route scoring)
    distance_penalty = distance * 0.3  # Reduced from 0.5 since route factor handles most distance logic
    score -= distance_penalty
    
    # Apply route factor (this is where the major differentiation happens)
    score *= route_factor
    
    # Debug logging for route scoring decisions (uncomment for detailed analysis)
    # if node in planned_route:
    #     route_pos = planned_route.index(node)
    #     print(f"        Node {node}: pos {route_pos} vs current {current_route_position}, "
    #           f"factor {route_factor:.1f} ({route_description})")
    
    return max(score, 0)  # Ensure non-negative scores


def get_station_max_power(station, connector_type):
    """
    Get maximum power available at this station for the specified connector type
    
    Args:
        station: EVChargingStation object
        connector_type: Required connector type
        
    Returns:
        float: Maximum power in kW for compatible connections
    """
    max_power = 0
    for connection in station.get_connections():
        # Check for direct match or universal compatibility (type 0)
        if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
            if connection.power_kw and connection.power_kw > max_power:
                max_power = connection.power_kw
    return max_power


def travel_to_charging_station(env, car_id, current_node, charging_node, graph, driver, travel_time_per_unit=1.0):
    """
    Handle travel from current location to a charging station using weighted distances
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        current_node: Starting node
        charging_node: Destination charging station node
        graph: NetworkX graph for pathfinding with weighted edges
        driver: EVDriver object for battery management
        travel_time_per_unit: Time per distance unit in simulation units
        
    Yields:
        SimPy timeout events for travel time
        
    Returns:
        bool: True if travel successful, False if no path found
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Traveling to charging station at node {charging_node}")
    
    # Get path to charging station using weights
    path_to_station = get_shortest_path(graph, current_node, charging_node)
    
    if not path_to_station:
        print(f"[T={env.now:.1f}] Car {car_id}: ERROR - No path to charging station {charging_node}!")
        return False
    
    print(f"[T={env.now:.1f}] Car {car_id}: Route to charging station: {path_to_station[:5]}{'...' if len(path_to_station) > 5 else ''}")
    
    # Calculate total weighted distance for the path
    total_distance = 0
    for i in range(len(path_to_station) - 1):
        node1, node2 = path_to_station[i], path_to_station[i + 1]
        if graph.has_edge(node1, node2):
            total_distance += graph.edges[node1, node2]['weight']
    
    if total_distance > 0:
        # Simulate travel time based on actual distance
        total_travel_time = total_distance * travel_time_per_unit
        yield env.timeout(total_travel_time)

        # Update battery based on actual distance traveled
        consumption_per_km = 1.0 / driver.get_battery_capacity()
        battery_consumption = total_distance * consumption_per_km
        driver.consume_battery(battery_consumption)

        print(f"[T={env.now:.1f}] Car {car_id}: Arrived at charging station (node {charging_node}) - traveled {total_distance:.2f}km, SoC: {driver.get_state_of_charge():.2f} ({driver.battery_percentage:.0f}%)")
    else:
        print(f"[T={env.now:.1f}] Car {car_id}: Already at charging station location")
    
    return True


def travel_to_next_node(env, car_id, driver, graph, travel_time_per_km=0.01):
    """
    Handle travel to the next node in the planned path using weighted distances
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        driver: EVDriver object
        graph: NetworkX graph with weighted edges
        travel_time_per_km: Time per kilometer in simulation units
        
    Yields:
        SimPy timeout events for travel time
    """
    current_node = driver.get_current_node()
    next_node = driver.move_to_next_node()
    
    if next_node is None:
        print(f"[T={env.now:.1f}] Car {car_id}: No next node - reached end of path")
        return
    
    # Get the actual distance in km between these nodes
    distance_km = 0.0
    if graph.has_edge(current_node, next_node):
        distance_km = graph.edges[current_node, next_node]['weight']
    else:
        print(f"[T={env.now:.1f}] Car {car_id}: WARNING - No edge between {current_node} and {next_node}")
        return
    
    # Simulate travel time based on actual distance
    travel_time = distance_km * travel_time_per_km
    yield env.timeout(travel_time)
    
    # Update battery based on actual distance traveled
    consumption_per_km = 1.0 / driver.get_battery_capacity()
    battery_consumption = distance_km * consumption_per_km
    driver.consume_battery(battery_consumption)
    
    print(f"[T={env.now:.1f}] Car {car_id}: Moved to node {next_node} (traveled {distance_km:.2f}km), SoC: {driver.get_state_of_charge():.2f} ({driver.battery_percentage:.0f}%)")


def find_nearest_nodes_with_stations(graph, current_node, max_distance=50, connector_type=None): 
    """
    Find nearest nodes with charging stations using weighted distances
    Optionally filter by connector compatibility
    
    Args:
        graph: NetworkX graph with charging stations and weighted edges
        current_node: Current node position
        max_distance: Maximum distance to search (in km)
        connector_type: Optional connector type filter
        
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
                    # Apply connector filter if specified
                    if connector_type is None or has_compatible_connector(stations, connector_type):
                        nearby_stations.append((node, distance))
        
        # Sort by distance
        nearby_stations.sort(key=lambda x: x[1])
        
    except nx.NodeNotFound:
        print(f"Error: Node {current_node} not found in graph")
        pass
    
    return nearby_stations