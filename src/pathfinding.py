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

def get_station_queue_and_speed_metrics(stations, connector_type):
    """
    Calculate queue and speed metrics for compatible stations at a node
    
    Args:
        stations: List of EVChargingStation objects
        connector_type: Required connector type
        
    Returns:
        tuple: (total_queue_length, max_compatible_power, compatible_points, avg_wait_time)
    """
    total_queue_length = 0
    max_compatible_power = 0
    compatible_points = 0
    
    for station in stations:
        # Check if station has compatible connectors
        station_has_compatible = False
        station_max_power = 0
        station_compatible_points = 0
        
        for connection in station.get_connections():
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                station_has_compatible = True
                if connection.power_kw:
                    station_max_power = max(station_max_power, connection.power_kw)
                station_compatible_points += connection.quantity
        
        if station_has_compatible:
            # Add queue length from this station
            if hasattr(station, 'simpy_resource'):
                total_queue_length += len(station.simpy_resource.queue)
            
            max_compatible_power = max(max_compatible_power, station_max_power)
            compatible_points += station_compatible_points
    
    # Calculate average wait time estimate (cars in queue / charging points)
    avg_wait_time = total_queue_length / max(compatible_points, 1)
    
    return total_queue_length, max_compatible_power, compatible_points, avg_wait_time

def find_nearest_charging_station(graph, current_node, planned_route, max_range, connector_type):
    """
    Find the best charging station based on queue times and charging speed for compatible connectors
    
    Args:
        graph: NetworkX graph with charging stations and weighted edges
        current_node: Current node position
        planned_route: List of nodes representing the planned route
        max_range: Maximum distance to search for stations
        connector_type: Required connector type for the vehicle
        
    Returns:
        int or None: Node ID of best charging station, or None if none found
    """
    print(f"    Looking for charging stations (connector type {connector_type}) optimized for route, queue times and charging speed")
    
    # Get all stations within range using weighted distances
    compatible_stations_data = []
    
    try:
        distances = nx.single_source_dijkstra_path_length(graph, current_node, cutoff=max_range, weight='weight')
        
        for node, distance in distances.items():
            if node != current_node and 'charging_stations' in graph.nodes[node]:
                stations = graph.nodes[node]['charging_stations']
                # Filter for compatibility
                if stations and has_compatible_connector(stations, connector_type):
                    # Get queue and speed metrics for this node
                    queue_length, max_power, compatible_points, avg_wait = get_station_queue_and_speed_metrics(stations, connector_type)
                    compatible_stations_data.append((node, distance, queue_length, max_power, compatible_points, avg_wait))
        
    except nx.NodeNotFound:
        pass
    
    if not compatible_stations_data:
        print(f"    No compatible charging stations found within {max_range}km!")
        return None
    
    print(f"    Found {len(compatible_stations_data)} compatible charging stations within range")
    
    # Score stations based on queue times and charging speed
    scored_stations = []
    
    for node, distance, queue_length, max_power, compatible_points, avg_wait in compatible_stations_data:
        
        # Base score starts high
        score = 1000
        
        # Major penalty for queue wait time (most important factor)
        queue_penalty = avg_wait * 100  # Heavy penalty for waiting
        score -= queue_penalty
        
        # Major bonus for charging speed (second most important)
        if max_power >= 150:  # Ultra-rapid charging (150kW+)
            speed_bonus = 200
        elif max_power >= 50:  # Rapid charging (50-149kW)
            speed_bonus = 100
        elif max_power >= 22:  # Fast AC charging (22-49kW)
            speed_bonus = 50
        elif max_power >= 7:   # Standard charging (7-21kW)
            speed_bonus = 20
        else:  # Slow charging (<7kW)
            speed_bonus = -50  # Penalty for very slow charging
        
        score += speed_bonus
        
        # Moderate bonus for having multiple compatible points (redundancy)
        capacity_bonus = min(compatible_points * 10, 50)  # Cap at 50 points
        score += capacity_bonus
        
        # Small distance penalty (least important factor now)
        distance_penalty = distance * 0.5  # Much reduced compared to original
        score -= distance_penalty
        
        # Route position factor (significant influence)
        route_factor = 1.0  # Multiplier for final score
        route_description = ""
        
        if node in planned_route:
            route_position = planned_route.index(node)
            current_position = planned_route.index(current_node) if current_node in planned_route else 0
            
            if route_position > current_position:
                # Station is ahead on our planned route - significant bonus
                route_factor = 1.5  # 50% bonus for on-route stations
                route_description = "ON_ROUTE"
            else:
                # Station is behind us on route - moderate bonus  
                route_factor = 1.2  # 20% bonus for stations we've passed
                route_description = "ROUTE_BEHIND"
        else:
            # Station requires detour - calculate detour penalty
            # Find closest point on route to return to after charging
            min_return_distance = float('inf')
            for route_node in planned_route:
                try:
                    return_distance = nx.shortest_path_length(graph, node, route_node, weight='weight')
                    min_return_distance = min(min_return_distance, return_distance)
                except:
                    continue
            
            if min_return_distance != float('inf'):
                total_detour = distance + min_return_distance
                # Penalty based on detour severity
                if total_detour <= distance * 1.5:  # Less than 50% extra distance
                    route_factor = 0.9  # Small penalty for minor detours
                    route_description = "MINOR_DETOUR"
                elif total_detour <= distance * 2.0:  # Less than double distance
                    route_factor = 0.7  # Moderate penalty for medium detours
                    route_description = "MEDIUM_DETOUR"
                else:
                    route_factor = 0.5  # Heavy penalty for major detours
                    route_description = "MAJOR_DETOUR"
            else:
                route_factor = 0.3  # Severe penalty if can't return to route
                route_description = "NO_RETURN"
        
        # Apply route factor to final score
        score = score * route_factor
        
        final_score = max(score, 0)  # Ensure non-negative scores
        
        # Create comprehensive priority description
        if queue_length == 0:
            queue_status = "NO_QUEUE"
        elif avg_wait <= 0.5:
            queue_status = "SHORT_QUEUE"
        elif avg_wait <= 1.0:
            queue_status = "MEDIUM_QUEUE"
        else:
            queue_status = "LONG_QUEUE"
        
        if max_power >= 150:
            speed_status = "ULTRA_RAPID"
        elif max_power >= 50:
            speed_status = "RAPID"
        elif max_power >= 22:
            speed_status = "FAST"
        else:
            speed_status = "SLOW"
        
        # Combine route position with charging characteristics
        priority = f"{route_description}_{queue_status}_{speed_status}"
        
        scored_stations.append((node, distance, final_score, priority, queue_length, max_power, avg_wait, compatible_points, route_factor))
    
    # Sort by score (highest first)
    scored_stations.sort(key=lambda x: x[2], reverse=True)
    
    # Show top options
    print("    Top charging station options (queue + speed + route optimized):")
    for i, (station_node, distance, score, priority, queue_len, max_pwr, avg_wait, comp_pts, route_mult) in enumerate(scored_stations[:5]):
        print(f"    {i+1}. Node {station_node}: score={score:.0f}, {priority}")
        print(f"       Distance: {distance:.1f}km, Queue: {queue_len} cars, "
              f"Wait: {avg_wait:.1f}x, Power: {max_pwr}kW, Points: {comp_pts}, Route: {route_mult:.1f}x")
    
    # Get best node and select best station at that node
    best_node_data = scored_stations[0]
    best_node = best_node_data[0]
    best_priority = best_node_data[3]
    
    print(f"    Selected best node: {best_node} ({best_priority})")
    
    # Get stations at the best node and select the best one
    stations_at_best_node = graph.nodes[best_node]['charging_stations']
    
    # Find best station at the selected node using queue and speed optimization
    from charging_utils import choose_charging_station_by_queue_and_speed
    best_station = choose_charging_station_by_queue_and_speed(stations_at_best_node, connector_type)
    
    if best_station:
        best_station_id = best_station.get_station_id()
        print(f"    Selected best station at node: {best_station_id}")
        return best_node, best_station_id
    else:
        print(f"    No compatible station found at best node {best_node}")
        return None, None


def travel_to_charging_station(env, car_id, current_node, charging_node, graph, driver, travel_time_per_unit=1.0):
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

        consumption_per_km = 1.0 / driver.get_battery_capacity()
        battery_consumption = total_distance * consumption_per_km
        driver.consume_battery(battery_consumption)

        print(f"[T={env.now:.1f}] Car {car_id}: Arrived at node {charging_node} (traveled distance: {total_distance:.2f}km. SoC: {driver.get_state_of_charge():.2f})")
    else:
        print(f"[T={env.now:.1f}] Car {car_id}: Already at node {charging_node}")
    
    return True

def travel_to_next_node(env, car_id, driver, graph, travel_time_per_km=0.01):
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
        pass
    
    return nearby_stations