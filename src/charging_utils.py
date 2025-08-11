"""
Queue and speed optimized charging utilities for EV simulation
"""
import random

def get_charging_resource(graph, node):
    """
    Get the SimPy resource for charging at a specific node
    This function is now deprecated since each station has its own resource
    
    Args:
        graph: NetworkX graph with charging stations
        node: Node ID to get charging resource for
        
    Returns:
        simpy.Resource or None: Charging resource if available (deprecated)
    """
    # This function is kept for backward compatibility but is no longer used
    # Individual stations now have their own simpy_resource attribute
    if 'simpy_resource' in graph.nodes[node]:
        return graph.nodes[node]['simpy_resource']
    return None

def calculate_charging_time(current_soc, target_soc=1.0, base_charging_time=10.0):
    """
    Calculate time needed to charge from current to target state of charge
    
    Args:
        current_soc: Current state of charge (0.0 to 1.0)
        target_soc: Target state of charge (0.0 to 1.0)
        base_charging_time: Time for full charge (0% to 100%)
        
    Returns:
        float: Charging time needed
    """
    if target_soc <= current_soc:
        return 0.0
    
    charge_needed = target_soc - current_soc
    return base_charging_time * charge_needed

def has_compatible_connector(charging_stations, connector_type):
    """
    Check if any charging station has compatible connector
    
    Args:
        charging_stations: List of EVChargingStation objects
        connector_type: Required connector type
        
    Returns:
        bool: True if compatible connector found
    """
    for station in charging_stations:
        for connection in station.get_connections():
            # Handle null/None connector types as universal compatibility
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                return True
    return False

def get_compatible_stations(charging_stations, connector_type):
    """
    Filter stations that have compatible connectors
    
    Args:
        charging_stations: List of EVChargingStation objects
        connector_type: Required connector type
        
    Returns:
        list: List of compatible EVChargingStation objects
    """
    compatible_stations = []
    for station in charging_stations:
        for connection in station.get_connections():
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                compatible_stations.append(station)
                break  # Found compatible connector, no need to check other connections
    return compatible_stations

def get_station_capacity(charging_stations):
    """
    Get total charging capacity at a node
    
    Args:
        charging_stations: List of EVChargingStation objects
        
    Returns:
        int: Total number of charging points
    """
    return sum(station.get_number_of_points() for station in charging_stations)

def choose_charging_station_by_queue_and_speed(charging_stations, connector_type):
    """
    Choose charging station optimized for queue length and charging speed
    
    Args:
        charging_stations: List of EVChargingStation objects at the node
        connector_type: Required connector type for the vehicle
        
    Returns:
        EVChargingStation: Selected charging station optimized for queue and speed
    """
    if not charging_stations:
        return None
    
    # Filter for connector compatibility
    compatible_stations = get_compatible_stations(charging_stations, connector_type)
    
    if not compatible_stations:
        print(f"    No stations with compatible connector type {connector_type} found!")
        return None
    
    print(f"    Found {len(compatible_stations)} compatible stations out of {len(charging_stations)} total")
    
    # Score each compatible station
    station_scores = []
    
    for station in compatible_stations:
        score = 1000  # Base score
        
        # Queue length penalty (most important factor)
        queue_length = 0
        if hasattr(station, 'simpy_resource'):
            queue_length = len(station.simpy_resource.queue)
        
        queue_penalty = queue_length * 100  # Heavy penalty for each car in queue
        score -= queue_penalty
        
        # Charging speed bonus (second most important)
        max_compatible_power = 0
        for connection in station.get_connections():
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                if connection.power_kw:
                    max_compatible_power = max(max_compatible_power, connection.power_kw)
        
        if max_compatible_power >= 150:  # Ultra-rapid charging
            speed_bonus = 200
        elif max_compatible_power >= 50:  # Rapid charging
            speed_bonus = 100
        elif max_compatible_power >= 22:  # Fast AC charging
            speed_bonus = 50
        elif max_compatible_power >= 7:   # Standard charging
            speed_bonus = 20
        else:  # Slow charging
            speed_bonus = -50
        
        score += speed_bonus
        
        # Capacity bonus (minor factor)
        compatible_points = 0
        for connection in station.get_connections():
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                compatible_points += connection.quantity
        
        capacity_bonus = min(compatible_points * 5, 25)  # Small bonus, capped
        score += capacity_bonus
        
        station_scores.append((station, score, queue_length, max_compatible_power, compatible_points))
    
    # Sort by score (highest first)
    station_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Log the selection decision
    best_station, best_score, queue_len, max_power, comp_points = station_scores[0]
    print(f"    Selected station {best_station.get_station_id()}: score={best_score:.0f}, "
          f"queue={queue_len}, power={max_power}kW, points={comp_points}")
    
    return best_station

def choose_charging_station(charging_stations, connector_type, selection_method="queue_and_speed"):
    """
    Choose which charging station to use - now defaults to queue and speed optimization
    
    Args:
        charging_stations: List of EVChargingStation objects at the node
        connector_type: Required connector type for the vehicle
        selection_method: Selection strategy (defaults to "queue_and_speed")
        
    Returns:
        EVChargingStation: Selected charging station that is compatible
    """
    if selection_method == "queue_and_speed":
        return choose_charging_station_by_queue_and_speed(charging_stations, connector_type)
    
    # Fallback to legacy methods for compatibility
    if not charging_stations:
        return None
    
    # Filter for connector compatibility
    compatible_stations = get_compatible_stations(charging_stations, connector_type)
    
    if not compatible_stations:
        print(f"    No stations with compatible connector type {connector_type} found!")
        return None
    
    print(f"    Found {len(compatible_stations)} compatible stations out of {len(charging_stations)} total")
    
    # Apply legacy selection strategy to compatible stations only
    if selection_method == "random":
        return random.choice(compatible_stations)
    
    elif selection_method == "shortest_queue":
        # Choose compatible station with shortest queue
        return min(compatible_stations, key=lambda s: len(s.simpy_resource.queue) if hasattr(s, 'simpy_resource') else 0)
    
    elif selection_method == "highest_capacity":
        # Choose compatible station with most charging points
        return max(compatible_stations, key=lambda s: s.get_number_of_points())
    
    else:
        # Default to queue and speed optimization
        return choose_charging_station_by_queue_and_speed(charging_stations, connector_type)

def charge_at_station(env, car_id, node, graph, stats, driver, target_soc=1.0, selection_method="queue_and_speed", specific_station_id=None):
    """
    Handle charging at a station using queue and speed optimized selection
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        node: Node with charging station
        graph: NetworkX graph with charging stations
        stats: Statistics dictionary to update
        driver: EVDriver object to update battery level
        target_soc: Target state of charge after charging
        selection_method: Selection strategy (defaults to "queue_and_speed")
        specific_station_id: Pre-selected station ID to use (from pathfinding)
        
    Yields:
        SimPy events for charging process
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Need to charge at node {node}")
    
    # Get charging stations at this node
    stations = graph.nodes[node]['charging_stations']
    if not stations:
        print(f"[T={env.now:.1f}] Car {car_id}: No charging stations at node {node}!")
        return
    
    # Get driver's connector type
    connector_type = driver.get_connector_type()
    
    # Choose which specific station to use
    chosen_station = None
    
    # If specific station was pre-selected in pathfinding, try to use it
    if specific_station_id:
        print(f"[T={env.now:.1f}] Car {car_id}: Attempting to use pre-selected station {specific_station_id}")
        
        for station in stations:
            if station.get_station_id() == specific_station_id:
                # Verify it's still compatible (safety check)
                if has_compatible_connector([station], connector_type):
                    chosen_station = station
                    print(f"[T={env.now:.1f}] Car {car_id}: Successfully using pre-selected station {specific_station_id}")
                    break
                else:
                    print(f"[T={env.now:.1f}] Car {car_id}: Pre-selected station {specific_station_id} no longer compatible!")
                    break
        
        if not chosen_station:
            print(f"[T={env.now:.1f}] Car {car_id}: Pre-selected station {specific_station_id} not found or unavailable, re-selecting...")
    
    # Fallback to normal selection if no specific station or it wasn't available
    if not chosen_station:
        chosen_station = choose_charging_station(stations, connector_type, selection_method)
        if chosen_station:
            print(f"[T={env.now:.1f}] Car {car_id}: Selected alternative station {chosen_station.get_station_id()}")
    
    if not chosen_station:
        print(f"[T={env.now:.1f}] Car {car_id}: No compatible charging station at node {node} for connector type {connector_type}!")
        return
    
    station_id = chosen_station.get_station_id()
    station_capacity = chosen_station.get_number_of_points()
    
    # Check if this station has a SimPy resource
    if not hasattr(chosen_station, 'simpy_resource'):
        print(f"[T={env.now:.1f}] Car {car_id}: Station {station_id} has no SimPy resource!")
        return
    
    # Get charging speed for this connector type
    charging_power = 0
    for connection in chosen_station.get_connections():
        if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
            if connection.power_kw:
                charging_power = max(charging_power, connection.power_kw)
    
    print(f"[T={env.now:.1f}] Car {car_id}: Chose station {station_id} (capacity: {station_capacity} points, "
          f"connector: {connector_type}, power: {charging_power}kW)")
    
    # Get current battery level from driver
    current_soc = driver.get_state_of_charge()
    print(f"[T={env.now:.1f}] Car {car_id}: Current battery: {current_soc:.2f} ({current_soc*100:.0f}%), target: {target_soc:.2f} ({target_soc*100:.0f}%)")
    
    # Request charging point at the specific chosen station
    with chosen_station.simpy_resource.request() as request:
        # Check queue status for this specific station
        queue_length = len(chosen_station.simpy_resource.queue)
        if queue_length > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Waiting in queue (position {queue_length + 1}) at station {station_id}")
        
        # Wait for an available charging point at this specific station
        yield request
        
        # Start charging (we got a charging point at this station)
        print(f"[T={env.now:.1f}] Car {car_id}: Started charging at station {station_id} (node {node})")
        
        # Calculate charging time based on power and current/target SoC
        charge_needed = target_soc - current_soc
        if charging_power > 0:
            # Adjust charging time based on power (higher power = faster charging)
            base_time = 10.0  # Base time for full charge at standard power
            power_factor = 50.0 / max(charging_power, 1.0)  # Normalize to 50kW standard
            charging_time = base_time * charge_needed * power_factor
        else:
            # Fallback to standard calculation
            charging_time = calculate_charging_time(current_soc, target_soc)
        
        if charging_time > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Charging for {charging_time:.1f} time units at {charging_power}kW "
                  f"(from {current_soc*100:.0f}% to {target_soc*100:.0f}%)")
            yield env.timeout(charging_time)
            
            # Update driver's battery level
            driver.set_battery_level(target_soc)
            print(f"[T={env.now:.1f}] Car {car_id}: Finished charging at station {station_id} (battery now: {target_soc*100:.0f}%)")
        else:
            print(f"[T={env.now:.1f}] Car {car_id}: No charging needed at station {station_id} (already at target level)")
        
        stats['total_charging_events'] += 1
        
        # Resource is automatically released when exiting the 'with' block

def setup_charging_resources(env, graph):
    """
    Create SimPy resources for each individual charging station
    
    Args:
        env: SimPy environment
        graph: NetworkX graph with charging stations
        
    Returns:
        int: Number of nodes with charging resources created
    """
    import simpy
    
    nodes_with_stations = 0
    total_stations_created = 0
    
    for node in graph.nodes:
        stations = graph.nodes[node]['charging_stations']
        if stations:
            nodes_with_stations += 1
            
            # Create individual SimPy resource for each charging station
            for station in stations:
                station_capacity = station.get_number_of_points()
                if station_capacity <= 0:
                    station_capacity = 1
                # Create separate resource for each station
                station.simpy_resource = simpy.Resource(env, capacity=station_capacity)
                total_stations_created += 1
            
    
    print(f"Created individual SimPy resources for {total_stations_created} charging stations across {nodes_with_stations} nodes")
    return nodes_with_stations