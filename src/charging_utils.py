"""
Charging-related utilities for EV simulation
"""

def get_charging_resource(graph, node):
    """
    Get the SimPy resource for charging at a specific node
    
    Args:
        graph: NetworkX graph with charging stations
        node: Node ID to get charging resource for
        
    Returns:
        simpy.Resource or None: Charging resource if available
    """
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
            if connection.current_type_id == connector_type or connection.current_type_id is None:
                return True
    return False

def get_station_capacity(charging_stations):
    """
    Get total charging capacity at a node
    
    Args:
        charging_stations: List of EVChargingStation objects
        
    Returns:
        int: Total number of charging points
    """
    return sum(station.get_number_of_points() for station in charging_stations)

def charge_at_station(env, car_id, node, graph, stats, driver, target_soc=1.0):
    """
    Handle charging at a station using SimPy's automatic queuing
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        node: Node with charging station
        graph: NetworkX graph with charging stations
        stats: Statistics dictionary to update
        driver: EVDriver object to update battery level
        target_soc: Target state of charge after charging
        
    Yields:
        SimPy events for charging process
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Need to charge at node {node}")
    
    # Check if this node has charging stations
    charging_resource = get_charging_resource(graph, node)
    if not charging_resource:
        print(f"[T={env.now:.1f}] Car {car_id}: No charging resource at node {node}!")
        return
    
    # Get charging stations info for logging
    stations = graph.nodes[node]['charging_stations']
    total_capacity = get_station_capacity(stations)
    
    # Get current battery level from driver
    current_soc = driver.get_state_of_charge()
    print(f"[T={env.now:.1f}] Car {car_id}: Current battery: {current_soc:.2f} ({current_soc*100:.0f}%), target: {target_soc:.2f} ({target_soc*100:.0f}%)")
    
    # Request charging point - SimPy automatically handles the queue
    with charging_resource.request() as request:
        # Check queue status
        queue_length = len(charging_resource.queue)
        if queue_length > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Waiting in queue (position {queue_length + 1}) at node {node}")
        
        # Wait for an available charging point (automatic queuing)
        yield request
        
        # Start charging (we got a charging point)
        print(f"[T={env.now:.1f}] Car {car_id}: Started charging at node {node} (capacity: {total_capacity})")
        
        # Calculate charging time based on actual current and target SoC
        charging_time = calculate_charging_time(current_soc, target_soc)
        
        if charging_time > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Charging for {charging_time:.1f} time units (from {current_soc*100:.0f}% to {target_soc*100:.0f}%)")
            yield env.timeout(charging_time)
            
            # Update driver's battery level
            driver.set_battery_level(target_soc)
            print(f"[T={env.now:.1f}] Car {car_id}: Finished charging at node {node} (battery now: {target_soc*100:.0f}%)")
        else:
            print(f"[T={env.now:.1f}] Car {car_id}: No charging needed (already at target level)")
        
        stats['total_charging_events'] += 1
        
        # Resource is automatically released when exiting the 'with' block

def setup_charging_resources(env, graph):
    """
    Create SimPy resources for all nodes with charging stations
    
    Args:
        env: SimPy environment
        graph: NetworkX graph with charging stations
        
    Returns:
        int: Number of nodes with charging resources created
    """
    import simpy
    
    nodes_with_stations = 0
    
    for node in graph.nodes:
        stations = graph.nodes[node]['charging_stations']
        if stations:
            # Calculate total capacity from existing stations
            total_capacity = get_station_capacity(stations)
            
            # Create SimPy resource - SimPy automatically handles the queue
            graph.nodes[node]['simpy_resource'] = simpy.Resource(env, capacity=total_capacity)
            nodes_with_stations += 1
    
    return nodes_with_stations