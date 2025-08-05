"""
Modified charging-related utilities for EV simulation with random station assignment
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

def choose_charging_station(charging_stations, selection_method="random"):
    """
    Choose which charging station to use based on different strategies
    
    Args:
        charging_stations: List of EVChargingStation objects at the node
        selection_method: "random", "shortest_queue", or "highest_capacity"
        
    Returns:
        EVChargingStation: Selected charging station
    """
    if not charging_stations:
        return None
    
    if selection_method == "random":
        return random.choice(charging_stations)
    
    elif selection_method == "shortest_queue":
        # Choose station with shortest queue
        return min(charging_stations, key=lambda s: len(s.simpy_resource.queue) if hasattr(s, 'simpy_resource') else 0)
    
    elif selection_method == "highest_capacity":
        # Choose station with most charging points
        return max(charging_stations, key=lambda s: s.get_number_of_points())
    
    else:
        # Default to random
        return random.choice(charging_stations)

def charge_at_station(env, car_id, node, graph, stats, driver, target_soc=1.0, selection_method="random"):
    """
    Handle charging at a station using individual station queues
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        node: Node with charging station
        graph: NetworkX graph with charging stations
        stats: Statistics dictionary to update
        driver: EVDriver object to update battery level
        target_soc: Target state of charge after charging
        selection_method: How to choose station ("random", "shortest_queue", "highest_capacity")
        
    Yields:
        SimPy events for charging process
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Need to charge at node {node}")
    
    # Get charging stations at this node
    stations = graph.nodes[node]['charging_stations']
    if not stations:
        print(f"[T={env.now:.1f}] Car {car_id}: No charging stations at node {node}!")
        return
    
    # Choose which specific station to use
    chosen_station = choose_charging_station(stations, selection_method)
    if not chosen_station:
        print(f"[T={env.now:.1f}] Car {car_id}: Could not choose a charging station at node {node}!")
        return
    
    station_id = chosen_station.get_station_id()
    station_capacity = chosen_station.get_number_of_points()
    
    # Check if this station has a SimPy resource
    if not hasattr(chosen_station, 'simpy_resource'):
        print(f"[T={env.now:.1f}] Car {car_id}: Station {station_id} has no SimPy resource!")
        return
    
    print(f"[T={env.now:.1f}] Car {car_id}: Chose station {station_id} (capacity: {station_capacity} points)")
    
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
        
        # Calculate charging time based on actual current and target SoC
        charging_time = calculate_charging_time(current_soc, target_soc)
        
        if charging_time > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Charging for {charging_time:.1f} time units at station {station_id} (from {current_soc*100:.0f}% to {target_soc*100:.0f}%)")
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
                # Create separate resource for each station
                station.simpy_resource = simpy.Resource(env, capacity=station_capacity)
                total_stations_created += 1
            
    
    print(f"Created individual SimPy resources for {total_stations_created} charging stations across {nodes_with_stations} nodes")
    return nodes_with_stations