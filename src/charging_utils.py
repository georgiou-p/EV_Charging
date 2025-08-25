import random
"""
Simplified charging utilities for EV simulation - No anxiety model
"""

def get_station_capacity(charging_stations):
    """
    Get total charging capacity at a node
    
    Args:
        charging_stations: List of EVChargingStation objects
        
    Returns:
        int: Total number of charging points
    """
    return sum(station.get_number_of_points() for station in charging_stations)


def charge_at_station_with_queue_tolerance(env, car_id, target_node, target_station_id, graph, stats, driver, simulation=None, target_soc=0.8):
    """
    Unified charging function with equipment discovery and alternative station checking
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        target_node: Node with the target charging station
        target_station_id: Specific station ID chosen based on predictions
        graph: NetworkX graph with charging stations
        stats: Statistics dictionary to update
        driver: EVDriver object to update battery level
        simulation: Simulation object for queue tracking (optional)
        target_soc: Target state of charge after charging
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Arrived at node {target_node}, checking station {target_station_id}")
    
    stations = graph.nodes[target_node]['charging_stations']
    if not stations:
        print(f"[T={env.now:.1f}] Car {car_id}: No charging stations at node {target_node}!")
        return False
    
    connector_type = driver.get_connector_type()
    
    # Find the target station
    target_station = None
    for station in stations:
        if station.get_station_id() == target_station_id:
            target_station = station
            break
    
    if not target_station:
        print(f"[T={env.now:.1f}] Car {car_id}: Target station {target_station_id} not found")
        return False
    
    # EQUIPMENT DISCOVERY PHASE - Check if target station has working compatible connections
    working_connections = target_station.get_working_connections(connector_type)
    chosen_station = None
    
    if not working_connections:
        print(f"[T={env.now:.1f}] Car {car_id}: Discovered target station {target_station_id} has no working compatible connections!")
        
        # Check for working alternatives at same node
        alternative_working_stations = []
        for station in stations:
            if station != target_station:
                station_working_connections = station.get_working_connections(connector_type)
                if station_working_connections:
                    queue_length = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                    estimated_wait = calculate_queue_wait_time(station, connector_type)
                    if estimated_wait <= 30:  # Only consider stations with queue <= 30 mins
                        alternative_working_stations.append((station, estimated_wait))
        
        if not alternative_working_stations:
            print(f"[T={env.now:.1f}] Car {car_id}: No working alternatives at node with acceptable queue times")
            return False
        else:
            # Switch to best working alternative
            best_alternative = min(alternative_working_stations, key=lambda x: x[1])
            chosen_station = best_alternative[0]
            print(f"[T={env.now:.1f}] Car {car_id}: Switched to working alternative {chosen_station.get_station_id()}")
            stats['alternative_station_switches'] += 1
    else:
        # Target station has working connections
        chosen_station = target_station
        print(f"[T={env.now:.1f}] Car {car_id}: Target station has working compatible connections")
    
    # Check current conditions at chosen station
    current_queue = len(chosen_station.simpy_resource.queue) if hasattr(chosen_station, 'simpy_resource') else 0
    estimated_wait = calculate_queue_wait_time(chosen_station, connector_type) 
    
    print(f"[T={env.now:.1f}] Car {car_id}: Queue at chosen station: {current_queue} cars, estimated wait: {estimated_wait:.1f} min")
    
    # ALTERNATIVE STATION CHECKING: Look for better alternatives at same node if wait is long
    if estimated_wait > 60:  # Check alternatives if wait > 1 hour
        print(f"[T={env.now:.1f}] Car {car_id}: Checking alternatives (long estimated wait: {estimated_wait:.1f} min)")
        
        alternative_options = []
        for station in stations:
            if station != chosen_station:
                station_working_connections = station.get_working_connections(connector_type)
                if station_working_connections:
                    alt_queue_len = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                    alt_wait_time = calculate_queue_wait_time(station, connector_type)
                    alt_power = get_station_max_power(station, connector_type)
                    
                    # Score this alternative
                    score = 1000
                    score -= alt_queue_len * 50  # Penalty for queue length
                    score += alt_power * 2       # Bonus for higher power
                    
                    alternative_options.append((station, score, alt_queue_len, alt_wait_time, alt_power))
                    print(f"[T={env.now:.1f}] Car {car_id}: Alternative {station.get_station_id()}: {alt_queue_len} cars, {alt_wait_time:.1f} min, {alt_power}kW, score={score}")
        
        if alternative_options:
            # Sort by score (highest first)
            alternative_options.sort(key=lambda x: x[1], reverse=True)
            best_alternative = alternative_options[0]
            
            station, score, queue_len, wait_time, power = best_alternative
            
            # Switch if the alternative is significantly better
            if wait_time < estimated_wait * 0.7:  # At least 30% better
                chosen_station = station
                print(f"[T={env.now:.1f}] Car {car_id}: Switched to better alternative {chosen_station.get_station_id()} (wait: {wait_time:.1f} min vs {estimated_wait:.1f} min)")
                stats['alternative_station_switches'] += 1
            else:
                print(f"[T={env.now:.1f}] Car {car_id}: No significantly better alternatives, staying with original choice")
        else:
            print(f"[T={env.now:.1f}] Car {car_id}: No alternative working stations available")
    
    # Proceed with charging at chosen station
    station_id = chosen_station.get_station_id()
    charging_power = get_station_max_power(chosen_station, connector_type)
    
    print(f"[T={env.now:.1f}] Car {car_id}: Using station {station_id} ({charging_power}kW)")
    
    if not hasattr(chosen_station, 'simpy_resource'):
        print(f"[T={env.now:.1f}] Car {car_id}: Station has no SimPy resource!")
        return False
    
    # QUEUING PROCESS
    queue_start_time = env.now
    
    with chosen_station.simpy_resource.request() as request:
        queue_position = len(chosen_station.simpy_resource.queue) + 1
        if queue_position > 1:
            print(f"[T={env.now:.1f}] Car {car_id}: Entering queue position {queue_position}")
        
        # Record initial queue stats
        stats['queue_length'].append(len(chosen_station.simpy_resource.queue))
        
        # Wait for the charging resource
        yield request
        
        # Successfully acquired the charging slot
        queue_end_time = env.now
        queue_time = queue_end_time - queue_start_time
        
        if queue_time > 0:
            stats['queue_times'].append(queue_time)
            stats['total_queue_time'] += queue_time
            print(f"[T={env.now:.1f}] Car {car_id}: Got charging slot after waiting {queue_time:.1f} min")
            
            # QUEUE TRACKING: Record in hourly tracker if simulation object provided
            if simulation and hasattr(simulation, 'record_queue_event'):
                simulation.record_queue_event(queue_time, queue_start_time)
        else:
            print(f"[T={env.now:.1f}] Car {car_id}: Got charging slot immediately")
        
        # CHARGING PROCESS
        current_soc = driver.get_state_of_charge()
        print(f"[T={env.now:.1f}] Car {car_id}: Started charging at {charging_power}kW (battery: {current_soc*100:.0f}%)")
        
        battery_capacity_kwh = driver.get_battery_capacity_kwh()
        if charging_power > 0 and battery_capacity_kwh > 0:
            charging_time = calculate_charging_time(current_soc, target_soc, battery_capacity_kwh, charging_power)
            charging_time = charging_time * 60  # Convert to minutes
        else:
            charging_time = calculate_charging_time(current_soc, target_soc, 75.0, 50)
            charging_time = charging_time * 60
        
        if charging_time > 0:
            print(f"[T={env.now:.1f}] Car {car_id}: Charging for {charging_time:.1f} min")
            yield env.timeout(charging_time)
            
            # Update driver's battery level
            driver.set_battery_level(target_soc)
            print(f"[T={env.now:.1f}] Car {car_id}: Finished charging (battery now: {target_soc*100:.0f}%)")
        else:
            print(f"[T={env.now:.1f}] Car {car_id}: No charging needed")
        
        stats['total_charging_events'] += 1
    
    return True


def calculate_charging_time(current_soc, target_soc, battery_capacity_kwh, charger_power_kw):
    """Calculate realistic charging time"""
    if target_soc <= current_soc or charger_power_kw <= 0:
        return 0.0
    
    soc_difference = target_soc - current_soc
    charging_efficiency = 0.94
    time_hours = (battery_capacity_kwh * soc_difference) / (charger_power_kw * charging_efficiency)
    return time_hours


def calculate_queue_wait_time(station, connector_type):
    """
    Calculate realistic wait time based on charging power and queue length
    """
    if not hasattr(station, 'simpy_resource'):
        return 0.0
    
    queue_length = len(station.simpy_resource.queue)
    if queue_length == 0:
        return 0.0
    
    charging_power = get_station_max_power(station, connector_type)
    if charging_power <= 0:
        charging_power = 50.0  # Default fallback
    
    # Estimate average charging session based on power
    if charging_power >= 150:  # Ultra-rapid charger
        avg_session_minutes = 25
    elif charging_power >= 50:   # Rapid charger  
        avg_session_minutes = 45
    elif charging_power >= 22:   # Fast charger
        avg_session_minutes = 90
    else:  # Slow charger
        avg_session_minutes = 180
    
    
    estimated_session_time = avg_session_minutes * queue_length
    
    # Simple calculation: assume one car per charging point
    total_capacity = station.get_number_of_points()
    if total_capacity <= 0:
        total_capacity = 1
    
    # If queue fits in capacity, minimal wait; otherwise proportional wait
    if queue_length <= total_capacity:
        return estimated_session_time 
    else:
        excess_queue = queue_length - total_capacity
        return (excess_queue / total_capacity) * estimated_session_time


def get_station_max_power(station, connector_type):
    """Get maximum power available at this station for the specified connector type"""
    max_power = 0
    for connection in station.get_connections():
        if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
            if connection.power_kw and connection.power_kw > max_power:
                max_power = connection.power_kw
    return max_power


def has_compatible_connector(charging_stations, connector_type):
    """Check if any charging station has compatible connector"""
    for station in charging_stations:
        for connection in station.get_connections():
            if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                return True
    return False


def setup_charging_resources(env, graph):
    """Create SimPy resources for each individual charging station"""
    import simpy
    
    nodes_with_stations = 0
    total_stations_created = 0
    
    for node in graph.nodes:
        stations = graph.nodes[node]['charging_stations']
        if stations:
            nodes_with_stations += 1
            
            for station in stations:
                station_capacity = station.get_number_of_points()
                if station_capacity <= 0:
                    station_capacity = 1
                station.simpy_resource = simpy.Resource(env, capacity=station_capacity)
                total_stations_created += 1
    
    print(f"Created SimPy resources for {total_stations_created} charging stations across {nodes_with_stations} nodes")
    return nodes_with_stations
