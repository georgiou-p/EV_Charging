import random
"""
Simplified charging utilities for EV simulation
Implements the new simplified charging logic without redundant functions
"""

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


def get_station_max_power(station, connector_type):
    """
    Get maximum power for compatible connections at this station
    
    Args:
        station: EVChargingStation object
        connector_type: Required connector type
        
    Returns:
        float: Maximum power in kW for compatible connections
    """
    max_power = 0
    for connection in station.get_connections():
        if connection.connection_type_id == connector_type or connection.connection_type_id == 0:
            if connection.power_kw:
                max_power = max(max_power, connection.power_kw)
    return max_power

def get_station_capacity(charging_stations):
    """
    Get total charging capacity at a node
    
    Args:
        charging_stations: List of EVChargingStation objects
        
    Returns:
        int: Total number of charging points
    """
    return sum(station.get_number_of_points() for station in charging_stations)

def calculate_charging_time(current_soc, target_soc, battery_capacity_kwh, charger_power_kw):
    """
    Calculate realistic charging time using the provided formula
    Time (h) = Battery × (Target SoC - Initial SoC) / (Charger Power × 0.94)
    
    Args:
        current_soc: Current state of charge (0.0 to 1.0)
        target_soc: Target state of charge (0.0 to 1.0)
        battery_capacity_kwh: Battery capacity in kWh
        charger_power_kw: Charger power in kW
        
    Returns:
        float: Charging time in hours
    """
    if target_soc <= current_soc or charger_power_kw <= 0:
        return 0.0
    
    # Apply the realistic formula
    soc_difference = target_soc - current_soc
    charging_efficiency = 0.94
    
    time_hours = (battery_capacity_kwh * soc_difference) / (charger_power_kw * charging_efficiency)
    
    return time_hours

def charge_at_station_with_queue_tolerance(env, car_id, target_node, target_station_id, graph, stats, driver, simulation =None, target_soc=0.8):
    """
    Try to charge at specific station (chosen based on predictions). 
    Re-evaluate with REAL conditions upon arrival.
    If queue too long, find alternative at same node.
    
    Args:
        env: SimPy environment
        car_id: Car identifier
        target_node: Node with the target charging station
        target_station_id: Specific station ID chosen based on predictions
        graph: NetworkX graph with charging stations
        stats: Statistics dictionary to update
        driver: EVDriver object to update battery level
        max_wait_minutes: Maximum acceptable wait time in minutes
        target_soc: Target state of charge after charging
        
    Yields:
        SimPy events for charging process
        
    Returns:
        bool: True if charging successful, False if no acceptable station found
    """
    print(f"[T={env.now:.1f}] Car {car_id}: Arrived at node {target_node}, checking actual conditions at target station {target_station_id}")
    
    stations = graph.nodes[target_node]['charging_stations']
    if not stations:
        print(f"[T={env.now:.1f}] Car {car_id}: No charging stations at node {target_node}!")
        return False
    
    connector_type = driver.get_connector_type()
    
    # Find the target station which was chosen based on predictions
    target_station = None
    for station in stations:
        if station.get_station_id() == target_station_id:
            target_station = station
            break
    
    if not target_station:
        print(f"[T={env.now:.1f}] Car {car_id}: Target station {target_station_id} not found at node {target_node}")
        return False
    
    # Verify target station is still compatible
    if not has_compatible_connector([target_station], connector_type):
        print(f"[T={env.now:.1f}] Car {car_id}: Target station {target_station_id} no longer compatible with connector {connector_type}")
        return False
    
    # RE-EVALUATE: Check the REAL queue conditions at target when driver arrives
    current_queue = len(target_station.simpy_resource.queue) if hasattr(target_station, 'simpy_resource') else 0
    estimated_wait_minutes = calculate_queue_wait_time(target_station, connector_type)
    
    print(f"[T={env.now:.1f}] Car {car_id}: REAL conditions at target station {target_station_id}: {current_queue} cars in queue (est. {estimated_wait_minutes} min wait)")
    
    chosen_station = target_station

    max_wait_minutes = 20  # Fixed value for queue tolerance
    
    # If REAL queue is worse than acceptable, RE-EVALUATE alternatives at this node
    if estimated_wait_minutes > max_wait_minutes:
        print(f"[T={env.now:.1f}] Car {car_id}: REAL queue too long ({estimated_wait_minutes} > {max_wait_minutes} min)")
        print(f"[T={env.now:.1f}] Car {car_id}: RE-EVALUATING alternatives at node {target_node} with REAL conditions")
        
        # Score all other stations at this node based on REAL current conditions
        alternative_options = []
        for station in stations:
            if (station != target_station and 
                has_compatible_connector([station], connector_type)):
                
                real_queue_len = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                real_wait_time = calculate_queue_wait_time(station, connector_type)
                
                if real_wait_time <= max_wait_minutes:
                    # Score this alternative based on REAL conditions
                    score = 1000
                    score -= real_queue_len * 100  # Heavy penalty for REAL queue
                    
                    max_power = get_station_max_power(station, connector_type)
                    if max_power >= 150:  # Ultra-rapid
                        power_bonus = 200
                    elif max_power >= 50:  # Rapid
                        power_bonus = 100
                    elif max_power >= 22:  # Fast
                        power_bonus = 50
                    else:  # Standard/slow
                        power_bonus = 20
                    
                    score += power_bonus
                    
                    alternative_options.append((station, score, real_queue_len, real_wait_time, max_power))
                    print(f"[T={env.now:.1f}] Car {car_id}: Alternative {station.get_station_id()}: {real_queue_len} cars, {real_wait_time} min, {max_power}kW, score={score}")
        
        if alternative_options:
            # Use weighted random selection
            stations_for_random = [opt[0] for opt in alternative_options]  # Get station objects
            weights = [opt[1] for opt in alternative_options]  # Get scores as weights
    
            # Normalize weights to probabilities
            total_weight = sum(weights)
            probabilities = [w/total_weight for w in weights]

            
            chosen_station = random.choices(stations_for_random, weights=probabilities)[0]

            # Find the matching option for logging
            chosen_option = next(opt for opt in alternative_options if opt[0] == chosen_station)
            _, best_score, queue_len, wait_time, max_power = chosen_option
            
            print(f"[T={env.now:.1f}] Car {car_id}: RANDOMLY selected {chosen_station.get_station_id()} (weighted by score {best_score:.0f}): {queue_len} cars ({wait_time} min wait, {max_power}kW)")
        else:
            #No acceptable alternatives, find station with smallest wait time (including target)
            print(f"[T={env.now:.1f}] Car {car_id}: No acceptable alternatives at node {target_node}")
            print(f"[T={env.now:.1f}] Car {car_id}: FALLBACK - selecting station with smallest wait time regardless of tolerance")
            
            all_compatible_stations = []
            # Include target station in fallback consideration
            if has_compatible_connector([target_station], connector_type):
                target_queue_len = len(target_station.simpy_resource.queue) if hasattr(target_station, 'simpy_resource') else 0
                target_wait_time = calculate_queue_wait_time(target_station, connector_type)
                target_power = get_station_max_power(target_station, connector_type)
                all_compatible_stations.append((target_station, target_queue_len, target_wait_time, target_power))
            
            # Include all other compatible stations
            for station in stations:
                if (station != target_station and 
                    has_compatible_connector([station], connector_type)):
                    real_queue_len = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                    real_wait_time = calculate_queue_wait_time(target_station, connector_type)
                    max_power = get_station_max_power(station, connector_type)
                    
                    all_compatible_stations.append((station, real_queue_len, real_wait_time, max_power))
            
            # Log all options being considered
            for station, queue_len, wait_time, max_power in all_compatible_stations:
                print(f"[T={env.now:.1f}] Car {car_id}: Option {station.get_station_id()}: {queue_len} cars, {wait_time} min, {max_power}kW")
            
            if all_compatible_stations:
                # Sort by wait time (smallest first), then by power (highest first) as tiebreaker
                all_compatible_stations.sort(key=lambda x: (x[2], -x[3]))  # x[2] is wait_time, x[3] is max_power
                chosen_station, queue_len, wait_time, max_power = all_compatible_stations[0]
                print(f"[T={env.now:.1f}] Car {car_id}: FALLBACK selection: {chosen_station.get_station_id()} with smallest wait time: {wait_time} min ({queue_len} cars, {max_power}kW)")
            else:
                print(f"[T={env.now:.1f}] Car {car_id}: CRITICAL ERROR - No compatible stations at all at node {target_node}")
                return False
    else:
        print(f"[T={env.now:.1f}] Car {car_id}: Target station {target_station_id} is acceptable with REAL conditions")
    
    # Proceed with charging at chosen station
    station_id = chosen_station.get_station_id()
    charging_power = get_station_max_power(chosen_station, connector_type)
    
    print(f"[T={env.now:.1f}] Car {car_id}: Using station {station_id} ({charging_power}kW)")
    
    if not hasattr(chosen_station, 'simpy_resource'):
        print(f"[T={env.now:.1f}] Car {car_id}: Station {station_id} has no SimPy resource!")
        return False
    
    queue_start_time = env.now
    # Request charging point at the chosen station
    with chosen_station.simpy_resource.request() as request:
        # Wait for an available charging point
        queue_position = len(chosen_station.simpy_resource.queue) + 1
        if queue_position > 1:
            print(f"[T={env.now:.1f}] Car {car_id}: Waiting in queue (position {queue_position}) at station {station_id}")
        
        #QUEUE STATS
        current_queue_length = len(chosen_station.simpy_resource.queue)
        stats['queue_length'].append(current_queue_length)
        yield request
        queue_end_time = env.now
        current_queue_length = len(chosen_station.simpy_resource.queue)
        queue_time = queue_end_time - queue_start_time
        if queue_time > 0:
            #Queue penalty
            driver.add_queue_penalty(queue_time * 2)
            stats['queue_times'].append(queue_time)
            stats['total_queue_time'] += queue_time

            if simulation is not None:
                simulation._record_queue_time(queue_time)
                    
        # Start charging
        current_soc = driver.get_state_of_charge()
        print(f"[T={env.now:.1f}] Car {car_id}: Started charging at station {station_id} (current battery: {current_soc*100:.0f}%)")
        
        battery_capacity_kwh = driver.get_battery_capacity_kwh()
        if charging_power > 0 and battery_capacity_kwh >0:
            charging_time = calculate_charging_time(current_soc, target_soc, battery_capacity_kwh, charging_power)
            charging_time = charging_time * 60 #converting to minutes
        else:
            charging_time = calculate_charging_time(current_soc, target_soc, battery_capacity_kwh or 75.0, 50)
        
        expected_fast_power = 50.0  # kW - what we consider "normal" charging speed

        if charging_power < expected_fast_power:
            expected_time = calculate_charging_time(current_soc, target_soc, battery_capacity_kwh, expected_fast_power)
            actual_time = charging_time
            extra_time_hours = actual_time - expected_time
    
            if extra_time_hours > 0:
                extra_time_minutes = extra_time_hours * 60  # Convert to minutes
                driver.add_slow_charger_penalty(extra_time_minutes)
                print(f"[T={env.now:.1f}] Car {car_id}: Slow charger penalty: {extra_time_minutes:.1f} extra minutes for {charging_power}kW vs {expected_fast_power}kW")
              
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
        
    
    return True

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

def calculate_queue_wait_time(station, connector_type):
    """
    Calculate realistic wait time based on charging power and typical usage patterns
    
    Args:
        station: EVChargingStation object
        connector_type: Connector type to get appropriate charging power
        
    Returns:
        float: Estimated wait time in minutes
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
        avg_session_minutes = 25  # Quick top-up session
    elif charging_power >= 50:   # Rapid charger  
        avg_session_minutes = 45  # Medium session
    elif charging_power >= 22:   # Fast charger
        avg_session_minutes = 90  # Longer session
    else:  # Slow charger
        avg_session_minutes = 180  # Very long session
    
    charging_factor = 0.8  # Assume 60% of full charge on average
    estimated_session_time = avg_session_minutes * charging_factor
    
    total_wait_time = queue_length * estimated_session_time
    
    return total_wait_time

