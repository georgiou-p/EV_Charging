import simpy
import random
import cProfile
import networkx as nx
import math
from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import (
    find_nearest_charging_station, 
    travel_to_charging_station,
    travel_to_next_node
)
from charging_utils import charge_at_station_with_queue_tolerance, setup_charging_resources

class SimpleEVSimulation:
    def __init__(self, graph, simulation_time=None):
        """
        Simple SimPy simulation using SimPy's built-in queuing
        
        Args:
            graph: NetworkX graph with charging stations already assigned
            simulation_time: How long to run simulation (None = run until all cars finish)
        """
        self.env = simpy.Environment()
        self.graph = graph
        self.simulation_time = simulation_time
        
        # Create SimPy resources from existing charging stations
        self._setup_charging_resources()
        
        # Statistics
        self.stats = {
            'cars_spawned': 0,
            'cars_completed': 0,
            'cars_stranded': 0,
            'total_charging_events': 0,
            'total_travel_time': 0,
            'connector_incompatibility_failures': 0,
            'queue_times': [],
            'total_queue_time': 0,
            'queue_length': [],
            'anxiety_levels': []  
        }
        
        # Hourly tracking (T1-T24 mapped to indices 0-23)
        self._anx_count = [0] * 24
        self._anx_mean = [0.0] * 24
        self._anx_M2 = [0.0] * 24
        self._driver_count = [0] * 24
        self._driver_count_samples = [0] * 24
        self._queue_count = [0] * 24
        self._queue_mean = [0.0] * 24
        self._queue_M2 = [0.0] * 24
    
    def _hour_bin(self) -> int:
        """Convert env.now (minutes) to hour bin 0..23 (T1 maps to 0, T24 maps to 23)"""
        return int(self.env.now // 60) % 24
    
    def _update_anxiety_stats(self, hour_idx: int, value: float) -> None:
        """Welford online algorithm update for anxiety statistics per hour bin"""
        n = self._anx_count[hour_idx] + 1
        delta = value - self._anx_mean[hour_idx]
        self._anx_mean[hour_idx] += delta / n
        delta2 = value - self._anx_mean[hour_idx]
        self._anx_M2[hour_idx] += delta * delta2
        self._anx_count[hour_idx] = n
    
    def _record_anxiety(self, driver):
        """Record current anxiety level in hourly bins """
        anxiety = driver.get_current_anxiety()
        hour_idx = self._hour_bin()
        self._update_anxiety_stats(hour_idx, anxiety)
        
        # Keep a few samples for compatibility, but limit growth
        if len(self.stats['anxiety_levels']) < 1000:  # Limit to prevent memory issues
            self.stats['anxiety_levels'].append(anxiety)
    
    def _record_driver_count(self):
        """Record current number of active drivers in hourly bins"""
        hour_idx = self._hour_bin()
        # Count active drivers 
        # Approximate based on cars spawned vs completed/stranded
        active_drivers = self.stats['cars_spawned'] - self.stats['cars_completed'] - self.stats['cars_stranded']
        # Update running average of driver count for this hour
        self._driver_count_samples[hour_idx] += 1
        n = self._driver_count_samples[hour_idx]
        self._driver_count[hour_idx] += (active_drivers - self._driver_count[hour_idx]) / n

    def _update_queue_stats(self, hour_idx: int, queue_time: float) -> None:
        """Welford online algorithm update for queue time statistics per hour bin"""
        n = self._queue_count[hour_idx] + 1
        delta = queue_time - self._queue_mean[hour_idx]
        self._queue_mean[hour_idx] += delta / n
        delta2 = queue_time - self._queue_mean[hour_idx]
        self._queue_M2[hour_idx] += delta * delta2
        self._queue_count[hour_idx] = n

    def _record_queue_time(self, queue_time_minutes):
        """Record queue time in hourly bins"""
        hour_idx = self._hour_bin()
        self._update_queue_stats(hour_idx, queue_time_minutes)
    
    def _setup_charging_resources(self):
        """Setup SimPy resources for nodes that already have charging stations"""
        nodes_with_stations = setup_charging_resources(self.env, self.graph)
        print(f"Created SimPy resources for {nodes_with_stations} nodes with charging stations")
    
    def car_process(self, car_id, path, initial_soc, connector_type, total_distance_km, battery_capacity_km):
        """
        SimPy process representing one car's journey
        """
        source_node = path[0]
        destination_node = path[-1]
        print(f"[T={self.env.now:.1f}] Car {car_id}: Starting journey from {source_node} to {destination_node} with initial battery: {initial_soc:.2f} ({initial_soc*100:.0f}%) and connector type {connector_type}")
        
        # Create the driver/car
        driver = EVDriver(source_node, destination_node, initial_soc, connector_type, battery_capacity_km)
        driver.set_current_path(path)

        # Record initial anxiety
        self._record_anxiety(driver)
        
        # Record driver count (we just added a new active driver)
        self._record_driver_count()
        
        # Configuration - battery range in kilometers
        current_range_km = driver.get_range_remaining()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Can travel {current_range_km:.2f}km out of {total_distance_km:.2f}km with current battery")
        
        #--------------------MAIN JOURNEY LOOP-----------------------------------------------------------------------------------------------------------------------
        loop_counter = 0
        max_loops = len(path) * 2  # Safety limit to prevent infinite loops
        last_decay_time = self.env.now
        
        while not driver.has_reached_destination() and loop_counter < max_loops:
            loop_counter += 1
            current_node = driver.get_current_node()

            time_since_decay = self.env.now - last_decay_time
            if time_since_decay >= 60:  # Every 60 simulation time units (1 hour)
                driver.decay_penalties()
                last_decay_time = self.env.now
            
            # Check if we can make the next move
            if not driver.can_reach_next_node(self.graph):
                print(f"[T={self.env.now:.1f}] Car {car_id}: Cannot reach next node! Charging at current node {current_node}")
                success = yield from self._charge_at_current_location(car_id, driver, current_node)
                if not success:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED at {current_node} - no compatible charging!")
                    self.stats['cars_stranded'] += 1
                    return
                continue  # Restart loop after charging
            
            # Check if driver is anxious about charging
            needs_charging, current_range, deficit, reason = driver.needs_charging_for_journey(self.graph)
            
            if needs_charging:
                print(f"[T={self.env.now:.1f}] Car {car_id}: {reason}")
                success = yield from self._handle_charging_stop(car_id, driver)
                if not success:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED - no compatible charging stations within range!")
                    self.stats['cars_stranded'] += 1
                    return
                continue  # Restart loop after charging
            else:
                # Log the reason for not charging
                if "Anxious" in reason:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: {reason} - continuing journey")
            
            # Travel to next node
            yield from travel_to_next_node(self.env, car_id, driver, self.graph, travel_time_per_km=0.75)
            
            # Record anxiety after movement
            self._record_anxiety(driver)
            
            # Record driver count periodically to avoid overhead
            if loop_counter % 5 == 0:  # Every 5th movement
                self._record_driver_count()
            
            # Check if we've somehow run out of battery 
            if driver.is_battery_empty():
                print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED! Battery empty at node {driver.get_current_node()}")
                self.stats['cars_stranded'] += 1
                return
        
        if loop_counter >= max_loops:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Stopped due to loop limit - possible infinite loop prevented")
            self.stats['cars_stranded'] += 1
            return
        
        # Journey completed
        print(f"[T={self.env.now:.1f}] Car {car_id}: Reached destination {destination_node}!")
        self.stats['cars_completed'] += 1
    
    
    def _handle_charging_stop(self, car_id, driver):
        """
        Charging process with proper path restoration:
        1. Find best station and go there
        2. If queue > max_wait_minutes, find alternative at same node  
        3. If no alternatives at node, find other stations within reach
        4. After charging, recalculate path from charging location to destination
        5. Reacalculate pathb from charing station to next node
        """
        current_node = driver.get_current_node()
        current_range_km = driver.get_range_remaining()
        planned_route = driver.get_current_path()
        connector_type = driver.get_connector_type()
        destination_node = driver.get_destination_node()

        print(f"[T={self.env.now:.1f}] Car {car_id}: Need charging, looking for best station within {current_range_km:.1f}km range")

        # Step 1: Find best station within reach
        result = find_nearest_charging_station(
            self.graph, current_node, planned_route, current_range_km, connector_type
        )

        if result == (None, None):
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging stations within range!")
            return False

        target_node, target_station_id = result
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected charging station {target_station_id} at node {target_node}")

        # Step 2: Travel to target node if needed
        if target_node != current_node:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Traveling to charging station at node {target_node}")
            travel_success = yield from travel_to_charging_station(
                self.env, car_id, current_node, target_node, self.graph, driver
            )
            if not travel_success:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Failed to reach charging station!")
                return False
        else:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging station is at current location")

        # Step 3: Try to charge with queue tolerance
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, target_node, target_station_id, self.graph, self.stats, driver, self)
        
        # Record anxiety after charging completes
        self._record_anxiety(driver) 

        # Step 4: Handle charging failure
        if not charging_success:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging failed at node {target_node}")
            print(f"[T={self.env.now:.1f}] Car {car_id}: No acceptable charging options found")

        # Step 5: Recalculate path from charging location to destination
        print(f"[T={self.env.now:.1f}] Car {car_id}: Charging complete, recalculating route from {target_node} to {destination_node}")

        driver.set_source_node(target_node)
        new_path = driver.find_shortest_path(self.graph)

        if not new_path:
            print(f"[T={self.env.now:.1f}] Car {car_id}: ERROR - No path from charging station {target_node} to destination {destination_node}!")
            return False

        print(f"[T={self.env.now:.1f}] Car {car_id}: Resuming journey from {target_node} to {destination_node}")

        return True
        
    def _charge_at_current_location(self, car_id, driver, current_node):
        """
        Try to charge at the current location --> For revaluation and if it can't reach next node
        """
        stations = self.graph.nodes[current_node]['charging_stations']
        if not stations:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging stations at current node {current_node}")
            return False
    
        connector_type = driver.get_connector_type()
        from charging_utils import has_compatible_connector
        from pathfinding import get_station_max_power

        if not has_compatible_connector(stations, connector_type):
            print(f"[T={self.env.now:.1f}] Car {car_id}: No compatible charging stations at node {current_node} for connector {connector_type}")
            self.stats['connector_incompatibility_failures'] += 1
            return False

        # Find the best station at current location
        # Collect all compatible stations with scores
        compatible_stations = []
        for station in stations:
            if has_compatible_connector([station], connector_type):
                # Score this station
                score = 1000
                queue_length = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                estimated_wait_time = queue_length * 10
                score -= estimated_wait_time * 10

                max_power = get_station_max_power(station, connector_type)
                if max_power >= 150:
                    score += 200
                elif max_power >= 50:
                    score += 100
                elif max_power >= 22:
                    score += 50
                else:
                    score += 20

                compatible_stations.append((station, score))

        if not compatible_stations:
            return False

        # Use weighted random selection
        stations_list = [opt[0] for opt in compatible_stations]
        weights = [opt[1] for opt in compatible_stations]
        total_weight = sum(weights)
        probabilities = [w/total_weight for w in weights]

        import random
        best_station = random.choices(stations_list, weights=probabilities)[0]
        best_station_id = best_station.get_station_id()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected best station at current location: {best_station_id}")

        # Use the queue tolerance charging function
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, current_node, best_station_id, self.graph, self.stats, driver, self)

        return charging_success
    
    
    
    def spawn_multiple_cars(self, total_cars, simulation_duration_hours=24):
        """
        Spawn cars gradually over a time period according to hourly distribution
        """
        # Hourly charging demand distribution (T1-T24)
        hourly_distribution = [
            0.5,  # T1  (00:00–01:00)
            0.4,  # T2  (01:00–02:00)
            0.3,  # T3  (02:00–03:00)
            0.8,  # T4  (03:00–04:00)
            8.0,  # T5  (04:00–05:00)  ← long-trip starts begin
            17.5, # T6  (05:00–06:00)
            20.0, # T7  (06:00–07:00)  ← peak
            11.0, # T8  (07:00–08:00)
            7.0,  # T9  (08:00–09:00)
            5.0,  # T10 (09:00–10:00)
            4.0,  # T11 (10:00–11:00)
            3.0,  # T12 (11:00–12:00)
            2.5,  # T13 (12:00–13:00)
            2.5,  # T14 (13:00–14:00)
            2.0,  # T15 (14:00–15:00)
            2.0,  # T16 (15:00–16:00)
            3.0,  # T17 (16:00–17:00)
            4.0,  # T18 (17:00–18:00)
            3.5,  # T19 (18:00–19:00)
            1.5,  # T20 (19:00–20:00)
            0.8,  # T21 (20:00–21:00)
            0.4,  # T22 (21:00–22:00)
            0.2,  # T23 (22:00–23:00)
            0.1   # T24 (23:00–00:00)
        ]
        
        # Convert simulation hours to simulation time units (assuming 1 hour = 60 time units)
        time_units_per_hour = 60
        simulation_duration_units = simulation_duration_hours * time_units_per_hour
        
        all_nodes = list(self.graph.nodes)
        battery_capacity_km = 300
        
        # Connector type distribution
        connector_types = [1, 2, 3]
        connector_weights = [0.2, 0.7, 0.1]
        
        print(f"=== Spawning {total_cars} cars over {simulation_duration_hours} hours ===")
        
        car_id = 1
        
        # Process each hour
        for hour_index, percentage in enumerate(hourly_distribution):
            hour_number = hour_index + 1
            
            # Calculate how many cars to spawn in this hour
            cars_this_hour = int(total_cars * percentage / 100)
            
            # Handle rounding by distributing remaining cars to peak hours
            if car_id + cars_this_hour - 1 > total_cars:
                cars_this_hour = total_cars - car_id + 1
            
            if cars_this_hour <= 0:
                continue
            
            print(f"\n[T={self.env.now:.1f}] Starting hour T{hour_number} - spawning {cars_this_hour} cars over next hour")
            
            # Calculate spawn intervals within this hour
            if cars_this_hour == 1:
                # Single car - spawn at random time within the hour
                spawn_times = [random.uniform(0, time_units_per_hour)]
            else:
                # Multiple cars - distribute evenly with some randomization
                base_interval = time_units_per_hour / cars_this_hour
                spawn_times = []
                
                for i in range(cars_this_hour):
                    # Base time + small random offset to avoid exact simultaneity
                    base_time = i * base_interval
                    random_offset = random.uniform(-base_interval * 0.3, base_interval * 0.3)
                    spawn_time = max(0, min(time_units_per_hour - 1, base_time + random_offset))
                    spawn_times.append(spawn_time)
                
                # Sort spawn times to ensure chronological order
                spawn_times.sort()
            
            # Spawn cars at calculated times within this hour
            for i, spawn_offset in enumerate(spawn_times):
                if car_id > total_cars:
                    break
                
                # Wait until the spawn time for this car
                if i == 0:
                    # First car in this hour - wait from start of hour
                    yield self.env.timeout(spawn_offset)
                else:
                    # Subsequent cars - wait for the interval between this and previous car
                    interval = spawn_offset - spawn_times[i-1]
                    if interval > 0:
                        yield self.env.timeout(interval)
                
                # Generate car parameters
                while True:  # Keep trying until we find a valid route
                    source = random.choice(all_nodes)
                    destination = random.choice(all_nodes)
                    
                    if source != destination:
                        try:
                            path = nx.shortest_path(self.graph, source, destination, weight='weight')
                            if path:
                                # Calculate total distance
                                total_distance_km = 0
                                for j in range(len(path) - 1):
                                    node1, node2 = path[j], path[j + 1]
                                    if self.graph.has_edge(node1, node2):
                                        total_distance_km += self.graph.edges[node1, node2]['weight']
                                
                                if 100 <= total_distance_km <= 500:
                                    break  # Valid route found
                        except nx.NetworkXNoPath:
                            continue  # No path found, try again
                
                # Generate car characteristics
                initial_soc = random.uniform(0.6, 0.9)
                connector_type = random.choices(connector_types, weights=connector_weights)[0]
                
                # Calculate current hour and minute for logging
                current_sim_time = self.env.now
                current_hour = int(current_sim_time // time_units_per_hour) + 1
                current_minute = int((current_sim_time % time_units_per_hour))
                
                print(f"[T={self.env.now:.1f}] (Hour T{current_hour}, +{current_minute}min) "
                      f"Spawning car {car_id} from {source} to {destination} "
                      f"({total_distance_km:.0f}km, SoC: {initial_soc:.2f}, connector: {connector_type})")
                
                # Start the car process
                self.env.process(self.car_process(car_id, path, initial_soc, connector_type, total_distance_km, battery_capacity_km))
                self.stats['cars_spawned'] += 1
                
                car_id += 1
            
            # Wait for any remaining time in this hour before moving to next hour
            time_spent_this_hour = spawn_times[-1] if spawn_times else 0
            remaining_time = time_units_per_hour - time_spent_this_hour
            if remaining_time > 0:
                yield self.env.timeout(remaining_time)
        
        print(f"\n=== Car spawning complete: {car_id - 1} cars spawned over {simulation_duration_hours} hours ===")
    
    def _finalize_anxiety_stats(self):
        """Compute final anxiety and queue statistics and store in self.stats"""
        anxiety_mean_T = []
        anxiety_std_T = []
        anxiety_count_T = self._anx_count[:]

        queue_mean_T = []
        queue_std_T = []
        queue_count_T = self._queue_count[:]

        for i in range(24):
            # Anxiety stats (existing code)
            n = self._anx_count[i]
            if n >= 2:
                mean = self._anx_mean[i]
                std = math.sqrt(self._anx_M2[i] / (n - 1))
            elif n == 1:
                mean = self._anx_mean[i]
                std = 0.0
            else:  # n == 0
                mean = 0.0
                std = 0.0

            anxiety_mean_T.append(mean)
            anxiety_std_T.append(std)

            # Queue stats (new code)
            n_queue = self._queue_count[i]
            if n_queue >= 2:
                queue_mean = self._queue_mean[i]
                queue_std = math.sqrt(self._queue_M2[i] / (n_queue - 1))
            elif n_queue == 1:
                queue_mean = self._queue_mean[i]
                queue_std = 0.0
            else:  # n_queue == 0
                queue_mean = 0.0
                queue_std = 0.0

            queue_mean_T.append(queue_mean)
            queue_std_T.append(queue_std)

        # Store in stats
        self.stats['anxiety_mean_T'] = anxiety_mean_T
        self.stats['anxiety_std_T'] = anxiety_std_T
        self.stats['anxiety_count_T'] = anxiety_count_T
        self.stats['driver_count_T'] = self._driver_count[:]  # Average drivers per hour
        self.stats['queue_mean_T'] = queue_mean_T
        self.stats['queue_std_T'] = queue_std_T
        self.stats['queue_count_T'] = queue_count_T

        # Acceptance checks 
        assert len(self.stats['anxiety_mean_T']) == 24, f"Expected 24 mean values, got {len(self.stats['anxiety_mean_T'])}"
        assert len(self.stats['anxiety_std_T']) == 24, f"Expected 24 std values, got {len(self.stats['anxiety_std_T'])}"
        assert len(self.stats['queue_mean_T']) == 24, f"Expected 24 queue mean values, got {len(self.stats['queue_mean_T'])}"
        assert len(self.stats['queue_std_T']) == 24, f"Expected 24 queue std values, got {len(self.stats['queue_std_T'])}"

        # Check value ranges
        for i, (mean, std) in enumerate(zip(anxiety_mean_T, anxiety_std_T)):
            assert 0.0 <= mean <= 1.0, f"Hour T{i+1}: mean {mean} out of range [0,1]"
            assert std >= 0.0, f"Hour T{i+1}: std {std} negative"

        # Check queue value ranges
        for i, (queue_mean, queue_std) in enumerate(zip(queue_mean_T, queue_std_T)):
            assert queue_mean >= 0.0, f"Hour T{i+1}: queue mean {queue_mean} negative"
            assert queue_std >= 0.0, f"Hour T{i+1}: queue std {queue_std} negative"

        # Check we have some data
        total_samples = sum(anxiety_count_T)
        assert total_samples > 0, f"No anxiety samples collected: {anxiety_count_T}"

        print(f"\nAnxiety statistics finalized: {total_samples} total samples across 24 hours")
        print(f"Hours with data: {sum(1 for c in anxiety_count_T if c > 0)}/24")

        total_queue_samples = sum(queue_count_T)
        print(f"Queue statistics finalized: {total_queue_samples} total queue events across 24 hours")
        print(f"Hours with queue data: {sum(1 for c in queue_count_T if c > 0)}/24")
    
    def run_simulation(self):
        """Run the simulation"""
        print("=== Starting EV Simulation===")
        print(f"Graph has {len(self.graph.nodes)} nodes")
        
        # Spawn cars with hourly demand distribution
        self.env.process(self.spawn_multiple_cars(total_cars=1000, simulation_duration_hours=24))
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until all cars complete their journeys ---")
            self.env.run(until=1440)  # Run until all processes finish
        
        # Finalize anxiety statistics
        self._finalize_anxiety_stats()
        
        # Print final statistics
        print("\n=== Simulation Complete ===")
        print("FINAL STATISTICS:")
        print(f"Cars spawned: {self.stats['cars_spawned']}")
        print(f"Cars completed journey: {self.stats['cars_completed']}")
        print(f"Cars stranded: {self.stats['cars_stranded']}")
        print(f"Total charging events: {self.stats['total_charging_events']}")
        print(f"Connector incompatibility failures: {self.stats['connector_incompatibility_failures']}")
        print(f"Queue statistics:")
        queue_times = self.stats['queue_times']
        queue_length = self.stats['queue_length']
        if queue_times:
            print(f"  Average queue time: {sum(queue_times)/len(queue_times):.2f} mins")
            print(f"  Min queue time: {min(queue_times):.2f} mins")
            print(f"  Max queue time: {max(queue_times):.2f} mins")
            print(f"  Total cars that queued: {len(queue_times)}")
        else:
            print(f"  No cars experienced queue time (all found empty stations)")
            print(f"  Total cars that queued: 0")

        if queue_length:
            print(f"  Average queue length: {sum(queue_length)/len(queue_length):.2f} cars")
            print(f"  Min queue length: {min(queue_length)} cars")
            print(f"  Max queue length: {max(queue_length)} cars")
        else:
            print(f"  No queue length data recorded")

        # Calculate success rate
        if self.stats['cars_spawned'] > 0:
            success_rate = (self.stats['cars_completed'] / self.stats['cars_spawned']) * 100
            print(f"Journey completion rate: {success_rate:.1f}%")

            if self.stats['connector_incompatibility_failures'] > 0:
                incompatibility_rate = (self.stats['connector_incompatibility_failures'] / self.stats['cars_spawned']) * 100
                print(f"Connector incompatibility rate: {incompatibility_rate:.1f}%")

        if self.stats['cars_completed'] == 0 and self.stats['cars_spawned'] > 0:
            print("No cars completed their journey - they may be stranded or need more time!")

        # Simple anxiety statistics
        if self.stats['anxiety_levels']:
            print(f"Legacy anxiety recordings: {len(self.stats['anxiety_levels'])} samples")
            print(f"Average anxiety (legacy): {sum(self.stats['anxiety_levels'])/len(self.stats['anxiety_levels']):.4f}")
        
        #Hourly anxiety statistics
        print(f"\nHourly Anxiety Statistics:")
        for i in range(24):
            hour_label = f"T{i+1}"
            count = self.stats['anxiety_count_T'][i]
            mean = self.stats['anxiety_mean_T'][i]
            std = self.stats['anxiety_std_T'][i]
            if count > 0:
                print(f"  {hour_label}: {count:4d} samples, mean={mean:.4f}, std={std:.4f}")
            else:
                print(f"  {hour_label}: {count:4d} samples, no data")


def main():
    print("Loading graph and charging stations...")
    
    # Loads data
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleaned_charging_stations.json"
    
    random.seed(1)
    try:
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is None:
            print("Failed to load graph!")
            return
        
        print(f"Graph loaded successfully with {len(graph.nodes)} nodes")
        
        # Create and run simulation
        simulation = SimpleEVSimulation(graph, simulation_time=None)  # No time limit
        simulation.run_simulation()
        
        # Plot anxiety profile
        print("\n" + "="*60)
        print("CREATING ANXIETY VISUALIZATION")
        print("="*60)
        
        try:
            from visualization import plot_anxiety_T_profile, print_anxiety_summary_table
            from visualization import plot_queue_time_T_profile, print_queue_summary_table
            
            # Get anxiety data
            mean_T = simulation.stats['anxiety_mean_T']
            std_T = simulation.stats['anxiety_std_T']
            count_T = simulation.stats['anxiety_count_T']
            driver_count_T = simulation.stats.get('driver_count_T', None)
            
            # Get queue data
            queue_mean_T = simulation.stats.get('queue_mean_T', [0] * 24)
            queue_std_T = simulation.stats.get('queue_std_T', [0] * 24)
            queue_count_T = simulation.stats.get('queue_count_T', [0] * 24)
            
            # Print detailed summary tables
            #print_anxiety_summary_table(mean_T, std_T, count_T, driver_count_T)
            #print_queue_summary_table(queue_mean_T, queue_std_T, queue_count_T, driver_count_T)
            
            # Create the anxiety plot with driver count
            print("\nGenerating anxiety profile plot with active driver count...")
            plot_anxiety_T_profile(mean_T, std_T, driver_count_T, 
                                 title="EV Driver Anxiety Throughout the Day (Public Charging)")
            
            # Create the queue time plot with driver count
            print("\nGenerating queue time profile plot with active driver count...")
            plot_queue_time_T_profile(queue_mean_T, queue_std_T, driver_count_T,
                                    title="EV Charging Queue Times Throughout the Day")
            
            print("All visualizations complete!")
            
        except ImportError as e:
            print(f"Could not import visualization functions: {e}")
            print("Data is available in simulation.stats for manual plotting:")
            print(f"  anxiety_mean_T: {len(simulation.stats['anxiety_mean_T'])} values")
            print(f"  anxiety_std_T: {len(simulation.stats['anxiety_std_T'])} values")
            print(f"  anxiety_count_T: {len(simulation.stats['anxiety_count_T'])} values")
            print(f"  queue_mean_T: {len(simulation.stats.get('queue_mean_T', []))} values")
            print(f"  queue_std_T: {len(simulation.stats.get('queue_std_T', []))} values")
            print(f"  queue_count_T: {len(simulation.stats.get('queue_count_T', []))} values")
            print(f"  driver_count_T: {len(simulation.stats.get('driver_count_T', []))} values")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()