import simpy
import random
import networkx as nx
import numpy as np
from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import (find_nearest_charging_station, travel_to_charging_station,travel_to_next_node)
from charging_utils import charge_at_station_with_queue_tolerance, setup_charging_resources
from queue_time_tracker import QueueTimeTracker  # Import the tracker

class SimpleEVSimulation:
    def __init__(self, graph, simulation_time=None):
        """
        Enhanced simulation with queue time tracking
        """
        self.env = simpy.Environment()
        self.graph = graph
        self.simulation_time = simulation_time
        
        # Add queue time tracker
        self.queue_tracker = QueueTimeTracker()
        
        # Create SimPy resources from existing charging stations
        self._setup_charging_resources()

        self.active_drivers = {}
        
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
            'queue_abandonments': 0,  # Track queue abandonments
            'alternative_station_switches': 0
        }
    
    def _setup_charging_resources(self):
        """Setup SimPy resources for nodes that already have charging stations"""
        nodes_with_stations = setup_charging_resources(self.env, self.graph)
        print(f"Created SimPy resources for {nodes_with_stations} nodes with charging stations")
    
    def record_queue_event(self, queue_time_minutes, queue_start_time=None):
        """Record a queue time event for hourly tracking"""
        # Use start time if provided, otherwise current time (for backward compatibility)
        record_time = queue_start_time if queue_start_time is not None else self.env.now
        self.queue_tracker.record_queue_time(record_time, queue_time_minutes)
    
    def register_active_driver(self, car_id, driver):
        """Register a driver as active"""
        self.active_drivers[car_id] = driver
    
    def unregister_active_driver(self, car_id):
        """Remove a driver from active list"""
        if car_id in self.active_drivers:
            del self.active_drivers[car_id]
    
    def update_driver_count_and_soc(self):
        """Update both driver count and SoC sampling"""
        current_time = self.env.now
        
        # Existing driver count update
        if self.queue_tracker.should_sample_driver_count(current_time):
            active_count = len(self.active_drivers)
            self.queue_tracker.record_driver_count(current_time, active_count)
        
        # NEW: SoC sampling
        if self.queue_tracker.should_sample_soc(current_time):
            if self.active_drivers:
                total_soc = sum(driver.get_state_of_charge() for driver in self.active_drivers.values())
                active_count = len(self.active_drivers)
                self.queue_tracker.record_soc_data(current_time, total_soc, active_count)
    
    def car_process(self, car_id, path, initial_soc, connector_type, total_distance_km, battery_capacity_km):
        """
        Car process
        """
        source_node = path[0]
        destination_node = path[-1]
        print(f"[T={self.env.now:.1f}] Car {car_id}: Starting journey from {source_node} to {destination_node} with initial battery: {initial_soc:.2f} ({initial_soc*100:.0f}%) and connector type {connector_type}")
        
        # Create the driver/car
        driver = EVDriver(source_node, destination_node, initial_soc, connector_type, battery_capacity_km)
        driver.set_current_path(path)

        self.register_active_driver(car_id, driver)
        
        # Configuration - battery range in kilometers
        current_range_km = driver.get_range_remaining()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Can travel {current_range_km:.2f}km out of {total_distance_km:.2f}km with current battery")
        
        # Update driver count when car starts
        self.update_driver_count_and_soc()
        
        #--------------------MAIN JOURNEY LOOP-----------------------------------------------------------------------------------------------------------------------
        loop_counter = 0
        max_loops = len(path) * 2  # Safety limit to prevent infinite loops
        
        while not driver.has_reached_destination() and loop_counter < max_loops:
            loop_counter += 1
            current_node = driver.get_current_node()
            
            # Periodically update driver count
            if loop_counter % 5 == 0:  # Every 5 loop iterations
                self.update_driver_count_and_soc()
            
            # Check if we can make the next move
            if not driver.can_reach_next_node(self.graph):
                print(f"[T={self.env.now:.1f}] Car {car_id}: Cannot reach next node! Charging at current node {current_node}")
                success = yield from self._charge_at_current_location(car_id, driver, current_node)
                if not success:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED at {current_node} - no compatible charging!")
                    self.stats['cars_stranded'] += 1
                    self.unregister_active_driver(car_id)
                    return
                continue  # Restart loop after charging
            
            # Check if driver needs charging based on personal threshold
            needs_charging, current_range, deficit, reason = driver.needs_charging_for_journey(self.graph)
            
            if needs_charging:
                print(f"[T={self.env.now:.1f}] Car {car_id}: {reason}")
                success = yield from self._handle_charging_stop(car_id, driver)
                if not success:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED - no compatible charging stations within range!")
                    self.stats['cars_stranded'] += 1
                    self.unregister_active_driver(car_id)
                    return
                continue  # Restart loop after charging
            else:
                # Log the reason for not charging
                print(f"[T={self.env.now:.1f}] Car {car_id}: {reason} - continuing journey")
            
            # Travel to next node
            yield from travel_to_next_node(self.env, car_id, driver, self.graph, travel_time_per_km=0.75)
            
            # Check if we've somehow run out of battery 
            if driver.is_battery_empty():
                print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED! Battery empty at node {driver.get_current_node()}")
                self.stats['cars_stranded'] += 1
                self.unregister_active_driver(car_id)
                return
        
        if loop_counter >= max_loops:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Stopped due to loop limit - possible infinite loop prevented")
            self.stats['cars_stranded'] += 1
            self.unregister_active_driver(car_id)
            return
        
        # Journey completed
        print(f"[T={self.env.now:.1f}] Car {car_id}: Reached destination {destination_node}!")
        self.stats['cars_completed'] += 1

        self.unregister_active_driver(car_id)
        
        # Update driver count when car completes
        self.update_driver_count_and_soc()
    
    def _handle_charging_stop(self, car_id, driver):
        """
        Charging process using unified charging function
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

            # Last resort: try nearby search with current range
            if current_range_km > 5:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Attempting last-resort nearby search")
                success = yield from self._search_and_travel_to_nearby_charging(car_id, driver)
                if success:
                    # After charging, recalculate path
                    charging_location = driver.get_current_node()
                    driver.set_source_node(charging_location)
                    new_path = driver.find_shortest_path(self.graph)
                    return new_path is not None

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

        # Step 3: Try to charge 
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, target_node, target_station_id, self.graph, self.stats, driver, 
            simulation=self  # Pass simulation for queue tracking
        )

        # Step 4: Handle charging failure with nearby search
        if not charging_success:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging failed at node {target_node}")

            # Try one more nearby search if we still have range
            remaining_range = driver.get_range_remaining()
            if remaining_range > 10:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Attempting alternative nearby search after charging failure")
                alternative_success = yield from self._search_and_travel_to_nearby_charging(car_id, driver)
                if alternative_success:
                    charging_success = True

        # Step 5: Recalculate path from charging location to destination
        if charging_success:
            current_location = driver.get_current_node()
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging complete, recalculating route from {current_location} to {destination_node}")

            driver.set_source_node(current_location)
            new_path = driver.find_shortest_path(self.graph)

            if not new_path:
                print(f"[T={self.env.now:.1f}] Car {car_id}: ERROR - No path from {current_location} to destination {destination_node}!")
                return False

            print(f"[T={self.env.now:.1f}] Car {car_id}: Resuming journey from {current_location} to {destination_node}")

        return charging_success

        
    def _charge_at_current_location(self, car_id, driver, current_node):
        """
        Enhanced charging at current location using unified charging function
        """
        stations = self.graph.nodes[current_node]['charging_stations']
        connector_type = driver.get_connector_type()
        remaining_range = driver.get_range_remaining()

        # Case 1: No charging stations at current node
        if not stations:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging stations at current node {current_node}")

            # Search nearby nodes if we have enough range
            if remaining_range > 5:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Searching for charging stations in nearby nodes (range: {remaining_range:.1f}km)")
                success = yield from self._search_and_travel_to_nearby_charging(car_id, driver)
                return success
            else:
               print(f"[T={self.env.now:.1f}] Car {car_id}: Insufficient range ({remaining_range:.1f}km) to search nearby nodes")
               return False
    
        from charging_utils import has_compatible_connector

        # Case 2: Stations exist but none are compatible
        if not has_compatible_connector(stations, connector_type):
            print(f"[T={self.env.now:.1f}] Car {car_id}: No compatible charging stations at node {current_node} for connector {connector_type}")
        
            # Search nearby nodes for compatible stations if we have enough range
            if remaining_range > 5:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Searching for compatible stations in nearby nodes (range: {remaining_range:.1f}km)")
                success = yield from self._search_and_travel_to_nearby_charging(car_id, driver)
                if success:
                    return True
                else:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: No compatible stations found within range")
            else:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Insufficient range to search for compatible stations elsewhere")

            self.stats['connector_incompatibility_failures'] += 1
            return False

        # Case 3: Compatible stations exist at current node
        print(f"[T={self.env.now:.1f}] Car {car_id}: Found compatible stations at current node")

        # Find the best station at current location
        from pathfinding import get_station_max_power
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

        best_station = random.choices(stations_list, weights=probabilities)[0]
        best_station_id = best_station.get_station_id()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected best station at current location: {best_station_id}")

        # Use the unified charging function WITH TRACKING
        from charging_utils import charge_at_station_with_queue_tolerance
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, current_node, best_station_id, self.graph, self.stats, driver,
            simulation=self  # Pass simulation for queue tracking
        )

        return charging_success

    
    def spawn_multiple_cars(self, total_cars, simulation_duration_hours=24):
        """
        Spawn cars gradually over a time period according to hourly distribution
        """
        # Hourly charging demand distribution (T1-T24)
        hourly_distribution = [
            0.4,  # T1  (00:00–01:00)
            0.3,  # T2  (01:00–02:00)
            0.3,  # T3  (02:00–03:00)
            1.0,  # T4  (03:00–04:00)
            3.0,  # T5  (04:00–05:00)
            6.0,  # T6  (05:00–06:00)
            9.5,  # T7  (06:00–07:00)
            11.0, # T8  (07:00–08:00)
            12.5, # T9  (08:00–09:00) ← peak
            11.8, # T10 (09:00–10:00)
            8.8,  # T11 (10:00–11:00)
            7.2,  # T12 (11:00–12:00)
            6.0,  # T13 (12:00–13:00)
            5.0,  # T14 (13:00–14:00)
            4.0,  # T15 (14:00–15:00)
            3.0,  # T16 (15:00–16:00)
            2.9,  # T17 (16:00–17:00)
            2.5,  # T18 (17:00–18:00)
            1.8,  # T19 (18:00–19:00)
            1.2,  # T20 (19:00–20:00)
            0.7,  # T21 (20:00–21:00)
            0.5,  # T22 (21:00–22:00)
            0.3,  # T23 (22:00–23:00)
            0.3   # T24 (23:00–00:00)
        ]
        
        # Convert simulation hours to simulation time units (assuming 1 hour = 60 time units)
        time_units_per_hour = 60
        
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
                spawn_times = [random.uniform(0, time_units_per_hour)]
            else:
                base_interval = time_units_per_hour / cars_this_hour
                spawn_times = []
                
                for i in range(cars_this_hour):
                    base_time = i * base_interval
                    random_offset = random.uniform(-base_interval * 0.3, base_interval * 0.3)
                    spawn_time = max(0, min(time_units_per_hour - 1, base_time + random_offset))
                    spawn_times.append(spawn_time)
                
                spawn_times.sort()
            
            # Spawn cars at calculated times within this hour
            for i, spawn_offset in enumerate(spawn_times):
                if car_id > total_cars:
                    break
                
                # Wait until the spawn time for this car
                if i == 0:
                    yield self.env.timeout(spawn_offset)
                else:
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
                                
                                if 50 <= total_distance_km <= 500:
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
    
    def run_simulation(self):
        """Run the simulation"""
        print("=== Starting EV Simulation with Queue Time Tracking ===")
        print(f"Graph has {len(self.graph.nodes)} nodes")
        
        # Spawn cars with hourly demand distribution
        self.env.process(self.spawn_multiple_cars(total_cars=10000, simulation_duration_hours=24)) #CARSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until completion ---")
            self.env.run()  # Run for 24 hours
        
        # Print final statistics
        print("\n=== Simulation Complete ===")
        print("FINAL STATISTICS:")
        print(f"Cars spawned: {self.stats['cars_spawned']}")
        print(f"Cars completed journey: {self.stats['cars_completed']}")
        print(f"Cars stranded: {self.stats['cars_stranded']}")
        print(f"Total charging events: {self.stats['total_charging_events']}")
        print(f"Connector incompatibility failures: {self.stats['connector_incompatibility_failures']}")
        print(f"Alternative station switches: {self.stats['alternative_station_switches']}")  
        print(f"Queue abandonments: {self.stats['queue_abandonments']}")
        
        print(f"Queue statistics:")
        queue_times = self.stats['queue_times']
        queue_length = self.stats['queue_length']
        if queue_times:
            print(f"  Average queue time: {sum(queue_times)/len(queue_times):.2f} mins")
            print(f"  Min queue time: {min(queue_times):.2f} mins")
            print(f"  Max queue time: {max(queue_times):.2f} mins")
            print(f"  Total cars that queued: {len(queue_times)}")
        else:
            print(f"  No cars experienced queue time")

        if queue_length:
            print(f"  Average queue length: {sum(queue_length)/len(queue_length):.2f} cars")
            print(f"  Min queue length: {min(queue_length)} cars")
            print(f"  Max queue length: {max(queue_length)} cars")

        # Calculate success rate
        if self.stats['cars_spawned'] > 0:
            success_rate = (self.stats['cars_completed'] / self.stats['cars_spawned']) * 100
            print(f"Journey completion rate: {success_rate:.1f}%")

            if self.stats['queue_abandonments'] > 0:
                abandonment_rate = (self.stats['queue_abandonments'] / self.stats['cars_spawned']) * 100
                print(f"Queue abandonment rate: {abandonment_rate:.1f}%")

        

        # ENHANCED: Print hourly queue analysis and create plots
        print("\n" + "="*60)
        print("HOURLY QUEUE TIME ANALYSIS")
        print("="*60)
        
        # Print detailed hourly summary
        self.queue_tracker.print_hourly_summary()
        
        # Create the visualization
        print("\nGenerating queue time visualization...")
        #self.queue_tracker.plot_hourly_queue_times("EV Charging Queue Times Throughout the Day")
        
        print("\n=== SoC Data Analysis ===")
        self.queue_tracker.export_soc_data_to_csv("simulation_soc_data.csv")
        
        # Print SoC summary
        (hours, avg_soc, std_soc, median_soc, q1_soc, q3_soc, counts) = self.queue_tracker.calculate_hourly_soc_statistics()
        # Print SoC summary
        print(f"SoC Statistics:")
        for i in range(24):
            if counts[i] > 0:
                print(f"  T{i+1}: Avg SoC = {avg_soc[i]:.3f} ({avg_soc[i]*100:.1f}%), "
                    f"Median = {median_soc[i]:.3f}, Samples = {counts[i]}")
    
        return self.queue_tracker


def main():
    print("Loading graph and charging stations...")
    
    # Load data
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleaned_charging_stations.json"
    
    #random.seed(1)
    #np.random.seed(1)
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
        simulation = SimpleEVSimulation(graph, simulation_time=None)
        queue_tracker = simulation.run_simulation()
        
        # You can also access the raw data if needed
        hours, avg_queue, q1_queue, q3_queue, avg_drivers, counts = queue_tracker.calculate_hourly_statistics()
        print(f"\nRaw data available for further analysis:")
        print(f"Hours: {len(hours)} data points")
        print(f"Peak queue time: {max(avg_queue):.2f} minutes")
        print(f"Peak driver count: {max(avg_drivers):.0f} active drivers")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()