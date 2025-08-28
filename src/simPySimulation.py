import simpy
import random
import networkx as nx
import numpy as np
from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import (find_nearest_charging_station, travel_to_charging_station, travel_to_next_node, get_station_max_power)
from charging_utils import charge_at_station_with_queue_tolerance, setup_charging_resources
from queue_time_tracker import QueueTimeTracker

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
            'queue_abandonments': 0,
            'alternative_station_switches': 0
        }
    
    def _setup_charging_resources(self):
        """Setup SimPy resources for nodes that already have charging stations"""
        nodes_with_stations = setup_charging_resources(self.env, self.graph)
        print(f"Created SimPy resources for {nodes_with_stations} nodes with charging stations")
    
    def record_queue_event(self, queue_time_minutes, queue_start_time=None):
        """Record a queue time event for hourly tracking"""
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
        
        # Driver count update
        if self.queue_tracker.should_sample_driver_count(current_time):
            active_count = len(self.active_drivers)
            self.queue_tracker.record_driver_count(current_time, active_count)
        
        # SoC sampling
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
        
        # Main journey loop
        loop_counter = 0
        max_loops = len(path) * 2  # Safety limit to prevent infinite loops
        
        while not driver.has_reached_destination() and loop_counter < max_loops:
            loop_counter += 1
            current_node = driver.get_current_node()
            
            # Periodically update driver count
            if loop_counter % 5 == 0:
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
                continue
            
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
                print(f"[T={self.env.now:.1f}] Car {car_id}: {reason} - continuing journey")
            
            # Travel to next node
            yield from travel_to_next_node(self.env, car_id, driver, self.graph, travel_time_per_km=0.75)
            
            # Check if battery is empty
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
        Charging process with equipment discovery retry logic
        """
        current_node = driver.get_current_node()
        current_range_km = driver.get_range_remaining()
        planned_route = driver.get_current_path()
        connector_type = driver.get_connector_type()
        destination_node = driver.get_destination_node()

        print(f"[T={self.env.now:.1f}] Car {car_id}: Need charging, looking for best station within {current_range_km:.1f}km range")

        # Find best station within reach
        result = find_nearest_charging_station(
            self.graph, current_node, planned_route, current_range_km, connector_type
        )

        if result == (None, None):
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging stations within range!")
            return False

        target_node, target_station_id = result
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected charging station {target_station_id} at node {target_node}")

        # Travel to target node if needed
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

        # Try to charge
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, target_node, target_station_id, self.graph, self.stats, driver, 
            simulation=self
        )

        # Handle charging failure with equipment discovery retry
        if not charging_success:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging failed at node {target_node} - equipment not working or unavailable")
            
            # Retry finding a new charging station
            remaining_range = driver.get_range_remaining()
            if remaining_range > 15:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Retrying charging station search due to equipment failure")
                
                retry_result = find_nearest_charging_station(
                    self.graph, driver.get_current_node(), driver.get_current_path(), 
                    remaining_range, connector_type
                )
                
                if retry_result != (None, None):
                    retry_node, retry_station_id = retry_result
                    print(f"[T={self.env.now:.1f}] Car {car_id}: Found alternative station {retry_station_id} at node {retry_node}")
                    
                    # Travel to new station if needed
                    if retry_node != driver.get_current_node():
                        travel_success = yield from travel_to_charging_station(
                            self.env, car_id, driver.get_current_node(), retry_node, self.graph, driver
                        )
                        if not travel_success:
                            print(f"[T={self.env.now:.1f}] Car {car_id}: Failed to reach retry station!")
                            return False
                    
                    # Try charging at retry station
                    retry_success = yield from charge_at_station_with_queue_tolerance(
                        self.env, car_id, retry_node, retry_station_id, self.graph, self.stats, driver, 
                        simulation=self
                    )
                    
                    if retry_success:
                        charging_success = True
                        print(f"[T={self.env.now:.1f}] Car {car_id}: Successfully charged at retry station")
                else:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: No alternative charging stations found within range")
            else:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Insufficient range ({remaining_range:.1f}km) for retry search")

        # Recalculate path from charging location to destination
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
        Enhanced charging at current location with equipment discovery
        """
        stations = self.graph.nodes[current_node]['charging_stations']
        connector_type = driver.get_connector_type()
        remaining_range = driver.get_range_remaining()

        # Case 1: No charging stations at current node
        if not stations:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging stations at current node {current_node}")
            if remaining_range > 5:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Searching for charging stations in nearby nodes (range: {remaining_range:.1f}km)")
                
                result = find_nearest_charging_station(
                    self.graph, current_node, driver.get_current_path(), 
                    remaining_range, connector_type
                )
                
                if result != (None, None):
                    alt_node, alt_station_id = result
                    print(f"[T={self.env.now:.1f}] Car {car_id}: Found alternative station {alt_station_id} at node {alt_node}")
                    
                    travel_success = yield from travel_to_charging_station(
                        self.env, car_id, current_node, alt_node, self.graph, driver
                    )
                    if travel_success:
                        return (yield from charge_at_station_with_queue_tolerance(
                            self.env, car_id, alt_node, alt_station_id, self.graph, self.stats, driver,
                            simulation=self
                        ))
                
                print(f"[T={self.env.now:.1f}] Car {car_id}: No alternative stations found")
            else:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Insufficient range ({remaining_range:.1f}km) to search nearby nodes")
            return False

        # Case 2: Check if any stations have working compatible connections
        stations_with_working_connections = []
        for station in stations:
            working_connections = station.get_working_connections(connector_type)
            if working_connections:
                stations_with_working_connections.append(station)

        if not stations_with_working_connections:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No working compatible charging stations at node {current_node}")
            
            if remaining_range > 10:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Searching for working compatible stations in nearby nodes (range: {remaining_range:.1f}km)")
                
                result = find_nearest_charging_station(
                    self.graph, current_node, driver.get_current_path(), 
                    remaining_range, connector_type
                )
                
                if result != (None, None):
                    alt_node, alt_station_id = result
                    travel_success = yield from travel_to_charging_station(
                        self.env, car_id, current_node, alt_node, self.graph, driver
                    )
                    if travel_success:
                        return (yield from charge_at_station_with_queue_tolerance(
                            self.env, car_id, alt_node, alt_station_id, self.graph, self.stats, driver,
                            simulation=self
                        ))
            else:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Insufficient range to search for compatible working stations elsewhere")

            self.stats['connector_incompatibility_failures'] += 1
            return False

        # Case 3: Working compatible stations exist at current node
        print(f"[T={self.env.now:.1f}] Car {car_id}: Found {len(stations_with_working_connections)} working compatible stations at current node")

        # Find the best working station at current location
        compatible_stations = []
        for station in stations_with_working_connections:
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
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected best working station at current location: {best_station_id}")

        # Use the unified charging function
        return (yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, current_node, best_station_id, self.graph, self.stats, driver,
            simulation=self
        ))

    
    def spawn_multiple_cars(self, total_cars, simulation_duration_hours=24):
        """
        Spawn cars gradually over a time period according to hourly distribution
        """
        # Hourly charging demand distribution (T1-T24)
        hourly_distribution = [
            0.4, 0.3, 0.3, 1.0, 3.0, 6.0, 9.5, 11.0, 12.5, 11.8, 8.8, 7.2,
            6.0, 5.0, 4.0, 3.0, 2.9, 2.5, 1.8, 1.2, 0.7, 0.5, 0.3, 0.3
        ]
        
        # Convert simulation hours to simulation time units 
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
        self.env.process(self.spawn_multiple_cars(total_cars=10000, simulation_duration_hours=24)) #CARSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until completion ---")
            self.env.run(until=1440) # Run for 24 hrs
        
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

        # Hourly queue time analysis
        print("\n" + "="*60)
        print("HOURLY QUEUE TIME ANALYSIS")
        print("="*60)
        
        self.queue_tracker.print_hourly_summary()
        
        # Create the visualization
        print("\nGenerating queue time visualization...")
        #self.queue_tracker.plot_hourly_queue_times("EV Charging Queue Times Throughout the Day")
        
        print("\n=== SoC Data Analysis ===")
        self.queue_tracker.export_soc_data_to_csv("simulation_soc_data.csv")
        
        # Print SoC summary
        (hours, avg_soc, std_soc, median_soc, q1_soc, q3_soc, counts) = self.queue_tracker.calculate_hourly_soc_statistics()
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
    stations_json_path = "data/TargetedWeightedFailures.json"
    
    random.seed(1)
    np.random.seed(1)
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
        
        # Access raw data if needed
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