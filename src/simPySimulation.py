import simpy
import random
import cProfile
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
        Simple SimPy simulation using SimPy's built-in queuing with connector compatibility
        
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
            'connector_incompatibility_failures': 0
        }
    
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
        
        # Configuration - battery range in kilometers
        current_range_km = driver.get_range_remaining()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Can travel {current_range_km:.2f}km out of {total_distance_km:.2f}km with current battery")
        
        #--------------------MAIN JOURNEY LOOP----------
        loop_counter = 0
        max_loops = len(path) * 2  # Safety limit to prevent infinite loops
        
        while not driver.has_reached_destination() and loop_counter < max_loops:
            loop_counter += 1
            current_node = driver.get_current_node()
            
            # Check if we can make the next move
            if not driver.can_reach_next_node(self.graph):
                print(f"[T={self.env.now:.1f}] Car {car_id}: Cannot reach next node! Charging at current node {current_node}")
                success = yield from self._charge_at_current_location(car_id, driver, current_node)
                if not success:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED at {current_node} - no compatible charging!")
                    self.stats['cars_stranded'] += 1
                    return
                continue  # Restart loop after charging
            
            # Check if driver is anxious about charging (50% SoC threshold)
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
            yield from travel_to_next_node(self.env, car_id, driver, self.graph, travel_time_per_km=0.01)
            
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
    
    
    def _handle_charging_stop(self, car_id, driver, max_wait_minutes=20):
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
            self.env, car_id, target_node, target_station_id, self.graph, self.stats, driver, max_wait_minutes
        )

        # Step 4: Handle charging failure
        if not charging_success:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Charging failed at node {target_node}")
            print(f"[T={self.env.now:.1f}] Car {car_id}: No acceptable charging options found")
            # Note: Enhanced fallback could search for other locations here
            return False

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
        best_station = None
        best_score = -1

        for station in stations:
            if has_compatible_connector([station], connector_type):
                # Score this station
                score = 1000

                # Queue penalty
                queue_length = len(station.simpy_resource.queue) if hasattr(station, 'simpy_resource') else 0
                estimated_wait_time = queue_length * 10
                score -= estimated_wait_time * 10

                # Power bonus
                max_power = get_station_max_power(station, connector_type)
                if max_power >= 150:
                    score += 200
                elif max_power >= 50:
                    score += 100
                elif max_power >= 22:
                    score += 50
                else:
                    score += 20

                if score > best_score:
                    best_score = score
                    best_station = station

        if not best_station:
            return False

        best_station_id = best_station.get_station_id()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Selected best station at current location: {best_station_id}")

        # Use the queue tolerance charging function
        charging_success = yield from charge_at_station_with_queue_tolerance(
            self.env, car_id, current_node, best_station_id, self.graph, self.stats, driver, max_wait_minutes=20
        )

        return charging_success
    
    
    
    def spawn_multiple_cars(self, num_cars):
        """Spawn multiple cars """
        all_nodes = list(self.graph.nodes)
        battery_capacity_km = 300  # Maximum km car can travel with full battery
        
        # Define connector type distribution (based on real UK EV market)
        connector_types = [1, 2, 3]  # Type 1, Type 2/3pin, CCS/CHAdeMO
        connector_weights = [0.2, 0.7, 0.1]  # Type 2 most common in UK
        
        for car_id in range(1, num_cars + 1):
            # Pick random source and destination
            while True:
                source = random.choice(all_nodes)
                destination = random.choice(all_nodes)
        
                if source != destination:
                    test_driver = EVDriver(source, destination, 1.0, 20, battery_capacity_km)
                    path = test_driver.find_shortest_path(self.graph)
                    if path:
                        # Check path distance 
                        total_distance_km = test_driver._calculate_path_distance(self.graph, path)
                        # Accept routes between 100km and 550km 
                        if 100 <= total_distance_km <= 550:
                            break
                        else:
                            print(f"We need a trip between 100 to 550Km. Rerandomising...")
         
            initial_soc = random.uniform(0.6, 0.9)  # Start with 60-90% battery
            connector_type = random.choices(connector_types, weights=connector_weights)[0]
        
            print(f"Spawning car {car_id} from {source} to {destination} with battery capacity {battery_capacity_km}Km, SoC {initial_soc:.2f}, and Connector: {connector_type}")
        
            self.env.process(self.car_process(car_id, path, initial_soc, connector_type, total_distance_km, battery_capacity_km))
            self.stats['cars_spawned'] += 1
    
    def run_simulation(self):
        """Run the simulation"""
        print("=== Starting EV Simulation===")
        print(f"Graph has {len(self.graph.nodes)} nodes")
        
        # Spawn cars
        self.spawn_multiple_cars(num_cars=1000)  # NUMBER OF CARSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until all cars complete their journeys ---")
            self.env.run()  # Run until all processes finish
        
        # Print final statistics
        print("\n=== Simulation Complete ===")
        print("FINAL STATISTICS:")
        print(f"Cars spawned: {self.stats['cars_spawned']}")
        print(f"Cars completed journey: {self.stats['cars_completed']}")
        print(f"Cars stranded: {self.stats['cars_stranded']}")
        print(f"Total charging events: {self.stats['total_charging_events']}")
        print(f"Connector incompatibility failures: {self.stats['connector_incompatibility_failures']}")
        
        # Calculate success rate
        if self.stats['cars_spawned'] > 0:
            success_rate = (self.stats['cars_completed'] / self.stats['cars_spawned']) * 100
            print(f"Journey completion rate: {success_rate:.1f}%")
            
            if self.stats['connector_incompatibility_failures'] > 0:
                incompatibility_rate = (self.stats['connector_incompatibility_failures'] / self.stats['cars_spawned']) * 100
                print(f"Connector incompatibility rate: {incompatibility_rate:.1f}%")
        
        if self.stats['cars_completed'] == 0 and self.stats['cars_spawned'] > 0:
            print("No cars completed their journey - they may be stranded or need more time!")



def main():
    print("Loading graph and charging stations...")
    
    # Loads data
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleaned_charging_stations.json"
    
    try:
        graph, node_stations = assign_charging_stations_to_nodes(
            geojson_path, 
            stations_json_path
        )
        
        if graph is None:
            print("Failed to load graph!")
            return
        
        print(f"Graph loaded successfully with {len(graph.nodes)} nodes")
        
        # Analyze connector type distribution in charging stations
        connector_stats = {}
        total_stations = 0
        for node in graph.nodes:
            stations = graph.nodes[node]['charging_stations']
            for station in stations:
                total_stations += 1
                for connection in station.get_connections():
                    conn_type = connection.connection_type_id
                    if conn_type == 0:
                        conn_type = "Universal"
                    connector_stats[conn_type] = connector_stats.get(conn_type, 0) + 1
        
        print("\nCharging station connector distribution:")
        # Sort with custom key to handle mixed int/string types
        sorted_items = sorted(connector_stats.items(), key=lambda x: (isinstance(x[0], str), x[0]))
        for conn_type, count in sorted_items:
            percentage = (count / sum(connector_stats.values())) * 100
            print(f"  Connector type {conn_type}: {count} connections ({percentage:.1f}%)")
        
        # Create and run simulation
        simulation = SimpleEVSimulation(graph, simulation_time=None)  # No time limit
        simulation.run_simulation()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()