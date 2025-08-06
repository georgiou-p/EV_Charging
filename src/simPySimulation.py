import simpy
import random
from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import (
    find_nearest_charging_station, 
    travel_to_charging_station,
    travel_to_next_node
)
from charging_utils import charge_at_station, setup_charging_resources

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
            'total_charging_events': 0,
            'total_travel_time': 0
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
        print(f"[T={self.env.now:.1f}] Car {car_id}: Starting journey from {source_node} to {destination_node} with initial battery: {initial_soc:.2f} ({initial_soc*100:.0f}%)")
        
        # Create the driver/car
        driver = EVDriver(source_node, destination_node, initial_soc, connector_type, battery_capacity_km)
        driver.set_current_path(path)
        
        # Configuration - battery range in kilometers
        print(f"[T={self.env.now:.1f}] Car {car_id}: Using pre-calculated route with {len(path)} traverses")        
        current_range_km = driver.get_range_remaining()
        print(f"[T={self.env.now:.1f}] Car {car_id}: Can travel {current_range_km:.2f}km out of {total_distance_km:.2f}km with current battery")
        
        #--------------------MAIN JOURNEY LOOP--------------------
        loop_counter = 0
        max_loops = len(path) * 2  # Safety limit to prevent infinite loops
        
        while not driver.has_reached_destination() and loop_counter < max_loops:
            loop_counter += 1
            current_node = driver.get_current_node()
            
            # Check if we can make the next move
            if not driver.can_reach_next_node(self.graph):
                print(f"[T={self.env.now:.1f}] Car {car_id}: Cannot reach next node! Charging at current node {current_node}")
                yield from charge_at_station(self.env, car_id, current_node, self.graph, self.stats, driver, target_soc=1.0)
                continue  # Restart loop after charging
            
            #Check if driver is anxious about charging (50% SoC threshold)
            needs_charging, current_range, deficit, reason = driver.needs_charging_for_journey(self.graph)
            
            if needs_charging:
                print(f"[T={self.env.now:.1f}] Car {car_id}: {reason}")
                yield from self._handle_charging_stop(car_id, driver)
                continue  # Restart loop after charging
            else:
                # Log the reason for not charging
                if "Anxious" in reason:
                    print(f"[T={self.env.now:.1f}] Car {car_id}: {reason} - continuing journey")
            
            # Travel to next node
            yield from travel_to_next_node(self.env, car_id, driver, self.graph, travel_time_per_km=0.01)
            
            # Check if we've somehow run out of battery (safety check)
            if driver.is_battery_empty():
                print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED! Battery empty at node {driver.get_current_node()}")
                return
        
        if loop_counter >= max_loops:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Stopped due to loop limit - possible infinite loop prevented")
            return
        
        # Journey completed
        print(f"[T={self.env.now:.1f}] Car {car_id}: Reached destination {destination_node}!")
        self.stats['cars_completed'] += 1
    
    def _handle_charging_stop(self, car_id, driver):
        """
        Handle the complete charging process including travel to station
        
        Args:
            car_id: Car identifier
            driver: EVDriver object
            battery_range_km: Maximum range with full battery in kilometers
        """
        current_node = driver.get_current_node()
        current_range_km = driver.get_range_remaining()
        planned_route = driver.get_current_path()  
        
        # Find best charging station considering the planned route
        charging_node = find_nearest_charging_station(
            self.graph, current_node, planned_route, current_range_km
        )
        
        if charging_node is None:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging station within range! STRANDED!")
            return
        
        # Travel to charging station if not already there
        if charging_node != current_node:
            yield from travel_to_charging_station(self.env, car_id, current_node, charging_node, self.graph, driver)
        
        # Charge at the station
        yield from charge_at_station(self.env, car_id, charging_node, self.graph, self.stats, driver, target_soc=1.0)
        # Note: driver's battery level is updated inside charge_at_station function
        
        # Recalculate path from charging station to destination
        print(f"[T={self.env.now:.1f}] Car {car_id}: Recalculating route from charging station {charging_node} to destination {driver.get_destination_node()}")
        driver.set_source_node(charging_node)
        new_path = driver.find_shortest_path(self.graph)
        if not new_path:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No path from charging station!")
            return
        
        new_distance_km = driver._calculate_path_distance(self.graph, new_path)
        print(f"[T={self.env.now:.1f}] Car {car_id}: New route from charging station: {len(new_path)} stops, {new_distance_km:.2f}km remaining")
    
    def spawn_multiple_cars(self, num_cars):
        """Spawn multiple cars for testing"""
        all_nodes = list(self.graph.nodes)
        battery_capacity_km = 300  # Maximum km car can travel with full battery
        
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
                        # Accept routes between 100km and 550km (reasonable range)
                        if 100 <= total_distance_km <= 550:
                            break
                        else:print(f"We need a trip between 100 to 550Km. Rerandomising...")
         
            initial_soc = 0.8
            connector_type = random.choice([10, 20, 30])
        
            print(f"Spawning car {car_id} from {source} to {destination} with battery capacity {(battery_capacity_km)}Km, SoC {initial_soc:.2f}, and Connector: {connector_type}")
        
            self.env.process(self.car_process(car_id, path, initial_soc, connector_type, total_distance_km, battery_capacity_km))
            self.stats['cars_spawned'] += 1
    
    
    
    def run_simulation(self):
        """Run the simulation"""
        print("=== Starting EV Simulation ===")
        print(f"Graph has {len(self.graph.nodes)} nodes")
        
        # Spawn cars
        self.spawn_multiple_cars(num_cars=100)  #NUMBER OF CARS
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until all cars complete their journeys ---")
            self.env.run()  # Run until all processes finish
        
        # Print final statistics
        print("FINAL STATISTICS") 
        print("\n=== Simulation Complete ===")
        print(f"Cars spawned: {self.stats['cars_spawned']}")
        print(f"Cars completed journey: {self.stats['cars_completed']}")
        print(f"Total charging events: {self.stats['total_charging_events']}")
        
        if self.stats['cars_completed'] == 0 and self.stats['cars_spawned'] > 0:
            print("No cars completed their journey - they may be stranded or need more time!")

def main():

    print("Loading graph and charging stations...")
    
    # Loads data
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    stations_json_path = "data/cleanedM_charging_stations.json"
    
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
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()