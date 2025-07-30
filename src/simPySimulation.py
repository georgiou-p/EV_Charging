import simpy
import random
from station_assignment import assign_charging_stations_to_nodes
from evDriver import EVDriver
from pathfinding import (
    find_nearest_charging_station, 
    calculate_charging_needed, 
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
    
    def car_process(self, car_id, source_node, destination_node, initial_soc, connector_type):
        """
        SimPy process representing one car's journey - High level orchestration only
        """
        print(f"[T={self.env.now:.1f}] Car {car_id}: Starting journey from {source_node} to {destination_node}")
        print(f"[T={self.env.now:.1f}] Car {car_id}: Initial battery: {initial_soc:.2f} ({initial_soc*100:.0f}%)")
        
        # Create the driver/car
        driver = EVDriver(source_node, destination_node, initial_soc, connector_type)
        
        # Find the route
        print(f"[T={self.env.now:.1f}] Car {car_id}: Planning initial route...")
        path = driver.find_shortest_path(self.graph)
        if not path:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No path found!")
            return
        
        # Configuration
        battery_range = 50  # Maximum nodes car can travel with full battery
        
        print(f"[T={self.env.now:.1f}] Car {car_id}: Route planned with {len(path)} stops")
        print(f"[T={self.env.now:.1f}] Car {car_id}: Total distance: {len(path)-1} hops")
        
        total_distance = len(path) - 1
        print(f"[T={self.env.now:.1f}] Car {car_id}: Can travel {driver.get_range_remaining(battery_range)} hops with current battery")
        print(f"[T={self.env.now:.1f}] Car {car_id}: Need to travel {total_distance} hops total")
        
        # Check if we need charging before even starting
        remaining_distance = driver.get_remaining_distance(self.graph)
        needs_charging, current_range, deficit = calculate_charging_needed(
            driver.get_state_of_charge(), battery_range, remaining_distance
        )
        
        if needs_charging:
            print(f"[T={self.env.now:.1f}] Car {car_id}: Need charging! Current range ({current_range}) < total distance ({total_distance})")
            yield from self._handle_charging_stop(car_id, driver, battery_range)
        
        # Main journey loop - high level orchestration
        while not driver.has_reached_destination():
            current_node = driver.get_current_node()
            
            # Check if charging needed before next move
            remaining_distance = driver.get_remaining_distance(self.graph)
            needs_charging, current_range, deficit = calculate_charging_needed(
                driver.get_state_of_charge(), battery_range, remaining_distance
            )
            
            if needs_charging:
                print(f"[T={self.env.now:.1f}] Car {car_id}: Need charging! Range({current_range}) < Distance({remaining_distance})")
                yield from self._handle_charging_stop(car_id, driver, battery_range)
                continue  # Restart loop after charging
            
            # Travel to next node
            yield from travel_to_next_node(self.env, car_id, driver, travel_time_per_hop=1.0, battery_range=battery_range)
            
            # Check if we've somehow run out of battery (safety check)
            if driver.is_battery_empty():
                print(f"[T={self.env.now:.1f}] Car {car_id}: STRANDED! Battery empty at node {driver.get_current_node()}")
                return
        
        # Journey completed
        print(f"[T={self.env.now:.1f}] Car {car_id}: Reached destination {destination_node}!")
        self.stats['cars_completed'] += 1
    
    def _handle_charging_stop(self, car_id, driver, battery_range):
        """
        Handle the complete charging process including travel to station
        
        Args:
            car_id: Car identifier
            driver: EVDriver object
            battery_range: Maximum range with full battery
        """
        current_node = driver.get_current_node()
        current_range = driver.get_range_remaining(battery_range)
        planned_route = driver.get_current_path()  
        
        # Find best charging station considering the planned route
        charging_node = find_nearest_charging_station(
            self.graph, current_node, planned_route, current_range
        )
        
        if charging_node is None:
            print(f"[T={self.env.now:.1f}] Car {car_id}: No charging station within range! STRANDED!")
            return
        
        # Travel to charging station if not already there
        if charging_node != current_node:
            yield from travel_to_charging_station(self.env, car_id, current_node, charging_node, self.graph)
            driver.consume_battery(0.02)  # Small battery consumption for detour
        
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
        
        print(f"[T={self.env.now:.1f}] Car {car_id}: New route from charging station: {len(new_path)} stops, {len(new_path)-1} hops remaining")
    
    def spawn_multiple_cars(self, num_cars):
        """Spawn multiple cars for testing"""
        all_nodes = list(self.graph.nodes)

        for car_id in range(1, num_cars + 1):
            # Pick random source and destination
            while True:
                source = random.choice(all_nodes)
                destination = random.choice(all_nodes)
        
                if source != destination:
                    test_driver = EVDriver(source, destination, 1.0, 20)
                    path = test_driver.find_shortest_path(self.graph)
                    if path and 10 <= len(path) <= 30:
                     break
    
            # These lines should be INSIDE the for loop
            initial_soc = random.uniform(0.1, 0.3)
            connector_type = random.choice([10, 20, 30])
        
            print(f"Spawning car {car_id} from {source} to {destination} with SoC {initial_soc:.2f}")
        
            self.env.process(self.car_process(car_id, source, destination, initial_soc, connector_type))
            self.stats['cars_spawned'] += 1
    
    def run_simulation(self):
        """Run the simulation"""
        print("=== Starting Simple SimPy EV Simulation ===")
        print(f"Graph has {len(self.graph.nodes)} nodes")
        
        # Spawn one car
        self.spawn_multiple_cars(num_cars=50)
        
        # Run simulation
        if self.simulation_time:
            print(f"\n--- Running simulation for {self.simulation_time} time units ---")
            self.env.run(until=self.simulation_time)
        else:
            print(f"\n--- Running simulation until all cars complete their journeys ---")
            self.env.run()  # Run until all processes finish
        
        # Print final statistics
        print("\n=== Simulation Complete ===")
        print(f"Cars spawned: {self.stats['cars_spawned']}")
        print(f"Cars completed journey: {self.stats['cars_completed']}")
        print(f"Total charging events: {self.stats['total_charging_events']}")
        
        if self.stats['cars_completed'] == 0 and self.stats['cars_spawned'] > 0:
            print("No cars completed their journey - they may be stranded or need more time!")

def main():
    """Main function to run the simple simulation"""
    print("Loading graph and charging stations...")
    
    # Load your existing data
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