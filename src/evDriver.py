import networkx as nx

class EVDriver:
    def __init__(self, source_node, destination_node, state_of_charge, connector_type, battery_capacity_km):
        """
        Initialize an EV Driver
        
        Args:
            source_node: Starting node ID on the graph
            destination_node: Target node ID on the graph  
            state_of_charge: Current battery charge level (0.0 to 1.0)
            connector_type: Type of connector the vehicle uses
        """
        self.source_node = source_node
        self.destination_node = destination_node
        self.state_of_charge = state_of_charge
        self.connector_type = connector_type
        self.current_path = []
        self.current_position_index = 0
        self.battery_capacity_km = battery_capacity_km
        self.charging_anxiety_threshold = 0.5  # 50% SoC anxiety threshold
        self.max_acceptable_wait_minutes = 20
        self.energy_consumption_kwh_per_km = 0.25 #Typical average
        self.battery_capacity_kwh = battery_capacity_km * self.energy_consumption_kwh_per_km
    
    # Getters
    def get_source_node(self):
        return self.source_node
    
    def get_destination_node(self):
        return self.destination_node
    
    def get_state_of_charge(self):
        return self.state_of_charge
    
    def get_connector_type(self):
        return self.connector_type
    
    def get_current_path(self):
        return self.current_path
    
    def get_current_position_index(self):
        return self.current_position_index
    
    def get_charging_anxiety_threshold(self):
        return self.charging_anxiety_threshold
    
    def get_battery_capacity(self):
        return self.battery_capacity_km
    
    def get_max_acceptable_wait(self):
        return self.max_acceptable_wait_minutes
    
    def get_battery_capacity_kwh(self):
        return self.battery_capacity_kwh
    
    # Setters
    def set_source_node(self, source_node):
        self.source_node = source_node
    
    def set_destination_node(self, destination_node):
        self.destination_node = destination_node
    
    def set_state_of_charge(self, state_of_charge):
        self.state_of_charge = state_of_charge
    
    def set_connector_type(self, connector_type):
        self.connector_type = connector_type
    
    def set_current_path(self, path):
        self.current_path = path
        self.current_position_index = 0
    
    def set_current_position_index(self, index):
        self.current_position_index = index

    def set_battery_capacity_km(self, battery_capacity_km):
        self.battery_capacity_km = battery_capacity_km
    
    def set_charging_anxiety_threshold(self, threshold):
        """Set the SoC threshold at which driver becomes anxious about charging"""
        self.charging_anxiety_threshold = max(0.0, min(1.0, threshold))

    def set_max_acceptable_wait(self, minutes):
        self.max_acceptable_wait_minutes = minutes

    def set_battery_capacity_kwh(self, capacity_kwh):
        """Set battery capacity in kWh and update km range accordingly"""
        self.battery_capacity_kwh = capacity_kwh
        self.battery_capacity_km = capacity_kwh / self.energy_consumption_kwh_per_km

    def set_energy_consumption(self, kwh_per_km):
        """Update energy consumption and recalculate capacities"""
        self.energy_consumption_kwh_per_km = kwh_per_km
        # Keep kWh as primary, recalculate km
        self.battery_capacity_km = self.battery_capacity_kwh / kwh_per_km
    
    # Path traversal methods
    def find_shortest_path(self, graph):
        """Find the shortest path from source to destination using NetworkX with weighted edges"""
        try:
            path = nx.shortest_path(graph, self.source_node, self.destination_node, weight='weight')
            self.current_path = path
            self.current_position_index = 0
            if path:
                # Calculate total distance in km
                total_distance_km = self._calculate_path_distance(graph, path)
                print(f"    Route calculated from {self.source_node} to {self.destination_node}:")
                print(f"    Full path: {path[:10]}{'...' if len(path) > 10 else ''}")
                print(f"    Total distance: {total_distance_km:.2f}km through {len(path)} nodes")
            return path
        except nx.NetworkXNoPath:
            print(f"No path found from {self.source_node} to {self.destination_node}")
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found in graph: {e}")
            return None
    
    def _calculate_path_distance(self, graph, path):
        """Calculate total distance in km for a given path"""
        total_distance = 0.0
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            if graph.has_edge(node1, node2):
                total_distance += graph.edges[node1, node2]['weight']
        return total_distance
    
    def get_current_node(self):
        """Get the current node in the path"""
        if self.current_path and 0 <= self.current_position_index < len(self.current_path):
            return self.current_path[self.current_position_index]
        return None
    
    def move_to_next_node(self):
        """Move to the next node in the path"""
        if self.current_path and self.current_position_index < len(self.current_path) - 1:
            self.current_position_index += 1
            return self.get_current_node()
        return None
    
    def has_reached_destination(self):
        """Check if the driver has reached the destination"""
        if not self.current_path:
            return False
        return (self.current_position_index == len(self.current_path) - 1 and 
                self.get_current_node() == self.destination_node)
    
    def get_remaining_distance(self, graph):
        """Get the remaining distance to destination in kilometers"""
        if not self.current_path or self.current_position_index >= len(self.current_path):
            return 0.0
        
        current_node = self.get_current_node()
        if current_node == self.destination_node:
            return 0.0
        
        try:
            return nx.shortest_path_length(graph, current_node, self.destination_node, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')
    
    def can_reach_next_node(self, graph):
        """
        Check if the car can reach the next node in its path
            
        Returns:
            bool: True if car can reach next node, False otherwise
        """
        current_node = self.get_current_node()
        current_path = self.get_current_path()
        current_index = self.get_current_position_index()
        
        if self.has_reached_destination():
            return True
        # If no path or at end of path, return True (no next node to reach)
        if not current_path or current_index >= len(current_path) - 1:
            return True
        
        next_node = current_path[current_index + 1]
        
        # Check if edge exists and get distance
        if graph.has_edge(current_node, next_node):
            edge_distance_km = graph.edges[current_node, next_node]['weight']
            current_range_km = self.get_range_remaining()
            
            # Add small safety margin
            required_range_km = edge_distance_km * 1.1  # 10% safety margin
            
            return current_range_km >= required_range_km
        
        return False  # No edge exists
    
    def is_anxious_about_charging(self):
        """
        Check if driver is anxious about charging (SoC below anxiety threshold)
        
        Returns:
            bool: True if driver is anxious about charging (SoC <= 50%)
        """
        return self.state_of_charge <= self.charging_anxiety_threshold
    
    def should_look_for_charging(self, graph):
        """
        Determine if driver should start looking for charging based on anxiety threshold
        and remaining journey requirements
        
        Args:
            graph: NetworkX graph
            battery_range_km: Maximum range with full battery
            
        Returns:
            tuple: (should_charge: bool, reason: str)
        """
        current_soc = self.get_state_of_charge()
        
        # Critical battery - must charge immediately
        if current_soc <= 0.1:  # 10% or less
            return True, f"Critical battery level: {current_soc*100:.0f}%"
        
        # Can't reach next node - must charge immediately
        if not self.can_reach_next_node(graph):
            return True, "Cannot reach next node"
        
        # Driver anxiety kicks in at 50% SoC
        if self.is_anxious_about_charging():
            remaining_distance_km = self.get_remaining_distance(graph)
            current_range_km = self.get_range_remaining()
            
            # If anxious AND can't complete journey, look for charging
            if remaining_distance_km > current_range_km:
                return True, f"Anxious at {current_soc*100:.0f}% SoC and cannot complete journey (need {remaining_distance_km:.1f}km, have {current_range_km:.1f}km range)"
            
            # If anxious but can complete journey, mention anxiety but don't charge yet
            return False, f"Anxious at {current_soc*100:.0f}% SoC but can complete journey (need {remaining_distance_km:.1f}km, have {current_range_km:.1f}km range)"
        
        # Above anxiety threshold - continue driving
        return False, f"Comfortable at {current_soc*100:.0f}% SoC (above {self.charging_anxiety_threshold*100:.0f}% anxiety threshold)"
    
    def needs_charging_for_journey(self, graph):
        """
        Check if charging is needed to complete the remaining journey
        
        Args:
            graph: NetworkX graph
            battery_range_km: Maximum range with full battery
            
        Returns:
            tuple: (needs_charging: bool, current_range: float, deficit: float, reason: str)
        """
        current_range_km = self.get_range_remaining()
        remaining_distance_km = self.get_remaining_distance(graph)
        
        # Check if driver should look for charging based on anxiety + journey requirements
        should_charge, reason = self.should_look_for_charging(graph)
        
        deficit = max(0, remaining_distance_km - current_range_km)
        
        return should_charge, current_range_km, deficit, reason
    
    # Battery management methods
    def consume_battery(self, consumption):
        """Consume battery energy"""
        self.state_of_charge -= consumption
        self.state_of_charge = max(0.0, self.state_of_charge)

    def get_range_remaining(self):
        """Get remaining travel range in kilometers"""
        return self.battery_capacity_km * self.state_of_charge

    @property
    def battery_percentage(self):
        """Get battery level as percentage for display"""
        return self.state_of_charge * 100
    
    def set_battery_level(self, new_soc):
        """
        Set battery to a specific level (e.g., after charging)
        
        Args:
            new_soc: New state of charge (0.0 to 1.0)
        """
        self.state_of_charge = max(0.0, min(1.0, new_soc))  # Clamp between 0 and 1
    
    def is_battery_empty(self):
        """Check if battery is empty"""
        return self.state_of_charge <= 0.0
    
    def __str__(self):
        return (f"EVDriver(Source: {self.source_node}, "
                f"Destination: {self.destination_node}, "
                f"Capacity: {self.battery_capacity_km}km, "
                f"SoC: {self.state_of_charge:.2f}, "
                f"Connector: {self.connector_type}, "
                f"Current: {self.get_current_node()})")
    
    def __repr__(self):
        return self.__str__()