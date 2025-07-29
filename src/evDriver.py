import networkx as nx

class EVDriver:
    def __init__(self, source_node, destination_node, state_of_charge, connector_type):
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
    
    # Path traversal methods
    def find_shortest_path(self, graph):
        """Find the shortest path from source to destination using NetworkX"""
        try:
            path = nx.shortest_path(graph, self.source_node, self.destination_node)
            self.current_path = path
            self.current_position_index = 0
            if path:
                print(f"    Route calculated from {self.source_node} to {self.destination_node}:")
                print(f"    Full path: {path[:10]}{'...' if len(path) > 10 else ''}")
                print(f"    Total distance: {len(path)-1} hops through {len(path)} nodes")
            return path
        except nx.NetworkXNoPath:
            print(f"No path found from {self.source_node} to {self.destination_node}")
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found in graph: {e}")
            return None
    
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
        """Get the remaining distance to destination"""
        if not self.current_path or self.current_position_index >= len(self.current_path):
            return 0
        
        current_node = self.get_current_node()
        if current_node == self.destination_node:
            return 0
        
        try:
            return nx.shortest_path_length(graph, current_node, self.destination_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')
    
    # Battery management methods
    # Simplified battery methods:
    def consume_battery(self, consumption):
     """Consume battery energy"""
     self.state_of_charge -= consumption
     self.state_of_charge = max(0.0, self.state_of_charge)

    def get_range_remaining(self, max_range):
     """Get remaining travel range in hops"""
     return int(max_range * self.state_of_charge)

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
                f"SoC: {self.state_of_charge:.2f}, "
                f"Connector: {self.connector_type}, "
                f"Current: {self.get_current_node()})")
    
    def __repr__(self):
        return self.__str__()