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
    
    # Getters
    def get_source_node(self):
        return self.source_node
    
    def get_destination_node(self):
        return self.destination_node
    
    def get_state_of_charge(self):
        return self.state_of_charge
    
    def get_connector_type(self):
        return self.connector_type
    
    # Setters
    def set_source_node(self, source_node):
        self.source_node = source_node
    
    def set_destination_node(self, destination_node):
        self.destination_node = destination_node
    
    def set_state_of_charge(self, state_of_charge):
        self.state_of_charge = state_of_charge
    
    def set_connector_type(self, connector_type):
        self.connector_type = connector_type
    
    def __str__(self):
        return (f"EVDriver(Source: {self.source_node}, "
                f"Destination: {self.destination_node}, "
                f"SoC: {self.state_of_charge:.2f}, "
                f"Connector: {self.connector_type})")
    
    def __repr__(self):
        return self.__str__()