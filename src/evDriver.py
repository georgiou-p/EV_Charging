import networkx as nx
from anxiety import Anxiety

class EVDriver:
    def __init__(self, source_node, destination_node, state_of_charge, connector_type, battery_capacity_km):
        """
        Initialize an EV Driver with anxiety model that increases below threshold
        """
        self.source_node = source_node
        self.destination_node = destination_node
        self.state_of_charge = state_of_charge
        self.connector_type = connector_type
        self.current_path = []
        self.current_position_index = 0
        self.battery_capacity_km = battery_capacity_km
        # Anxiety model
        self.anxiety_model = Anxiety()
        # Track penalties
        self.current_penalties = {
            'queue_time': 0.0,
            'slow_charger_time': 0.0,
            'threshold_penalty': 0.0  #Penalty for being below threshold
        }
        
        # Energy and battery settings
        self.energy_consumption_kwh_per_km = 0.25
        self.battery_capacity_kwh = battery_capacity_km * self.energy_consumption_kwh_per_km
        
        #Threshod anxiety tracking
        self.last_soc_check = state_of_charge
        self.threshold_anxiety_rate = 2.0  # Minutes of penalty per 1% below threshold

        # Log driver profile
        print(f"[Driver] Created: {self.anxiety_model.get_profile_description()}")
    
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
        return self.anxiety_model.get_threshold()
    
    def get_battery_capacity(self):
        return self.battery_capacity_km
    
    def get_battery_capacity_kwh(self):
        return self.battery_capacity_kwh
    
    def get_anxiety_model(self):
        return self.anxiety_model
    
    def get_current_anxiety(self):
        """Compute current anxiety based on accumulated penalties"""
        total_penalties = sum(self.current_penalties.values())
        anxiety = self.anxiety_model.compute(total_penalties)
        return anxiety
    
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
    
    def set_battery_capacity_kwh(self, capacity_kwh):
        self.battery_capacity_kwh = capacity_kwh
        self.battery_capacity_km = capacity_kwh / self.energy_consumption_kwh_per_km

    def set_energy_consumption(self, kwh_per_km):
        self.energy_consumption_kwh_per_km = kwh_per_km
        self.battery_capacity_km = self.battery_capacity_kwh / kwh_per_km
    
    def update_threshold_anxiety(self, charging_relief_factor=0.0):
        """
        Update anxiety penalty based on how far below threshold the driver is
        Called whenever SoC changes

        Args:
            charging_relief_factor: Factor to reduce threshold anxiety when charging (0.0 = no relief, 0.75 = 75% relief)
        """
        current_soc = self.get_state_of_charge()
        threshold = self.anxiety_model.get_threshold()
        old_penalty = self.current_penalties['threshold_penalty']

        if current_soc < threshold:
            # Driver is below their personal threshold - calculate new penalty
            below_threshold_pct = (threshold - current_soc) * 100
            base_penalty = below_threshold_pct * self.threshold_anxiety_rate

            # Apply charging relief if provided
            if charging_relief_factor > 0.0:
                final_penalty = base_penalty * (1.0 - charging_relief_factor)
                relief_msg = f" (with {charging_relief_factor*100:.0f}% charging relief)"
            else:
                final_penalty = base_penalty
                relief_msg = ""

            # Set the new penalty
            self.current_penalties['threshold_penalty'] = final_penalty

            # Log significant changes
            if abs(final_penalty - old_penalty) > 2.0:
                anxiety = self.get_current_anxiety()
                print(f"[Driver] Threshold anxiety: SoC {current_soc*100:.0f}% < {threshold*100:.0f}% threshold, "
                      f"penalty: {final_penalty:.1f}min{relief_msg}, total anxiety: {anxiety:.3f}")

        elif charging_relief_factor > 0.0:
            # Driver is above threshold AND charging relief is being applied
            if old_penalty > 0:
                # Apply relief to existing threshold penalty from previous low battery experience
                relief_penalty = old_penalty * (1.0 - charging_relief_factor)
                self.current_penalties['threshold_penalty'] = relief_penalty

                anxiety = self.get_current_anxiety()
                print(f"[Driver] Above threshold with charging relief: SoC {current_soc*100:.0f}% >= {threshold*100:.0f}%, "
                      f"threshold penalty reduced from {old_penalty:.1f}min to {relief_penalty:.1f}min "
                      f"({charging_relief_factor*100:.0f}% relief), anxiety: {anxiety:.3f}")
            else:
                # No existing penalty to apply relief to - keep at zero
                self.current_penalties['threshold_penalty'] = 0.0
    
    def set_threshold_anxiety_rate(self, rate):
        """Set how much anxiety increases per percentage point below threshold"""
        self.threshold_anxiety_rate = rate
    
    # Existing penalty tracking methods 
    def add_queue_penalty(self, queue_time_minutes):
        self.current_penalties['queue_time'] += queue_time_minutes
        anxiety = self.get_current_anxiety()
        print(f"[Driver] Queue penalty: +{queue_time_minutes:.1f}min, Total penalties: {sum(self.current_penalties.values()):.1f}min, Anxiety: {anxiety:.3f}")
    
    def add_slow_charger_penalty(self, extra_charging_time_minutes):
        # Add time-based penalty multiplier
        if extra_charging_time_minutes < 15:        # Small delay
            penalty_multiplier = 1.0
        elif extra_charging_time_minutes < 30:      # Moderate delay
            penalty_multiplier = 1.5
        elif extra_charging_time_minutes < 60:      # Significant delay
            penalty_multiplier = 2.0
        elif extra_charging_time_minutes < 120:     # Long delay
            penalty_multiplier = 2.5
        else:                                       # Very long delay
            penalty_multiplier = 3.0

        # Apply multiplier and add to penalties
        adjusted_penalty = extra_charging_time_minutes * penalty_multiplier
        self.current_penalties['slow_charger_time'] += adjusted_penalty

        anxiety = self.get_current_anxiety()
        print(f"[Driver] Slow charger penalty: +{extra_charging_time_minutes:.1f}min x{penalty_multiplier:.1f} = {adjusted_penalty:.1f}min, Total penalties: {sum(self.current_penalties.values()):.1f}min, Anxiety: {anxiety:.3f}")
    
    def decay_penalties(self, decay_factor=0.95):
        """Gradually reduce penalties over time (except threshold penalty)"""
        old_total = sum(self.current_penalties.values())
        
        # Decay everything except threshold penalty
        for penalty_type in self.current_penalties:
            if penalty_type != 'threshold_penalty':
                self.current_penalties[penalty_type] *= decay_factor
        
        new_total = sum(self.current_penalties.values())
        
        if abs(old_total - new_total) > 0.1:
            anxiety = self.get_current_anxiety()
            print(f"[Driver] Penalties decayed: {old_total:.1f} -> {new_total:.1f}min, Anxiety: {anxiety:.3f}")
    
    def consume_battery(self, consumption):
        """Consume battery energy and update threshold anxiety"""
        old_soc = self.state_of_charge
        self.state_of_charge -= consumption
        self.state_of_charge = max(0.0, self.state_of_charge)
        
        # Update threshold-based anxiety whenever SoC changes
        self.update_threshold_anxiety(charging_relief_factor=0.0)
    
    # def set_battery_level(self, new_soc):
    #     """Set battery to a specific level and update threshold anxiety"""
    #     old_soc = self.state_of_charge
        
    #     # Calculate charging relief BEFORE updating anything
    #     relief_factor = 0.0
    #     if new_soc > old_soc and abs(new_soc - old_soc) > 0.05:  # Charging occurred
    #         charge_amount = new_soc - old_soc
    #         relief_factor = min(0.75, charge_amount * 2.5)  # Cap at 75% relief
            
    #         # Store current anxiety BEFORE any changes
    #         old_total_anxiety = self.get_current_anxiety()
            
    #         # Apply relief to accumulated penalties (queue and slow charger)
    #         if relief_factor > 0.1:  # Only apply relief if >10%
    #             self.current_penalties['queue_time'] *= (1.0 - relief_factor)
    #             self.current_penalties['slow_charger_time'] *= (1.0 - relief_factor)
        
    #     self.state_of_charge = max(0.0, min(1.0, new_soc))
        
    #     # Update threshold-based anxiety with charging relief applied
    #     if abs(new_soc - old_soc) > 0.05:  # 5% change
    #         self.update_threshold_anxiety(charging_relief_factor=relief_factor)
            
    #         # Log charging relief if it occurred
    #         if new_soc > old_soc and relief_factor > 0.1:
    #             new_total_anxiety = self.get_current_anxiety()
    #             print(f"[Driver] Charging relief: +{charge_amount*100:.0f}% SoC, {relief_factor*100:.0f}% penalty reduction ({old_total_anxiety:.3f} -> {new_total_anxiety:.3f})")
    
    # Path traversal methods
    def find_shortest_path(self, graph):
        """Find the shortest path from source to destination using NetworkX with weighted edges"""
        try:
            path = nx.shortest_path(graph, self.source_node, self.destination_node, weight='weight')
            self.current_path = path
            self.current_position_index = 0
            if path:
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
        """Check if the car can reach the next node in its path"""
        current_node = self.get_current_node()
        current_path = self.get_current_path()
        current_index = self.get_current_position_index()
        
        if self.has_reached_destination():
            return True
        if not current_path or current_index >= len(current_path) - 1:
            return True
        
        next_node = current_path[current_index + 1]
        
        if graph.has_edge(current_node, next_node):
            edge_distance_km = graph.edges[current_node, next_node]['weight']
            current_range_km = self.get_range_remaining()
            required_range_km = edge_distance_km * 1.1
            return current_range_km >= required_range_km
        
        return False
    
    def should_look_for_charging(self, graph):
        """
        Determine if driver should look for charging based on SoC threshold
        """
        current_soc = self.get_state_of_charge()
        threshold = self.anxiety_model.get_threshold()
        safety_margin = 0.10
        search_threshold = threshold + safety_margin
        
        # Critical battery - must charge immediately
        if current_soc <= 0.05:
            return True, f"Critical battery level: {current_soc*100:.0f}%"
        
        # Can't reach next node - must charge immediately
        if not self.can_reach_next_node(graph):
            return True, "Cannot reach next node"
        
        # Threshold-based search decision
        if current_soc <= search_threshold:
            anxiety = self.get_current_anxiety()
            return True, f"SoC {current_soc*100:.0f}% <= search threshold {search_threshold*100:.0f}% (base {threshold*100:.0f}% + 10% margin, anxiety: {anxiety:.3f})"
        
        # Above personal threshold - continue driving
        anxiety = self.get_current_anxiety()
        return False, f"SoC {current_soc*100:.0f}% > search threshold {threshold*100:.0f}% (anxiety: {anxiety:.3f})"

    def needs_charging_for_journey(self, graph):
        """
        Check if charging is needed using anxiety model
        """
        current_range_km = self.get_range_remaining()
        remaining_distance_km = self.get_remaining_distance(graph)
        
        should_charge, reason = self.should_look_for_charging(graph)
        
        deficit = max(0, remaining_distance_km - current_range_km)
        
        return should_charge, current_range_km, deficit, reason
    
    # Battery management methods
    def get_range_remaining(self):
        """Get remaining travel range in kilometers"""
        return self.battery_capacity_km * self.state_of_charge

    @property
    def battery_percentage(self):
        """Get battery level as percentage for display"""
        return self.state_of_charge * 100
    
    def is_battery_empty(self):
        """Check if battery is empty"""
        return self.state_of_charge <= 0.0
    
    def get_anxiety_summary(self):
        """Get comprehensive anxiety status summary"""
        current_soc = self.state_of_charge
        total_penalties = sum(self.current_penalties.values())
        anxiety = self.get_current_anxiety()
        
        return {
            'threshold_pct': self.anxiety_model.get_threshold() * 100,
            'patience_level': self.anxiety_model.get_patience_level(),
            'current_soc_pct': current_soc * 100,
            'should_search': self.anxiety_model.should_search(current_soc),
            'total_penalties_min': total_penalties,
            'anxiety_score': anxiety,
            'penalties_breakdown': self.current_penalties.copy(),
            'profile_description': self.anxiety_model.get_profile_description()
        }
    
    def __str__(self):
        anxiety_desc = self.anxiety_model.get_profile_description()
        anxiety_score = self.get_current_anxiety()
        return (f"EVDriver(Source: {self.source_node}, "
                f"Destination: {self.destination_node}, "
                f"Capacity: {self.battery_capacity_km}km, "
                f"SoC: {self.state_of_charge:.2f}, "
                f"Connector: {self.connector_type}, "
                f"Current: {self.get_current_node()}, "
                f"Profile: {anxiety_desc}, "
                f"Anxiety: {anxiety_score:.3f})")
    
    def __repr__(self):
        return self.__str__()