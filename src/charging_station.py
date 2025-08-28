import json
from typing import List, Dict, Any
from collections import deque

class Connection:
    def __init__(self, power_kw: float, connection_type_id: int = None, 
                 amps: float = None, voltage: float = None, quantity: int = 1, working: bool = True):
        self.power_kw = power_kw
        self.connection_type_id = connection_type_id
        self.amps = amps
        self.voltage = voltage
        self.quantity = quantity
        self.working = working

    def __str__(self):
        return f"Connection(Power: {self.power_kw}kW, Type: {self.connection_type_id}, Amps: {self.amps}, Voltage: {self.voltage}, Qty: {self.quantity})"

class EVChargingStation:
    def __init__(self, station_id: str, latitude: float, longitude: float, 
                 number_of_points: int, connections: List[Connection]):
        self.station_id = station_id 
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_points = number_of_points
        self.connections = connections
        self.charging_queue = deque()  # Queue for charging
         #ADD COST in the future
    
    # Getters
    def get_station_id(self):
        return self.station_id
    
    def get_latitude(self):
        return self.latitude
    
    def get_longitude(self):
        return self.longitude
    
    def get_number_of_points(self):
        return self.number_of_points
    
    def get_connections(self):
        return self.connections
    
    def get_position(self):
        return (self.longitude, self.latitude)
    
    def get_charging_queue(self):
        return self.charging_queue
    
    # Setters
    def set_station_id(self, station_id: str): 
        self.station_id = station_id
    
    def set_latitude(self, latitude: float):
        self.latitude = latitude
    
    def set_longitude(self, longitude: float):
        self.longitude = longitude
    
    def set_number_of_points(self, number_of_points: int):
        self.number_of_points = number_of_points
    
    def set_connections(self, connections: List[Connection]):
        self.connections = connections
    
    # Queue methods
    def add_to_queue(self, driver):
        """Add a driver to the charging queue"""
        self.charging_queue.append(driver)
    
    def remove_from_queue(self):
        """Remove and return the next driver from the queue"""
        if self.charging_queue:
            return self.charging_queue.popleft()
        return None
    
    def get_queue_length(self):
        """Get the current length of the charging queue"""
        return len(self.charging_queue)
    
    def is_connection_working(self, connection_index):
        """Check if a specific connection is working"""
        if 0 <= connection_index < len(self.connections):
            return self.connections[connection_index].working
        return False

    def get_working_connections(self, connector_type=None):
        """Get all working connections, optionally filtered by connector type"""
        working_connections = []
        for connection in self.connections:
            if hasattr(connection, 'working') and connection.working:
                if connector_type is None or connection.connection_type_id == connector_type or connection.connection_type_id == 0:
                    working_connections.append(connection)
        return working_connections
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]):
        """Create EVChargingStation from JSON data"""
        connections = [
            Connection(
                power_kw=conn.get('PowerKW', 0),
                connection_type_id=conn.get('ConnectionTypeID'),
                amps=conn.get('Amps'),
                voltage=conn.get('Voltage'),
                quantity=conn.get('Quantity', 1),
                working=conn.get('Working', True)
            )
            for conn in json_data.get('Connections', [])
        ]
        
        return cls(
            station_id=json_data.get('StationID'),
            latitude=json_data.get('Latitude'),
            longitude=json_data.get('Longitude'),
            number_of_points=json_data.get('NumberOfPoints', 1),
            connections=connections
        )
    
    def __str__(self):
        return f"EVChargingStation(ID={self.station_id}, Lat={self.latitude}, Lon={self.longitude}, Points={self.number_of_points}, Queue={self.get_queue_length()})"

def load_stations_from_json(json_file_path: str) -> List[EVChargingStation]:
    """Load charging stations from JSON file"""
    stations = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both array format and single object format
    if isinstance(data, list):
        station_data_list = data
    else:
        station_data_list = [data]
    
    for station_data in station_data_list:
        try:
            station = EVChargingStation.from_json(station_data)
            stations.append(station)
        except Exception as e:
            print(f"Error loading station {station_data.get('ID', 'unknown')}: {e}")
    
    return stations

