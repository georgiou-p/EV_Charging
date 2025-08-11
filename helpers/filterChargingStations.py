import json

# Mapping from original OCM IDs to new simplified IDs
id_mapping = {
    0: 0,     # Unknown
    33: 1,    # CCS2
    25: 2,    # Type 2 socket
    1036: 2,  # Type 2 tethered
    3: 2,     # 3-pin
    1: 3,     # Type 1
    2: 3      # CHAdeMO
}

# Load your data with UTF-8 encoding
with open('data/uk_charging_stations_pretty.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

cleaned_data = []
station_counter = 1

for station in raw_data:
    station_id = f"CS{station_counter:05d}"
    lat = station.get("AddressInfo", {}).get("Latitude")
    lon = station.get("AddressInfo", {}).get("Longitude")

    connections = []
    num_points = 0

    for conn in station.get("Connections", []):
        quantity = conn.get("Quantity", 1)
        num_points += quantity

        # Map the ConnectionTypeID to the simplified ID, default to 0 if not found
        original_id = conn.get("ConnectionTypeID", 0)
        new_id = id_mapping.get(original_id, 0)

        conn_entry = {
            "PowerKW": conn.get("PowerKW"),
            "Amps": conn.get("Amps"),
            "Voltage": conn.get("Voltage"),
            "ConnectionTypeID": new_id,
            "Quantity": quantity
        }
        connections.append(conn_entry)

    station_entry = {
        "StationID": station_id,
        "Latitude": lat,
        "Longitude": lon,
        "NumberOfPoints": num_points,
        "Connections": connections
    }

    cleaned_data.append(station_entry)
    station_counter += 1

# Save as JSON with UTF-8 encoding
with open("cleaned_charging_stations.json", "w", encoding='utf-8') as f:
    json.dump(cleaned_data, f, indent=4)

print("Done! Saved cleaned_charging_stations.json")
