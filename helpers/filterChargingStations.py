import json

# Load your data with UTF-8 encoding
with open('data/uk_charging_stations_pretty.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Prepare list for cleaned data
cleaned_data = []

# Unique internal ID counter
station_counter = 1

# Iterate through each station entry
for station in raw_data:
    station_id = f"CS{station_counter:05d}"
    lat = station["AddressInfo"].get("Latitude")
    lon = station["AddressInfo"].get("Longitude")

    connections = []
    num_points = 0

    for conn in station.get("Connections", []):
        quantity = conn.get("Quantity")
        if quantity is None:
            quantity = 1
        num_points += quantity

        conn_entry = {
            "PowerKW": conn.get("PowerKW"),
            "Amps": conn.get("Amps"),
            "Voltage": conn.get("Voltage"),
            "CurrentTypeID": conn.get("CurrentTypeID"),
            "Quantity": quantity  # This ensures we don't keep null
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
