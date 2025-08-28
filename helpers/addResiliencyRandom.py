import json
import random

def add_working_status_to_charging_stations(input_file, output_file, failure_rate=0.1):
    """
    Add a 'Working' boolean flag to each connection in the charging stations data.
    Sets 10% of total charging points to Working: false.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the output JSON file
        failure_rate (float): Percentage of charging points to set as not working (default 0.1 = 10%)
    """
    
    # Load the charging stations data
    print("Loading charging stations data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        stations_data = json.load(f)
    
    # Count total charging points across all stations
    total_charging_points = 0
    all_connections = []  # Store references to all connections for random selection
    
    for station in stations_data:
        for connection in station.get('Connections', []):
            quantity = connection.get('Quantity', 1)
            total_charging_points += quantity
            # Add reference for each individual charging point
            for _ in range(quantity):
                all_connections.append(connection)
    
    print(f"Total charging points found: {total_charging_points}")
    
    # Calculate how many points should fail
    points_to_fail = int(total_charging_points * failure_rate)
    print(f"Points to set as not working: {points_to_fail} ({failure_rate*100}%)")
    
    # Initially set all connections to Working: true
    for station in stations_data:
        for connection in station.get('Connections', []):
            connection['Working'] = True
    
    # Randomly select charging points to fail
    if points_to_fail > 0:
        # Create a list of unique connection objects to avoid setting the same connection multiple times
        unique_connections = []
        connection_counts = {}
        
        for station in stations_data:
            for connection in station.get('Connections', []):
                conn_id = id(connection)  # Use object id as unique identifier
                if conn_id not in connection_counts:
                    unique_connections.append(connection)
                    connection_counts[conn_id] = connection.get('Quantity', 1)
        
        # Randomly select connections to fail, weighted by their quantity
        random.shuffle(unique_connections)
        failed_points = 0
        
        for connection in unique_connections:
            if failed_points >= points_to_fail:
                break
            
            # Set this connection to not working
            connection['Working'] = False
            failed_points += connection.get('Quantity', 1)
            
            print(f"Set connection (PowerKW: {connection.get('PowerKW', 'N/A')}, "
                  f"Type: {connection.get('ConnectionTypeID', 'N/A')}, "
                  f"Quantity: {connection.get('Quantity', 1)}) to Working: false")
    
    # Count actual failed points
    actual_failed_points = 0
    stations_with_failures = 0
    
    for station in stations_data:
        station_has_failure = False
        for connection in station.get('Connections', []):
            if not connection.get('Working', True):
                actual_failed_points += connection.get('Quantity', 1)
                station_has_failure = True
        
        if station_has_failure:
            stations_with_failures += 1
    
    print(f"\nActual failed points: {actual_failed_points} ({(actual_failed_points/total_charging_points)*100:.1f}%)")
    print(f"Stations with failures: {stations_with_failures} out of {len(stations_data)}")
    
    # Save the updated data
    print(f"\nSaving updated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stations_data, f, indent=4, ensure_ascii=False)
    
    print(" Successfully added Working status to all connections!")
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"  Total stations: {len(stations_data)}")
    print(f"  Total charging points: {total_charging_points}")
    print(f"  Working points: {total_charging_points - actual_failed_points}")
    print(f"  Failed points: {actual_failed_points}")
    print(f"  Failure rate: {(actual_failed_points/total_charging_points)*100:.1f}%")
    print(f"  Stations affected: {stations_with_failures}")
    
    return {
        'total_stations': len(stations_data),
        'total_points': total_charging_points,
        'working_points': total_charging_points - actual_failed_points,
        'failed_points': actual_failed_points,
        'failure_rate': (actual_failed_points/total_charging_points)*100,
        'stations_affected': stations_with_failures
    }

def verify_output_format(output_file):
    """
    Verify that the output file has the correct format
    """
    print(f"\nVerifying output format in {output_file}...")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check first few stations
    for i, station in enumerate(data[:3]):
        print(f"\nStation {station.get('StationID', 'Unknown')}:")
        print(f"  Connections: {len(station.get('Connections', []))}")
        
        for j, connection in enumerate(station.get('Connections', [])):
            working_status = connection.get('Working', 'MISSING')
            print(f"    Connection {j+1}: PowerKW={connection.get('PowerKW', 'N/A')}, "
                  f"Type={connection.get('ConnectionTypeID', 'N/A')}, "
                  f"Quantity={connection.get('Quantity', 1)}, "
                  f"Working={working_status}")
    
    print("Output format verification complete!")

def main():
    """
    Main function to add working status to charging stations
    """
    input_file = "./data/cleaned_charging_stations.json"
    output_file = "RandomFailuresChargingStations.json"
    
    # Set random seed for reproducible results 
    random.seed(42)
    
    try:
        # Add working status
        stats = add_working_status_to_charging_stations(input_file, output_file)
        
        # Verify the output format
        verify_output_format(output_file)
        
        print(f"\n{'='*60}")
        print("CHARGING STATION FAILURE SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Failure simulation: {stats['failure_rate']:.1f}% of charging points set to non-working")
        print("Ready for resiliency analysis!")
        
    except FileNotFoundError:
        print(f" Error: Could not find input file '{input_file}'")
        print("Please ensure the cleaned_charging_stations.json file exists in the current directory.")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()