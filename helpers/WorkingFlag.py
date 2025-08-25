import json

def add_working_flag_all_true(input_file, output_file):
    """
    Add a 'Working' boolean flag set to True for all connections in the charging stations data.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the output JSON file
    """
    
    # Load the charging stations data
    print("Loading charging stations data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        stations_data = json.load(f)
    
    # Count total charging points and connections
    total_charging_points = 0
    total_connections = 0
    total_stations = len(stations_data)
    
    # Add Working: true to all connections
    for station in stations_data:
        for connection in station.get('Connections', []):
            connection['Working'] = True
            total_connections += 1
            quantity = connection.get('Quantity', 1)
            total_charging_points += quantity
    
    print(f"Added Working: true to all connections")
    print(f"Total stations processed: {total_stations}")
    print(f"Total connections processed: {total_connections}")
    print(f"Total charging points: {total_charging_points}")
    
    # Save the updated data
    print(f"\nSaving updated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stations_data, f, indent=4, ensure_ascii=False)
    
    print(" Successfully added Working: true flag to all connections!")
    
    return {
        'total_stations': total_stations,
        'total_connections': total_connections,
        'total_points': total_charging_points,
        'working_points': total_charging_points,
        'failed_points': 0,
        'failure_rate': 0.0
    }

def verify_output_format(output_file):
    """
    Verify that the output file has the correct format with all Working flags set to true
    """
    print(f"\nVerifying output format in {output_file}...")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check first few stations
    all_working = True
    non_working_count = 0
    
    for i, station in enumerate(data[:3]):
        print(f"\nStation {station.get('StationID', 'Unknown')}:")
        print(f"  Connections: {len(station.get('Connections', []))}")
        
        for j, connection in enumerate(station.get('Connections', [])):
            working_status = connection.get('Working', 'MISSING')
            if working_status != True:
                all_working = False
                if working_status == False:
                    non_working_count += 1
            
            print(f"    Connection {j+1}: PowerKW={connection.get('PowerKW', 'N/A')}, "
                  f"Type={connection.get('ConnectionTypeID', 'N/A')}, "
                  f"Quantity={connection.get('Quantity', 1)}, "
                  f"Working={working_status}")
    
    # Quick scan of all stations to verify all are working
    for station in data:
        for connection in station.get('Connections', []):
            if connection.get('Working') != True:
                all_working = False
                non_working_count += 1
    
    if all_working:
        print(" All connections have Working: true - verification successful!")
    else:
        print(f"  Found {non_working_count} connections that are not set to Working: true")
    
    print(" Output format verification complete!")

def main():
    """
    Main function to add Working: true status to all charging station connections
    """
    input_file = "./data/cleaned_charging_stations.json"
    output_file = "AllWorkingChargingStations.json"
    
    try:
        # Add working status (all true)
        stats = add_working_flag_all_true(input_file, output_file)
        
        # Verify the output format
        verify_output_format(output_file)
        
        print(f"\n{'='*60}")
        print("ALL WORKING CHARGING STATIONS FILE CREATED")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"All {stats['total_points']} charging points set to Working: true")
        print("Ready for baseline comparison testing!")
        
    except FileNotFoundError:
        print(f" Error: Could not find input file '{input_file}'")
        print("Please ensure the cleaned_charging_stations.json file exists in the current directory.")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()