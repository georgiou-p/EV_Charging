import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

def load_station_data(json_file):
    """
    Load and parse charging station data from JSON file.
    Returns lists of coordinates, total points, and failed points.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        stations = json.load(f)
    
    lats, lons, tot, fail = [], [], [], []
    
    for station in stations:
        lat = station.get("Latitude")
        lon = station.get("Longitude")
        
        if not lat or not lon:
            continue
            
        total_points = 0
        failed_points = 0
        
        for connection in station.get("Connections", []):
            quantity = int(connection.get("Quantity", 1) or 1)
            total_points += quantity
            
            if connection.get("Working") is False:
                failed_points += quantity
        
        if total_points == 0:
            continue
            
        lats.append(lat)
        lons.append(lon)
        tot.append(total_points)
        fail.append(failed_points)
    
    return np.array(lats), np.array(lons), np.array(tot, float), np.array(fail, float)

def create_uk_basemap(ax, geojson_path):
    """
    Create UK basemap from GeoJSON file.
    """
    try:
        uk = gpd.read_file(geojson_path)
        uk.plot(ax=ax, linewidth=0.4, edgecolor="white", 
               facecolor="lightblue", alpha=0.6)
        
        # Set UK bounds
        ax.set_xlim(-8.8, 2.6)
        ax.set_ylim(49.5, 61.2)
        ax.set_aspect("equal", adjustable="datalim")
        
    except Exception as e:
        print(f"Warning: Could not load basemap - {e}")
        ax.set_aspect("equal")

def create_failure_map(json_file, geojson_path, title, save_path):
    """
    Create a map showing charging station failures.
    """
    # Load station data
    lats, lons, tot, fail = load_station_data(json_file)
    
    if len(lats) == 0:
        print(f"No valid stations found in {json_file}")
        return
    
    # Calculate failure share
    share = np.divide(fail, tot, out=np.zeros_like(fail), where=tot > 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    
    # Add UK basemap
    create_uk_basemap(ax, geojson_path)
    
    # Calculate marker sizes based on total charging points
    max_total = max(1, np.percentile(tot, 95))
    sizes = 10 + 90 * (tot / max_total)
    sizes = np.clip(sizes, 10, 120)
    
    # Create scatter plot
    sc = ax.scatter(lons, lats, c=share, s=sizes, 
                   cmap="Reds", vmin=0, vmax=1,
                   edgecolors="black", linewidths=0.3, 
                   alpha=0.85, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Share of failed charging points per station", 
                  rotation=270, labelpad=18)
    
    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    
    # Add statistics text
    total_stations = len(lats)
    failed_stations = np.sum(fail > 0)
    total_points = np.sum(tot)
    total_failed_points = np.sum(fail)
    
    stats_text = (f"Stations: {total_stations}\n"
                 f"Failed Stations: {failed_stations}\n"
                 f"Total Points: {int(total_points)}\n"
                 f"Failed Points: {int(total_failed_points)}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor="white", alpha=0.8))
    
    # Save and display
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved map to {save_path}")
    plt.close()

def create_all_failure_maps():
    """
    Create all three failure maps for different scenarios.
    """
    # File paths - adjust these to match your file structure
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    
    # Map configurations: (json_file, title, output_file)
    map_configs = [
        {
            'json_file': 'data/AllWorkingChargingStations.json',
            'title': 'EV Charging Stations - All Working',
            'output_file': 'uk_map_all_working.png'
        },
        {
            'json_file': 'data/RandomFailuresChargingStations.json', 
            'title': 'EV Charging Stations - 10% Random Failures',
            'output_file': 'uk_map_random_failures.png'
        },
        {
            'json_file': 'data/TargetedWeightedFailures.json',
            'title': 'EV Charging Stations - 10% Targeted Failures', 
            'output_file': 'uk_map_targeted_failures.png'
        }
    ]
    
    print("Creating UK charging station failure maps...")
    print("=" * 60)
    
    for config in map_configs:
        print(f"Processing: {config['title']}")
        try:
            create_failure_map(
                json_file=config['json_file'],
                geojson_path=geojson_path,
                title=config['title'],
                save_path=config['output_file']
            )
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
        except Exception as e:
            print(f"Error creating map: {e}")
    
    print("=" * 60)
    print("Map generation completed!")
    print("\nGenerated files:")
    for config in map_configs:
        print(f"  - {config['output_file']}")

def create_comparison_map(geojson_path="data/UK_Mainland_GB_simplified.geojson"):
    """
    Create a side-by-side comparison of all three scenarios in dark mode.
    """
    # Map configurations with corrected filenames
    map_configs = [
        ('data/AllWorkingChargingStations.json', 'Baseline - All Working'),
        ('data/RandomFailuresChargingStations.json', '10% Random Failures'),
        ('data/TargetedWeightedFailures.json', '10% Targeted Failures')
    ]
    
    # Create figure with black background and high DPI
    fig = plt.figure(figsize=(21, 9), dpi=1000, facecolor='black')
    
    # Create subplots with black background
    axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, facecolor='black')
        axes.append(ax)
    
    fig.suptitle('EV Charging Station Failure Scenarios Comparison', 
                fontsize=18, fontweight='bold', y=0.93, color='white')
    
    for idx, (json_file, title) in enumerate(map_configs):
        ax = axes[idx]
        
        try:
            # Load data
            lats, lons, tot, fail = load_station_data(json_file)
            
            if len(lats) == 0:
                ax.text(0.5, 0.5, f'No data for\n{title}', 
                       ha='center', va='center', transform=ax.transAxes, 
                       color='white', fontsize=12)
                continue
            
            try:
                uk = gpd.read_file(geojson_path)
                uk.plot(ax=ax, linewidth=0.4, edgecolor="white", 
                       facecolor="lightblue", alpha=0.6)
                
                ax.set_xlim(-8.5, 2.2)   
                ax.set_ylim(49.8, 61.0)  
                ax.set_aspect("equal", adjustable="box")
                
            except Exception as e:
                print(f"Warning: Could not load basemap - {e}")
                ax.set_aspect("equal")
                ax.set_xlim(-8.5, 2.2)   
                ax.set_ylim(49.8, 61.0)
            
            # Calculate failure share and sizes
            share = np.divide(fail, tot, out=np.zeros_like(fail), where=tot > 0)
            max_total = max(1, np.percentile(tot, 95))
            sizes = 8 + 40 * (tot / max_total) 
            sizes = np.clip(sizes, 4, 60)
            
            # Create scatter plot 
            sc = ax.scatter(lons, lats, c=share, s=sizes,
                           cmap="Reds", vmin=0, vmax=1,
                           alpha=0.8, zorder=5)
            
            ax.set_title(title, fontsize=16, fontweight="bold", 
                        pad=20, color='white')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(False)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Set black background
            ax.set_facecolor('black')
            
            # Remove margins for edge-to-edge plotting
            ax.margins(0)
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{title}\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, color='white')
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.02, right=0.98, wspace=0.05)
    
    # Add horizontal colorbar with dark styling
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.04])  # [left, bottom, width, height]
    cbar_ax.set_facecolor('black')
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Failure Share", fontsize=14, labelpad=12, color='white')
    
    # Style colorbar for dark mode
    cbar.ax.tick_params(colors='white', labelsize=12)
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)
    
    # Save 
    plt.savefig('uk_maps_comparison_dark.png', bbox_inches="tight", dpi=1000, 
               facecolor='black', edgecolor='none')
    print("Saved comparison map to uk_maps_comparison_dark.png")
    plt.close()

if __name__ == "__main__":
    # Create individual maps
    create_all_failure_maps()
    
    # Create comparison map
    print("\nCreating dark mode comparison map...")
    try:
        create_comparison_map()
    except Exception as e:
        print(f"Could not create comparison map: {e}")
    
    print("\nAll maps generated successfully!")