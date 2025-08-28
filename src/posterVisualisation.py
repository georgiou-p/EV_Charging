import matplotlib.pyplot as plt
import networkx as nx
import geopandas
import numpy as np
from libpysal import weights
from haversine import haversine, Unit

def create_poster_map(geojson_path, figsize=(20, 24), dpi=300, save_path="uk_poster_map.png"):
    """
    Create a clean UK map visualization for poster presentation
    
    Args:
        geojson_path (str): Path to the UK GeoJSON file
        figsize (tuple): Figure size in inches (width, height)
        dpi (int): Resolution for saved image
        save_path (str): Path to save the generated image
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    print("Loading UK regions data...")
    # Load the GeoJSON data
    map_regions = geopandas.read_file(geojson_path)
    print(f"Loaded {len(map_regions)} regions")
    
    # Extract centroids for network connections
    centroids = np.column_stack((map_regions.centroid.x, map_regions.centroid.y))
    print("Calculated region centroids")
    
    # Create Queen adjacency graph 
    print("Creating adjacency network...")
    queen = weights.Queen.from_dataframe(map_regions)
    graph = queen.to_networkx()
    
    # Create positions dictionary for network plotting
    positions = dict(zip(graph.nodes, centroids))
    
    # Manual edge addition: 322 <-> 323
    if 322 in positions and 323 in positions:
        pos1 = positions[322]
        pos2 = positions[323]
        distance_km = haversine((pos1[1], pos1[0]), (pos2[1], pos2[0]), unit=Unit.KILOMETERS)
        graph.add_edge(322, 323, weight=distance_km)
        print(f"Added manual edge between nodes 322 and 323 (distance: {distance_km:.2f}km)")
    
    print(f"Created network with {len(graph.nodes)} nodes and {len(graph.edges)} connections")
    
    # Create the visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot the ward boundaries with light blue fill
    map_regions.plot(
        ax=ax, 
        linewidth=0.3,           
        edgecolor="white",      
        facecolor="lightblue",   
        alpha=0.8               # Slight transparency
    )
    
    # Plot the network connections between adjacent regions
    nx.draw_networkx_edges(
        graph, 
        positions, 
        ax=ax,
        edge_color='navy',       
        alpha=0.4,              
        width=0.3              
    )
    
    #Plot centroids as small points
    nx.draw_networkx_nodes(
        graph, 
        positions, 
        ax=ax,
        node_color='darkblue', 
        node_size=2,            
        alpha=0.6
    )
    
    # Set the geographic bounds for UK
    ax.set_xlim(-8.5, 2.5)      # Longitude range
    ax.set_ylim(49.5, 61)       # Latitude range
    
    # Remove axes and make clean
    ax.set_aspect('equal')      
    ax.axis('off')              
    
    
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as both SVG and JPEG formats
    print(f"Saving files in multiple formats...")
    
    # Save as SVG 
    svg_path = save_path.replace('.png', '.svg')
    plt.savefig(
        svg_path,
        format='svg',
        bbox_inches='tight',     
        pad_inches=0,          
        facecolor='white',      
        edgecolor='none'        
    )
    
    # Save as JPEG 
    jpeg_path = save_path.replace('.png', '.jpg')
    plt.savefig(
        jpeg_path,
        format='jpeg',
        dpi=dpi,               
        bbox_inches='tight',     
        pad_inches=0,           
        facecolor='white',      
        edgecolor='none'       
    )
    
    print(f"Files saved:")
    print(f"  SVG: {svg_path} (vector - scalable)")
    print(f"  JPEG: {jpeg_path} (raster - {figsize[0]*dpi:.0f}x{figsize[1]*dpi:.0f} pixels)")
    
    # Display the image
    plt.show()
    
    return fig

def create_minimal_poster_map(geojson_path, figsize=(16, 20), dpi=300, save_path="uk_minimal_poster.png"):
    """
    Create an even cleaner version with just boundaries for poster background
    
    Args:
        geojson_path (str): Path to the UK GeoJSON file
        figsize (tuple): Figure size in inches
        dpi (int): Resolution for saved image
        save_path (str): Path to save the generated image
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    print("Creating minimal poster map...")
    # Load the GeoJSON data
    map_regions = geopandas.read_file(geojson_path)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot only the ward boundaries with very light styling
    map_regions.plot(
        ax=ax, 
        linewidth=0.2,         
        edgecolor="lightgray",  
        facecolor="lightblue",  
        alpha=0.3             
    )
    
    # Set the geographic bounds for UK
    ax.set_xlim(-8.5, 2.5)
    ax.set_ylim(49.5, 61)

    ax.set_aspect('equal')
    ax.axis('off')
    
    # Remove padding
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as vector graphics
    plt.savefig(
        save_path.replace('.png', '.svg'),
        format='svg',
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        edgecolor='none'
    )
    
    plt.savefig(
        save_path.replace('.png', '.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        edgecolor='none'
    )
    
    print(f" Minimal poster map saved as vector graphics:")
    print(f"  SVG: {save_path.replace('.png', '.svg')}")
    print(f"  PDF: {save_path.replace('.png', '.pdf')}")
    plt.show()
    
    return fig

def main():
    """
    Main function to generate UK poster visualizations
    """
    print("="*60)
    print("UK POSTER MAP GENERATOR")
    print("="*60)
    
    # Path to the simplified UK GeoJSON file
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"
    
    try:
        print("\n1. Creating detailed poster map with network connections...")
        print("-" * 50)
        
        # Create the main poster map with network connections
        fig1 = create_poster_map(
            geojson_path=geojson_path,
            figsize=(20, 24),         
            dpi=300,                  
            save_path="uk_poster_detailed.png"
        )
        
        print("\n2. Creating minimal poster map for background use...")
        print("-" * 50)
        
        # Create a minimal version for background use
        fig2 = create_minimal_poster_map(
            geojson_path=geojson_path,
            figsize=(16, 20),         
            dpi=300,                   
            save_path="uk_poster_minimal.png"
        )
        
        print("\n" + "="*60)
        print(" POSTER MAPS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("• uk_poster_detailed.svg - Vector map (scalable, perfect quality)")
        print("• uk_poster_detailed.jpg - High-resolution raster image")
        print("\nUse SVG for presentations and web, JPEG for printing and sharing.")
        
    except FileNotFoundError:
        print(f" Error: Could not find GeoJSON file at '{geojson_path}'")
        print("Make sure the UK_Mainland_GB_simplified.geojson file exists in the data/ directory")
        
    except Exception as e:
        print(f" Error creating poster maps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()