import geopandas as gpd

# Load the filtered mainland UK GeoJSON (from earlier step)
mainland_gdf = gpd.read_file("data/UK_Mainland_GB.geojson")

# Simplify geometries – adjust tolerance as needed
simplified_gdf = mainland_gdf.copy()
simplified_gdf["geometry"] = simplified_gdf["geometry"].simplify(
    tolerance=0.001,  # degrees (~100–120 meters at UK latitude)
    preserve_topology=True
)

# Save the simplified version
simplified_gdf.to_file("UK_Mainland_GB_simplified.geojson", driver="GeoJSON")

print("Simplified GeoJSON saved as 'UK_Mainland_GB_simplified.geojson'")
