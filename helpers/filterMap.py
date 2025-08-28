import geopandas as gpd

# Load your full UK GeoJSON file
gdf = gpd.read_file("data/LAD_MAY_2025_UK_BFC_338470394117970014.geojson")

# Print available columns to confirm LAD code and country name field
print(gdf.columns)

#SINCE PROJECT IS FOR MAINLAND UK WE FILTER OUT ISLANDS

# === Step 1: Remove entire Northern Ireland ===
# If country name field exists, filter directly:
if "CTRY25NM" in gdf.columns:
    gdf = gdf[~gdf["CTRY25NM"].str.contains("Northern Ireland", case=False)]
else:
    # Or exclude based on LAD codes starting with N
    gdf = gdf[~gdf["LAD25CD"].str.startswith("N")]

# === Step 2: Exclude disconnected islands (by LAD25CD) ===
excluded_codes = [
    "E06000053",  # Isles of Scilly
    "E06000046",  # Isle of Wight
    "S12000013",  # Na h-Eileanan Siar (Western Isles)
    "S12000023",  # Orkney Islands
    "S12000027",  # Shetland Islands
    "S12000035",  # Argyll and Bute (optional)
]

# Filter out those districts
mainland_gdf = gdf[~gdf["LAD25CD"].isin(excluded_codes)]

# Save to new GeoJSON
mainland_gdf.to_file("UK_Mainland_GB.geojson", driver="GeoJSON")

print(" Filtered GeoJSON saved as 'UK_Mainland_GB.geojson'")
