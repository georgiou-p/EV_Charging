from libpysal import weights
import matplotlib.pyplot as plt
import networkx as nx
import geopandas
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

# read in example data from geojson. GeoJSON is a file format
# for encoding geographic data based on JSON. It is useful for
# presenting geographic data on the web, and is increasingly
# used as a file format for geographic data.
filepath = "data/UK_Mainland_GB_simplified.geojson"
map_regions = geopandas.read_file(filepath)

# extract the centroids for connecting the regions, which is
# the average of the coordinates that define the polygon's boundary
centroids = np.column_stack((map_regions.centroid.x, map_regions.centroid.y))

# construct the "Queen" adjacency graph. In geographical applications,
# the "Queen" adjacency graph considers two polygons as connected if
# they share a single point on their boundary. This is an analogue to
# the "Moore" neighborhood nine surrounding cells in a regular grid.
queen = weights.Queen.from_dataframe(map_regions)

# Then, we can convert the graph to networkx object using the
# .to_networkx() method.
graph = queen.to_networkx()

# To plot with networkx, we need to merge the nodes back to
# their positions in order to plot in networkx
positions = dict(zip(graph.nodes, centroids))

# Add weights to edges based on Haversine distance in kilometers
for edge in graph.edges():
    node1, node2 = edge
    pos1 = positions[node1]  # (longitude, latitude)
    pos2 = positions[node2]  # (longitude, latitude)
    
    # Calculate Haversine distance in kilometers
    distance_km = haversine(pos1[0], pos1[1], pos2[0], pos2[1])
    
    # Add distance as weight to the edge
    graph.edges[node1, node2]['weight'] = distance_km

# plot with a nice basemap
ax = map_regions.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
ax.axis([-12, 45, 33, 66])
ax.axis("off")
nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
plt.show()
print(graph)
print(f"Graph now has weighted edges in kilometers. Sample edge weights: {[(edge[0], edge[1], f'{edge[2]['weight']:.2f}km') for edge in list(graph.edges(data=True))[:3]]}")