# import geopandas as gpd

# # 用途地域
# # file = "A29-11_13.shp"
# # 防火地域
# file = "防火準防火.shp"

# # Load the shapefile
# gdf = gpd.read_file(file, encoding='Shift_JIS')

# # Save to GeoJSON
# gdf.to_file("output_file.geojson", driver='GeoJSON', encoding='Shift_JIS')

import geopandas as gpd
from pyproj import CRS

# Load the shapefile
file = "防火準防火.shp"
gdf = gpd.read_file(file, encoding='Shift_JIS')
gdf.crs = "EPSG:6669"

# Define the source CRS
source_crs = CRS.from_string(
    "+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")

# Define the target CRS (WGS 84)
target_crs = CRS.from_epsg(4326)

# Transform the geometry to the target CRS
gdf = gdf.to_crs(target_crs)

# Save to GeoJSON
gdf.to_file("output_file.geojson", driver='GeoJSON', encoding='utf-8')
