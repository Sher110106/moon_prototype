#!/usr/bin/env python3

"""
Script to create empty aoi_landslide_boulder.gpkg placeholder
Run this after setting up the conda environment
"""

import geopandas as gpd

def create_empty_gpkg():
    """Create empty GeoPackage with proper structure"""
    
    # Create empty GeoDataFrame with expected columns
    gdf = gpd.GeoDataFrame(
        columns=['geometry', 'feature_type', 'confidence', 'area_m2'], 
        crs='EPSG:104903'
    )
    
    # Save as GeoPackage
    gdf.to_file('aoi_landslide_boulder.gpkg', driver='GPKG')
    print("Created empty aoi_landslide_boulder.gpkg")

if __name__ == "__main__":
    create_empty_gpkg()