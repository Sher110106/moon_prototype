#!/usr/bin/env python3

"""
Terrain Derivatives Script for Lunar Landslide Prototype
- Computes slope and curvature using RichDEM
- Applies 3x3 Gaussian filter to reduce stair-step artifacts
- Works with DTM data (dtm_snap.tif)
"""

import numpy as np
import richdem as rd
import rioxarray as rxr
from scipy import ndimage
import sys
import os
from pathlib import Path

def apply_gaussian_filter(data, sigma=1.0):
    """
    Apply 3x3 Gaussian filter to reduce stair-step artifacts
    
    Args:
        data: Input array
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Filtered array
    """
    return ndimage.gaussian_filter(data, sigma=sigma)

def compute_terrain_derivatives(dem_file):
    """
    Compute slope and curvature from DEM using RichDEM
    
    Args:
        dem_file: Path to input DEM file
    
    Returns:
        Tuple of (slope, curvature, original_dem)
    """
    print(f"Loading DEM: {dem_file}")
    
    # Load DEM using rioxarray
    dem_xr = rxr.open_rasterio(dem_file).squeeze()
    
    # Convert to float32 for processing
    dem_data = dem_xr.values.astype(np.float32)
    
    print(f"DEM shape: {dem_data.shape}")
    print(f"DEM data range: {dem_data.min():.2f} to {dem_data.max():.2f}")
    
    # Apply Gaussian filter to reduce stair-step artifacts
    print("Applying 3x3 Gaussian filter...")
    dem_filtered = apply_gaussian_filter(dem_data, sigma=1.0)
    
    # Convert to RichDEM array
    dem_rd = rd.rdarray(dem_filtered, no_data=-9999)
    
    # Compute slope in degrees
    print("Computing slope...")
    slope = rd.TerrainAttribute(dem_rd, attrib="slope_degrees")
    
    # Compute curvature
    print("Computing curvature...")
    curvature = rd.TerrainAttribute(dem_rd, attrib="curvature")
    
    return slope, curvature, dem_xr, dem_filtered

def save_terrain_products(slope, curvature, dem_filtered, original_dem, output_dir):
    """
    Save terrain derivative products to files
    
    Args:
        slope: Slope array
        curvature: Curvature array
        dem_filtered: Filtered DEM array
        original_dem: Original DEM xarray
        output_dir: Output directory path
    """
    # Get spatial reference info from original DEM
    crs = original_dem.rio.crs
    transform = original_dem.rio.transform()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save slope
    slope_file = output_dir / "slope_degrees.tif"
    slope_xr = rxr.DataArray(
        slope,
        dims=["y", "x"],
        coords={"y": original_dem.y, "x": original_dem.x}
    )
    slope_xr.rio.write_crs(crs, inplace=True)
    slope_xr.rio.to_raster(slope_file, compress="lzw")
    print(f"Saved slope: {slope_file}")
    
    # Save curvature
    curvature_file = output_dir / "curvature.tif"
    curvature_xr = rxr.DataArray(
        curvature,
        dims=["y", "x"],
        coords={"y": original_dem.y, "x": original_dem.x}
    )
    curvature_xr.rio.write_crs(crs, inplace=True)
    curvature_xr.rio.to_raster(curvature_file, compress="lzw")
    print(f"Saved curvature: {curvature_file}")
    
    # Save filtered DEM
    dem_filtered_file = output_dir / "dem_filtered.tif"
    dem_filtered_xr = rxr.DataArray(
        dem_filtered,
        dims=["y", "x"],
        coords={"y": original_dem.y, "x": original_dem.x}
    )
    dem_filtered_xr.rio.write_crs(crs, inplace=True)
    dem_filtered_xr.rio.to_raster(dem_filtered_file, compress="lzw")
    print(f"Saved filtered DEM: {dem_filtered_file}")
    
    return slope_file, curvature_file, dem_filtered_file

def print_statistics(slope, curvature, dem_filtered):
    """
    Print statistics for terrain derivatives
    
    Args:
        slope: Slope array
        curvature: Curvature array
        dem_filtered: Filtered DEM array
    """
    print("\n=== Terrain Derivatives Statistics ===")
    
    print(f"Filtered DEM:")
    print(f"  Min: {dem_filtered.min():.2f} m")
    print(f"  Max: {dem_filtered.max():.2f} m")
    print(f"  Mean: {dem_filtered.mean():.2f} m")
    print(f"  Std: {dem_filtered.std():.2f} m")
    
    print(f"\nSlope (degrees):")
    print(f"  Min: {slope.min():.2f}°")
    print(f"  Max: {slope.max():.2f}°")
    print(f"  Mean: {slope.mean():.2f}°")
    print(f"  Std: {slope.std():.2f}°")
    
    print(f"\nCurvature:")
    print(f"  Min: {curvature.min():.4f}")
    print(f"  Max: {curvature.max():.4f}")
    print(f"  Mean: {curvature.mean():.4f}")
    print(f"  Std: {curvature.std():.4f}")

def main():
    """Main terrain derivatives workflow"""
    print("Starting terrain derivatives computation...")
    
    # Change to data directory
    data_dir = Path(__file__).parent.parent / "data"
    os.chdir(data_dir)
    
    # Look for DTM file
    import glob
    dtm_files = glob.glob("dtm_snap.tif")
    
    if not dtm_files:
        # Fallback to other DTM files
        dtm_files = glob.glob("*dtm*.tif")
    
    if not dtm_files:
        print("Error: No DTM files found")
        print("Expected: dtm_snap.tif")
        sys.exit(1)
    
    dtm_file = dtm_files[0]
    print(f"Using DTM file: {dtm_file}")
    
    try:
        # Compute terrain derivatives
        slope, curvature, original_dem, dem_filtered = compute_terrain_derivatives(dtm_file)
        
        # Save products
        slope_file, curvature_file, dem_filtered_file = save_terrain_products(
            slope, curvature, dem_filtered, original_dem, "."
        )
        
        # Print statistics
        print_statistics(slope, curvature, dem_filtered)
        
        print("\n=== Terrain Derivatives Complete ===")
        print("Generated files:")
        print(f"- slope_degrees.tif (Slope in degrees)")
        print(f"- curvature.tif (Profile curvature)")
        print(f"- dem_filtered.tif (Gaussian filtered DEM)")
        
        print("\nNote: Files are ready for use in rule-based baseline processing")
        
    except Exception as e:
        print(f"Error during terrain derivatives computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()