#!/usr/bin/env python3

"""
Hapke Normalisation Script for Lunar Landslide Prototype
- Implements both cosine and Hapke photometric corrections
- Prototypes on one 512x512 TMC tile
- Uses solar elevation = 41.3° (incidence angle = 48.7°)
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import sys
import os
from pathlib import Path

def cosine_correction(image, incidence_angle_deg):
    """
    Apply cosine correction to normalize for illumination
    
    Args:
        image: Input image array
        incidence_angle_deg: Solar incidence angle in degrees
    
    Returns:
        Cosine-corrected image
    """
    i_rad = np.deg2rad(incidence_angle_deg)
    cos_i = np.cos(i_rad)
    
    # Avoid division by zero
    cos_i = np.clip(cos_i, 0.01, 1.0)
    
    return image / cos_i

def hapke_correction(image, incidence_angle_deg, emission_angle_deg=0, albedo=0.11):
    """
    Apply Hapke photometric correction
    
    Args:
        image: Input image array
        incidence_angle_deg: Solar incidence angle in degrees
        emission_angle_deg: Emission angle in degrees (default 0 for nadir)
        albedo: Single scattering albedo (default 0.11 for mature mare regolith)
    
    Returns:
        Hapke-corrected image
    """
    i_rad = np.deg2rad(incidence_angle_deg)
    e_rad = np.deg2rad(emission_angle_deg)
    
    mu0 = np.cos(i_rad)  # cosine of incidence angle
    mu = np.cos(e_rad)   # cosine of emission angle
    
    # Avoid division by zero
    mu0 = np.clip(mu0, 0.01, 1.0)
    mu = np.clip(mu, 0.01, 1.0)
    
    w = albedo
    
    def H(x, w):
        """Hapke H function"""
        return (1 + 2 * x) / (1 + 2 * x * np.sqrt(1 - w))
    
    # Hapke correction formula
    correction_factor = (mu0 + H(mu0, w)) / (mu + H(mu, w))
    
    return image * correction_factor

def extract_tile(input_file, output_file, tile_size=512):
    """
    Extract a 512x512 tile from the center of the input raster
    
    Args:
        input_file: Input raster file path
        output_file: Output tile file path
        tile_size: Size of tile to extract (default 512)
    """
    with rasterio.open(input_file) as src:
        # Get image dimensions
        height, width = src.height, src.width
        
        # Calculate center coordinates
        center_row = height // 2
        center_col = width // 2
        
        # Calculate tile bounds
        half_tile = tile_size // 2
        row_start = max(0, center_row - half_tile)
        row_end = min(height, center_row + half_tile)
        col_start = max(0, center_col - half_tile)
        col_end = min(width, center_col + half_tile)
        
        # Read the tile
        window = rasterio.windows.Window(col_start, row_start, 
                                       col_end - col_start, 
                                       row_end - row_start)
        tile_data = src.read(1, window=window)
        
        # Get transform for the tile
        tile_transform = rasterio.windows.transform(window, src.transform)
        
        # Write the tile
        with rasterio.open(output_file, 'w', 
                          driver='GTiff',
                          height=tile_data.shape[0],
                          width=tile_data.shape[1],
                          count=1,
                          dtype=tile_data.dtype,
                          crs=src.crs,
                          transform=tile_transform) as dst:
            dst.write(tile_data, 1)
    
    print(f"Extracted {tile_data.shape[0]}x{tile_data.shape[1]} tile to {output_file}")
    return output_file

def main():
    """Main Hapke normalization workflow"""
    print("Starting Hapke normalization workflow...")
    
    # Change to data directory
    data_dir = Path(__file__).parent.parent / "data"
    os.chdir(data_dir)
    
    # Solar parameters from TMC header
    sun_elevation = 41.3  # degrees
    incidence_angle = 90.0 - sun_elevation  # 48.7 degrees
    
    print(f"Solar elevation: {sun_elevation}°")
    print(f"Incidence angle: {incidence_angle}°")
    
    # Find TMC ortho file
    import glob
    tmc_files = glob.glob("*tmc*orth*_cog.tif")
    
    if not tmc_files:
        print("Error: No TMC ortho files found")
        sys.exit(1)
    
    input_file = tmc_files[0]
    print(f"Using TMC file: {input_file}")
    
    # Extract a 512x512 tile for processing
    tile_file = "tmc_tile.tif"
    extract_tile(input_file, tile_file)
    
    # Read the tile
    with rasterio.open(tile_file) as src:
        tile_data = src.read(1).astype(np.float32)
        profile = src.profile
    
    print(f"Tile shape: {tile_data.shape}")
    print(f"Tile data type: {tile_data.dtype}")
    print(f"Tile min/max: {tile_data.min():.2f} / {tile_data.max():.2f}")
    
    # Apply cosine correction
    print("\nApplying cosine correction...")
    cosine_corrected = cosine_correction(tile_data, incidence_angle)
    
    # Apply Hapke correction
    print("Applying Hapke correction...")
    hapke_corrected = hapke_correction(tile_data, incidence_angle, 
                                     emission_angle_deg=0, albedo=0.11)
    
    # Save corrected images
    profile.update(dtype=np.float32)
    
    # Save cosine corrected
    with rasterio.open("tmc_tile_cosine.tif", 'w', **profile) as dst:
        dst.write(cosine_corrected.astype(np.float32), 1)
    
    # Save Hapke corrected
    with rasterio.open("tmc_tile_hapke.tif", 'w', **profile) as dst:
        dst.write(hapke_corrected.astype(np.float32), 1)
    
    print("\n=== Photometric Correction Complete ===")
    print("Generated files:")
    print(f"- tmc_tile.tif (Original 512x512 tile)")
    print(f"- tmc_tile_cosine.tif (Cosine corrected)")
    print(f"- tmc_tile_hapke.tif (Hapke corrected)")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Original - Min: {tile_data.min():.2f}, Max: {tile_data.max():.2f}, Mean: {tile_data.mean():.2f}")
    print(f"Cosine - Min: {cosine_corrected.min():.2f}, Max: {cosine_corrected.max():.2f}, Mean: {cosine_corrected.mean():.2f}")
    print(f"Hapke - Min: {hapke_corrected.min():.2f}, Max: {hapke_corrected.max():.2f}, Mean: {hapke_corrected.mean():.2f}")
    
    print("\nNote: Keep tmc_tile.tif for runtime benchmarking")
    print("Use cosine correction for bulk processing")

if __name__ == "__main__":
    main()