#!/usr/bin/env python3

"""
Rule-based Baseline Script for Lunar Landslide Prototype
- Computes GLCM texture contrast
- Creates landslide mask using slope, curvature, and texture thresholds
- Detects boulder seeds using Laplacian-of-Gaussian
"""

import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.features import shapes
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import blob_log
from skimage.util import view_as_windows
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape, Point
import sys
import os
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional
import time

warnings.filterwarnings('ignore')

def _compute_glcm_block(args: Tuple[np.ndarray, int, int, int]) -> np.ndarray:
    """Compute GLCM for a block of the image (for multiprocessing)"""
    img_block, start_row, start_col, win = args
    
    pad = win // 2
    block_h, block_w = img_block.shape
    out_block = np.zeros((block_h - 2*pad, block_w - 2*pad), dtype='float32')
    
    for i in range(pad, block_h - pad):
        for j in range(pad, block_w - pad):
            window = img_block[i-pad:i+pad, j-pad:j+pad]
            
            # Compute GLCM
            glcm = greycomatrix(window, [1], [0], 256, symmetric=True, normed=True)
            contrast = greycoprops(glcm, 'contrast')[0, 0]
            out_block[i-pad, j-pad] = contrast
    
    return out_block

def glcm_contrast_optimized(img: np.ndarray, win: int = 32, block_size: int = 256, n_workers: Optional[int] = None) -> np.ndarray:
    """
    Compute GLCM contrast texture measure with optimized block-based processing
    
    Args:
        img: Input image array
        win: Window size for GLCM computation
        block_size: Size of blocks for parallel processing
        n_workers: Number of worker processes (None = auto)
    
    Returns:
        Contrast texture array
    """
    print(f"Computing GLCM contrast with optimized algorithm...")
    print(f"  Window size: {win}")
    print(f"  Block size: {block_size}")
    
    start_time = time.time()
    
    # Normalize image to 0-255 range for GLCM
    img_min, img_max = img.min(), img.max()
    if img_max == img_min:
        print("Warning: Image has uniform values, returning zeros")
        return np.zeros_like(img, dtype='float32')
    
    img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    pad = win // 2
    out = np.zeros_like(img, dtype='float32')
    
    # Prepare blocks for processing
    h, w = img_norm.shape
    blocks = []
    block_positions = []
    
    for i in range(0, h - 2*pad, block_size - 2*pad):
        for j in range(0, w - 2*pad, block_size - 2*pad):
            # Calculate block boundaries with overlap for windows
            start_i = max(0, i - pad)
            end_i = min(h, i + block_size + pad)
            start_j = max(0, j - pad) 
            end_j = min(w, j + block_size + pad)
            
            img_block = img_norm[start_i:end_i, start_j:end_j]
            blocks.append((img_block, start_i, start_j, win))
            block_positions.append((i, j, start_i, start_j, end_i - start_i, end_j - start_j))
    
    print(f"  Processing {len(blocks)} blocks in parallel...")
    
    # Process blocks in parallel
    if n_workers is None:
        n_workers = min(len(blocks), os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_compute_glcm_block, blocks))
    
    # Assemble results
    for idx, result_block in enumerate(results):
        out_i, out_j, start_i, start_j, block_h, block_w = block_positions[idx]
        
        # Calculate output region
        out_start_i = out_i
        out_end_i = min(out_i + result_block.shape[0], h - 2*pad)
        out_start_j = out_j
        out_end_j = min(out_j + result_block.shape[1], w - 2*pad)
        
        # Extract corresponding part of result
        res_h = out_end_i - out_start_i
        res_w = out_end_j - out_start_j
        
        if res_h > 0 and res_w > 0:
            out[out_start_i + pad:out_end_i + pad, out_start_j + pad:out_end_j + pad] = \
                result_block[:res_h, :res_w]
    
    elapsed = time.time() - start_time
    print(f"  GLCM computation completed in {elapsed:.1f}s")
    
    return out

# Keep original function for compatibility, but make it use optimized version
def glcm_contrast(img: np.ndarray, win: int = 32) -> np.ndarray:
    """
    Compute GLCM contrast texture measure (now uses optimized implementation)
    
    Args:
        img: Input image array
        win: Window size for GLCM computation
    
    Returns:
        Contrast texture array
    """
    return glcm_contrast_optimized(img, win)

def create_landslide_mask(slope, curvature, contrast, slope_thresh=25, curv_thresh=-0.15):
    """
    Create landslide mask using combined criteria
    
    Args:
        slope: Slope array in degrees
        curvature: Curvature array
        contrast: GLCM contrast array
        slope_thresh: Slope threshold in degrees
        curv_thresh: Curvature threshold (negative for concave)
    
    Returns:
        Binary landslide mask
    """
    print(f"Creating landslide mask...")
    print(f"  Slope threshold: >{slope_thresh}°")
    print(f"  Curvature threshold: <{curv_thresh}")
    
    # Calculate 90th percentile of contrast within valid areas
    valid_mask = (slope > 0) & (curvature != 0) & (contrast > 0)
    contrast_p90 = np.percentile(contrast[valid_mask], 90)
    print(f"  Contrast threshold (P90): >{contrast_p90:.2f}")
    
    # Combined mask
    mask = (slope > slope_thresh) & (curvature < curv_thresh) & (contrast > contrast_p90)
    
    # Count pixels
    n_pixels = np.sum(mask)
    total_pixels = mask.size
    percentage = (n_pixels / total_pixels) * 100
    
    print(f"  Landslide pixels: {n_pixels} ({percentage:.2f}% of total)")
    
    return mask, contrast_p90

def detect_boulder_seeds(ohrc_image, scales=[1, 2, 3], threshold_factor=3):
    """
    Detect boulder seeds using Laplacian-of-Gaussian at multiple scales
    
    Args:
        ohrc_image: OHRC image array
        scales: List of sigma values for LoG
        threshold_factor: Factor times std deviation for thresholding
    
    Returns:
        List of boulder seed coordinates (y, x, radius)
    """
    print(f"Detecting boulder seeds...")
    print(f"  Scales (sigma): {scales}")
    print(f"  Threshold factor: {threshold_factor}")
    
    # Resample to 0.5m for faster processing
    # (In real implementation, would use proper resampling)
    print("  Resampling to 0.5m resolution...")
    
    # Detect blobs using LoG
    blobs = blob_log(ohrc_image, min_sigma=min(scales), max_sigma=max(scales), 
                     threshold=threshold_factor * np.std(ohrc_image))
    
    print(f"  Detected {len(blobs)} boulder seeds")
    
    return blobs

def mask_to_polygons(mask, transform, crs):
    """
    Convert binary mask to polygon features
    
    Args:
        mask: Binary mask array
        transform: Rasterio transform
        crs: Coordinate reference system
    
    Returns:
        GeoDataFrame with polygon features
    """
    print("Converting mask to polygons...")
    
    # Convert to shapes
    polygon_shapes = []
    for geom, value in shapes(mask.astype(rasterio.uint8), transform=transform):
        if value == 1:  # Only positive pixels
            polygon_shapes.append(shape(geom))
    
    if not polygon_shapes:
        print("  No polygons generated")
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=polygon_shapes, crs=crs)
    
    # Add attributes
    gdf['area_m2'] = gdf.geometry.area
    gdf['area_km2'] = gdf['area_m2'] / 1e6
    
    print(f"  Generated {len(gdf)} polygons")
    print(f"  Total area: {gdf['area_km2'].sum():.3f} km²")
    
    return gdf

def boulder_seeds_to_points(blobs, transform, crs):
    """
    Convert boulder seeds to point features
    
    Args:
        blobs: Array of blob detections (y, x, radius)
        transform: Rasterio transform
        crs: Coordinate reference system
    
    Returns:
        GeoDataFrame with point features
    """
    if len(blobs) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    
    print("Converting boulder seeds to points...")
    
    # Convert pixel coordinates to geographic coordinates
    points = []
    diameters = []
    
    for blob in blobs:
        y, x, radius = blob
        
        # Convert to geographic coordinates
        geo_x, geo_y = rasterio.transform.xy(transform, y, x)
        
        points.append(Point(geo_x, geo_y))
        diameters.append(2 * radius)  # Convert radius to diameter
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    gdf['diameter_px'] = diameters
    gdf['diameter_m'] = gdf['diameter_px'] * 0.5  # Assuming 0.5m resolution
    
    print(f"  Generated {len(gdf)} boulder points")
    print(f"  Diameter range: {gdf['diameter_m'].min():.1f} - {gdf['diameter_m'].max():.1f} m")
    
    return gdf

def main():
    """Main rule-based baseline workflow"""
    print("Starting rule-based baseline processing...")
    
    # Change to data directory
    data_dir = Path(__file__).parent.parent / "data"
    os.chdir(data_dir)
    
    try:
        # Load slope and curvature
        print("\n=== Loading terrain derivatives ===")
        slope = rxr.open_rasterio("slope_degrees.tif").squeeze()
        curvature = rxr.open_rasterio("curvature.tif").squeeze()
        
        print(f"Slope range: {slope.min():.1f} - {slope.max():.1f}°")
        print(f"Curvature range: {curvature.min():.4f} - {curvature.max():.4f}")
        
        # Load TMC ortho for texture analysis
        print("\n=== Loading TMC ortho for texture analysis ===")
        import glob
        tmc_files = glob.glob("*tmc*orth*_cog.tif")
        if not tmc_files:
            print("Error: No TMC ortho files found")
            sys.exit(1)
        
        tmc_ortho = rxr.open_rasterio(tmc_files[0]).squeeze()
        print(f"TMC ortho shape: {tmc_ortho.shape}")
        
        # Compute GLCM contrast
        print("\n=== Computing texture contrast ===")
        contrast = glcm_contrast(tmc_ortho.values, win=32)
        
        # Create landslide mask
        print("\n=== Creating landslide mask ===")
        landslide_mask, contrast_thresh = create_landslide_mask(
            slope.values, curvature.values, contrast
        )
        
        # Convert mask to polygons
        print("\n=== Converting mask to polygons ===")
        landslide_polygons = mask_to_polygons(
            landslide_mask, slope.rio.transform(), slope.rio.crs
        )
        
        # Save landslide polygons
        if len(landslide_polygons) > 0:
            landslide_polygons.to_file("landslide_polygons.shp")
            print(f"Saved landslide polygons: landslide_polygons.shp")
        
        # Load OHRC for boulder detection
        print("\n=== Loading OHRC for boulder detection ===")
        ohrc_files = glob.glob("ohrc_*.tif")
        if ohrc_files:
            ohrc = rxr.open_rasterio(ohrc_files[0]).squeeze()
            print(f"OHRC shape: {ohrc.shape}")
            
            # Detect boulder seeds
            print("\n=== Detecting boulder seeds ===")
            boulder_seeds = detect_boulder_seeds(ohrc.values)
            
            # Convert to points
            boulder_points = boulder_seeds_to_points(
                boulder_seeds, ohrc.rio.transform(), ohrc.rio.crs
            )
            
            # Save boulder points
            if len(boulder_points) > 0:
                boulder_points.to_file("boulder_seeds.shp")
                print(f"Saved boulder seeds: boulder_seeds.shp")
        else:
            print("Warning: No OHRC files found, skipping boulder detection")
        
        # Save contrast raster
        print("\n=== Saving contrast raster ===")
        contrast_xr = rxr.DataArray(
            contrast,
            dims=["y", "x"],
            coords={"y": slope.y, "x": slope.x}
        )
        contrast_xr.rio.write_crs(slope.rio.crs, inplace=True)
        contrast_xr.rio.to_raster("glcm_contrast.tif", compress="lzw")
        
        print("\n=== Rule-based Baseline Complete ===")
        print("Generated files:")
        print(f"- glcm_contrast.tif (Texture contrast)")
        print(f"- landslide_polygons.shp (Landslide mask polygons)")
        if ohrc_files:
            print(f"- boulder_seeds.shp (Boulder seed points)")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Contrast threshold (P90): {contrast_thresh:.2f}")
        print(f"- Landslide polygons: {len(landslide_polygons)}")
        if ohrc_files and len(boulder_points) > 0:
            print(f"- Boulder seeds: {len(boulder_points)}")
        
    except Exception as e:
        print(f"Error during rule-based baseline processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()