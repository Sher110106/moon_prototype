#!/usr/bin/env python3
"""
09_fusion_and_filter.py
Cross-scale fusion and physics filter for landslide validation.

Implements:
- U-Net inference on full TMC tiles
- YOLO inference on OHRC crops
- Physics-based validation filters
- Shadow geometry checks
"""

import os
import sys
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
import cv2
from pathlib import Path
import json
import warnings
from skimage.morphology import skeletonize

# Import model architectures
from ultralytics import YOLO
import segmentation_models_pytorch as smp

from utils import load_config

def load_models(model_dir):
    """Load trained U-Net and YOLO models."""
    # Load U-Net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    unet_model.load_state_dict(torch.load(f"{model_dir}/best_landslide_unet.pth", map_location=device))
    unet_model.to(device)
    unet_model.eval()
    
    # Load YOLO
    yolo_model = YOLO(f"{model_dir}/yolo_boulder/weights/best.pt")
    
    return unet_model, yolo_model

def create_sliding_windows(width, height, tile_size=512, overlap=64):
    """Create sliding windows for tile-based inference."""
    windows = []
    step = tile_size - overlap
    
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            windows.append(Window(x, y, tile_size, tile_size))
    
    # Handle edge cases
    if height % step != 0:
        y = height - tile_size
        for x in range(0, width - tile_size + 1, step):
            windows.append(Window(x, y, tile_size, tile_size))
    
    if width % step != 0:
        x = width - tile_size
        for y in range(0, height - tile_size + 1, step):
            windows.append(Window(x, y, tile_size, tile_size))
    
    return windows

def predict_landslides_full_tile(unet_model, tmc_path, slope_path, curv_path, output_path, tile_size=512):
    """Run U-Net inference on full TMC tile with sliding windows."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Open input rasters
    with rasterio.open(tmc_path) as tmc_src, \
         rasterio.open(slope_path) as slope_src, \
         rasterio.open(curv_path) as curv_src:
        
        # Get metadata
        profile = tmc_src.profile.copy()
        profile.update({'count': 1, 'dtype': 'float32'})
        
        # Initialize output array
        height, width = tmc_src.height, tmc_src.width
        prediction = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Create sliding windows
        windows = create_sliding_windows(width, height, tile_size)
        
        print(f"Processing {len(windows)} windows...")
        
        with torch.no_grad():
            for i, window in enumerate(windows):
                if i % 10 == 0:
                    print(f"Processing window {i+1}/{len(windows)}")
                
                # Read window data
                tmc_tile = tmc_src.read(1, window=window).astype(np.float32)
                slope_tile = slope_src.read(1, window=window).astype(np.float32)
                curv_tile = curv_src.read(1, window=window).astype(np.float32)
                
                # Skip if any tile is empty
                if tmc_tile.size == 0 or slope_tile.size == 0 or curv_tile.size == 0:
                    continue
                
                # Normalize inputs
                tmc_tile = (tmc_tile - tmc_tile.min()) / (tmc_tile.max() - tmc_tile.min() + 1e-8)
                slope_tile = slope_tile / 90.0  # Normalize to 0-1
                curv_tile = np.clip(curv_tile, -1, 1)  # Clip extreme values
                
                # Stack channels
                input_tile = np.stack([tmc_tile, slope_tile, curv_tile], axis=0)
                input_tensor = torch.from_numpy(input_tile).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = torch.sigmoid(unet_model(input_tensor))
                    pred_tile = output.cpu().numpy()[0, 0]
                
                # Add to prediction map with overlap handling
                y, x = window.row_off, window.col_off
                h, w = pred_tile.shape
                prediction[y:y+h, x:x+w] += pred_tile
                count_map[y:y+h, x:x+w] += 1
        
        # Average overlapping predictions
        prediction = np.divide(prediction, count_map, out=np.zeros_like(prediction), where=count_map!=0)
        
        # Save prediction
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
    
    print(f"Landslide prediction saved to: {output_path}")
    return output_path

def raster_to_polygons(raster_path, threshold=0.5, min_area=100):
    """Convert raster predictions to polygon features."""
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Threshold and clean
    binary = (image > threshold).astype(np.uint8)
    
    # Remove small objects
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Convert contour to polygon coordinates
        contour = contour.squeeze()
        if len(contour) < 3:
            continue
        
        # Transform pixel coordinates to geographic coordinates
        coords = []
        for point in contour:
            x, y = rasterio.transform.xy(transform, point[1], point[0])
            coords.append((x, y))
        
        if len(coords) >= 3:
            polygon = Polygon(coords)
            if polygon.is_valid and polygon.area > 0:
                polygons.append(polygon)
    
    # Create GeoDataFrame
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
        gdf['area'] = gdf.geometry.area
        gdf['perimeter'] = gdf.geometry.length
        gdf = gdf.reset_index(drop=True)
    else:
        gdf = gpd.GeoDataFrame(columns=['geometry', 'area', 'perimeter'], crs=crs)
    
    return gdf

def crop_ohrc_window(ohrc_path, center_point, size=512):
    """Crop OHRC around a center point."""
    with rasterio.open(ohrc_path) as src:
        # Convert center point to pixel coordinates
        row, col = src.index(center_point.x, center_point.y)
        
        # Define window
        half_size = size // 2
        window = Window(
            col - half_size, row - half_size,
            size, size
        )
        
        # Read window
        try:
            crop = src.read(window=window)
            transform = src.window_transform(window)
            
            # Handle single band
            if crop.shape[0] == 1:
                crop = crop[0]
            
            return crop, transform
        except:
            return None, None

def validate_with_yolo(yolo_model, ohrc_crop):
    """Run YOLO inference on OHRC crop to detect boulders."""
    if ohrc_crop is None or ohrc_crop.size == 0:
        return False, []
    
    # Normalize crop to 0-255 for YOLO
    if ohrc_crop.dtype != np.uint8:
        crop_norm = ((ohrc_crop - ohrc_crop.min()) / 
                    (ohrc_crop.max() - ohrc_crop.min() + 1e-8) * 255).astype(np.uint8)
    else:
        crop_norm = ohrc_crop
    
    # Convert to 3-channel if needed
    if len(crop_norm.shape) == 2:
        crop_norm = np.stack([crop_norm] * 3, axis=-1)
    
    # Run YOLO inference
    results = yolo_model(crop_norm, verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        return False, []
    
    # Extract boulder masks and compute diameters
    boulders = []
    masks = results[0].masks.data.cpu().numpy()
    
    for mask in masks:
        # Calculate boulder diameter: d = 2 * sqrt(A / pi)
        area = np.sum(mask > 0.5)
        if area > 10:  # Minimum area threshold
            diameter = 2 * np.sqrt(area / np.pi)
            boulders.append({'area': area, 'diameter': diameter, 'mask': mask})
    
    has_boulders = len(boulders) > 0
    return has_boulders, boulders

def estimate_boulder_height(mask, ohrc_crop, pixel_size_m, sun_elev_deg, sun_az_deg,
                            slope_deg, aspect_deg):
    """Estimate boulder height using shadow skeleton length and slope-aware correction.

    Args:
        mask: 2-D boolean numpy array for one boulder (YOLO mask resized to crop).
        ohrc_crop: 2-D grayscale numpy array of the same size as mask.
        pixel_size_m: OHRC pixel size in metres (≈0.23).
        sun_elev_deg: Solar elevation angle.
        sun_az_deg: Solar azimuth angle east of north.
        slope_deg: Local slope in degrees.
        aspect_deg: Local aspect in degrees.
    Returns:
        height in metres or None if shadow not found.
    """
    # 1. shadow extraction – assume shadows are darker than 30 % percentile
    thresh_val = np.percentile(ohrc_crop, 30)
    shadow = (ohrc_crop < thresh_val).astype(np.uint8)

    # Remove the boulder body from shadow mask to avoid self-mask
    shadow[mask > 0.5] = 0

    # Skeletonize the shadow mask
    if np.count_nonzero(shadow) < 10:
        return None
    skel = skeletonize(shadow > 0)
    coords = np.column_stack(np.nonzero(skel))
    if coords.shape[0] < 5:
        return None

    # PCA for main axis = shadow direction
    mean = coords.mean(axis=0)
    centered = coords - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    length_px = s[0] * 2 / shadow.shape[0]  # approximate – scale factor cancels later
    # Better: project coords on principal axis and get range
    vec = vt[0]
    proj = centered @ vec
    length_px = proj.max() - proj.min()
    if length_px <= 0:
        return None

    L = length_px * pixel_size_m  # metres
    # Slope component along sun plane
    beta_parallel = slope_deg * np.cos(np.deg2rad(sun_az_deg - aspect_deg))
    h = L * np.tan(np.deg2rad(sun_elev_deg)) + L * np.tan(np.deg2rad(beta_parallel))
    return max(h, 0.01)

def physics_shadow_filter(boulders, ohrc_crop, pixel_size_m, sun_elev, sun_az,
                          slope_raster, aspect_raster, transform):
    """Advanced physics validity check using height/diameter ratio."""
    if not boulders:
        return False

    valid_count = 0
    for b in boulders:
        y, x = np.argwhere(b['mask'] > 0.5)[0]
        # Convert crop pixel to world coords using transform of crop
        # transform is window transform; we approximate by center pixel
        world_x, world_y = rasterio.transform.xy(transform, y, x)

        # Sample slope & aspect rasters at this location
        try:
            slope_value = list(slope_raster.sample([(world_x, world_y)]))[0][0]
            aspect_value = list(aspect_raster.sample([(world_x, world_y)]))[0][0]
        except Exception:
            slope_value = 0.0
            aspect_value = 0.0

        h = estimate_boulder_height(b['mask'], ohrc_crop, pixel_size_m,
                                     sun_elev, sun_az,
                                     slope_value, aspect_value)
        if h is None:
            continue
        ratio = h / b['diameter']
        if 0.5 < ratio < 2.5:
            valid_count += 1

    return valid_count > 0

def calculate_mean_slope(polygon, slope_path):
    """Calculate mean slope within polygon."""
    try:
        with rasterio.open(slope_path) as src:
            # Mask raster with polygon
            masked, _ = mask(src, [polygon], crop=True, nodata=np.nan)
            slope_values = masked[0]
            
            # Calculate mean slope (excluding nodata)
            valid_slopes = slope_values[~np.isnan(slope_values)]
            if len(valid_slopes) > 0:
                return np.mean(valid_slopes)
            else:
                return 0.0
    except:
        return 0.0

def fusion_and_filter(landslide_polygons, ohrc_path, slope_path, aspect_path, yolo_model,
                      min_slope_threshold=18.0):
    """Main fusion and filtering pipeline."""
    validated_polygons = []
    
    print(f"Processing {len(landslide_polygons)} landslide candidates...")
    
    # Load config for sun parameters
    cfg = load_config()
    sun_elev = cfg['photometric'].get('sun_elevation_degrees', 41.3)
    sun_az = cfg['photometric'].get('sun_azimuth_degrees', 135.0)

    # Open slope and aspect rasters once
    slope_src = rasterio.open(slope_path)
    aspect_src = rasterio.open(aspect_path) if aspect_path and os.path.exists(aspect_path) else None

    for idx, row in landslide_polygons.iterrows():
        polygon = row.geometry
        centroid = polygon.centroid
        
        # Calculate mean slope
        mean_slope = calculate_mean_slope(polygon, slope_path)
        
        # Crop OHRC around centroid
        ohrc_crop, crop_transform = crop_ohrc_window(ohrc_path, centroid, size=512)
        
        # Validate with YOLO
        has_boulders, boulders = validate_with_yolo(yolo_model, ohrc_crop)
        
        # Apply physics filter with advanced check
        if aspect_src is not None:
            physics_valid = physics_shadow_filter(
                boulders, ohrc_crop, 0.23, sun_elev, sun_az,
                slope_src, aspect_src, crop_transform)
        else:
            physics_valid = False
        
        # Decision logic
        validated = False
        validation_reason = ""
        
        if has_boulders and physics_valid:
            validated = True
            validation_reason = "boulder_detected"
        elif mean_slope > min_slope_threshold:
            validated = True
            validation_reason = "high_slope"
        else:
            validation_reason = "rejected"
        
        # Add to results
        polygon_data = {
            'geometry': polygon,
            'area': row.get('area', polygon.area),
            'perimeter': row.get('perimeter', polygon.length),
            'mean_slope': mean_slope,
            'has_boulders': has_boulders,
            'num_boulders': len(boulders),
            'physics_valid': physics_valid,
            'validated': validated,
            'validation_reason': validation_reason
        }
        
        if boulders:
            polygon_data['boulder_diameters'] = [b['diameter'] for b in boulders]
            polygon_data['mean_boulder_diameter'] = np.mean([b['diameter'] for b in boulders])
        
        validated_polygons.append(polygon_data)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(landslide_polygons)} polygons")
    
    # Create output GeoDataFrame
    result_gdf = gpd.GeoDataFrame(validated_polygons, crs=landslide_polygons.crs)
    
    print(f"Validation results:")
    print(f"  Total candidates: {len(result_gdf)}")
    print(f"  Validated: {result_gdf['validated'].sum()}")
    print(f"  By boulders: {(result_gdf['validation_reason'] == 'boulder_detected').sum()}")
    print(f"  By slope: {(result_gdf['validation_reason'] == 'high_slope').sum()}")
    
    # Close rasters
    slope_src.close()
    if aspect_src is not None:
        aspect_src.close()

    return result_gdf

def main():
    """Main fusion and filtering pipeline."""
    if len(sys.argv) < 7:
        print("Usage: python 09_fusion_and_filter.py <model_dir> <tmc_path> <slope_path> <curv_path> <aspect_path> <ohrc_path> [output_dir]")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    tmc_path = sys.argv[2]
    slope_path = sys.argv[3]
    curv_path = sys.argv[4]
    aspect_path = sys.argv[5]
    ohrc_path = sys.argv[6]
    output_dir = sys.argv[7] if len(sys.argv) > 7 else "outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models...")
    unet_model, yolo_model = load_models(model_dir)
    
    print("Running U-Net inference on full tile...")
    prediction_path = f"{output_dir}/landslide_prediction.tif"
    predict_landslides_full_tile(unet_model, tmc_path, slope_path, curv_path, prediction_path)
    
    print("Converting predictions to polygons...")
    landslide_polygons = raster_to_polygons(prediction_path, threshold=0.5, min_area=100)
    
    if len(landslide_polygons) == 0:
        print("No landslide candidates detected!")
        return
    
    print("Running fusion and filtering...")
    validated_results = fusion_and_filter(
        landslide_polygons, ohrc_path, slope_path, aspect_path, yolo_model
    )
    
    # Save results
    output_gpkg = f"{output_dir}/aoi_landslide_boulder.gpkg"
    validated_results.to_file(output_gpkg, driver='GPKG')
    
    # Save summary statistics
    stats = {
        'total_candidates': len(landslide_polygons),
        'validated_landslides': int(validated_results['validated'].sum()),
        'boulder_validated': int((validated_results['validation_reason'] == 'boulder_detected').sum()),
        'slope_validated': int((validated_results['validation_reason'] == 'high_slope').sum()),
        'validation_rate': float(validated_results['validated'].sum() / len(validated_results)),
        'mean_validated_area': float(validated_results[validated_results['validated']]['area'].mean()) if validated_results['validated'].any() else 0,
        'mean_slope_validated': float(validated_results[validated_results['validated']]['mean_slope'].mean()) if validated_results['validated'].any() else 0
    }
    
    with open(f"{output_dir}/fusion_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to: {output_gpkg}")
    print(f"Fusion statistics: {stats}")

if __name__ == "__main__":
    main()