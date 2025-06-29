#!/usr/bin/env bash

# Co-registration Script for Lunar Landslide Prototype
# One-afternoon GCP-based co-registration workflow

set -e

echo "Starting co-registration workflow..."

# Change to data directory
cd "$(dirname "$0")/../data"

echo "=== Step 3-A: Manual QGIS Georeferencer Workflow ==="
echo "Manual steps to perform in QGIS:"
echo "1. Open QGIS → Raster → Georeferencer"
echo "2. Load 'tmc_ortho_cog.tif' as target reference"
echo "3. Load 'ohrc_eq.tif' as unreferenced raster"
echo "4. Drop 10 GCPs on common crater rims (pick near AOI center)"
echo "5. Save GCP points as 'gcp_points.txt'"
echo "6. Target RMSE < 0.5 px (≈ 12 cm)"
echo ""
echo "Press Enter when GCP collection is complete and gcp_points.txt exists..."
read -r

# Check if GCP file exists
if [ ! -f "gcp_points.txt" ]; then
    echo "Error: gcp_points.txt not found. Please complete GCP collection in QGIS first."
    exit 1
fi

echo "=== Step 3-B: Inject GCPs and Warp ==="

# Inject GCPs into the OHRC file
echo "Injecting GCPs into OHRC..."
gdal_translate -of GTiff -gcp_file gcp_points.txt \
    ohrc_eq.tif ohrc_gcp.tif

# Warp using the GCPs
echo "Warping OHRC with GCPs..."
gdalwarp -r cubic -t_srs EPSG:104903 -order 1 \
         ohrc_gcp.tif ohrc_coreg.tif

echo "=== Co-registration Complete ==="
echo "Generated files:"
echo "- ohrc_gcp.tif (OHRC with GCPs)"
echo "- ohrc_coreg.tif (Co-registered OHRC)"

# Check final file
if [ -f "ohrc_coreg.tif" ]; then
    echo "Co-registration successful!"
    echo "File info:"
    gdalinfo ohrc_coreg.tif | head -10
else
    echo "Error: Co-registration failed"
    exit 1
fi

echo ""
echo "Note: Verify co-registration quality by overlaying with TMC ortho in QGIS"
echo "Target accuracy: RMSE < 0.5 px (≈ 12 cm)"