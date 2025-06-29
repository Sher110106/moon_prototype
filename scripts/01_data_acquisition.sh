#!/usr/bin/env bash

# Data Acquisition Script for Lunar Landslide Prototype
# AOI: 20 km × 20 km window centred at 6.20 °S, 226.40 °E

set -e

echo "Starting data acquisition for lunar landslide prototype..."

# Change to data directory
cd "$(dirname "$0")/../data"

# 1-A: AOI polygon already created as aoi.geojson

# 1-B: Discover matching product IDs
echo "Searching for TMC2 products (ortho + DTM)..."
pradan-cli search --aoi aoi.geojson --instrument TMC2

echo "Searching for OHRC products..."
pradan-cli search --aoi aoi.geojson --instrument OHRC

# 1-C: Download only intersecting rows (≈ 8% of full strips)
echo "Fetching TMC ortho data..."
pradan-cli fetch ch2_tmc_*oth* --clip aoi.geojson

echo "Fetching TMC DTM data..."
pradan-cli fetch ch2_tmc_*dtm* --clip aoi.geojson

echo "Fetching OHRC imagery..."
pradan-cli fetch ch2_ohrc_*img* --clip aoi.geojson

echo "Data acquisition complete!"
echo "Note: Using --clip to minimize download size (<4GB instead of >2GB)"

# List downloaded files
echo "Downloaded files:"
ls -la *.img *.tif 2>/dev/null || echo "No .img or .tif files found in current directory"