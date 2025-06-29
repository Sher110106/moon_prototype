# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a lunar landslide detection prototype that processes Chandrayaan-2 satellite imagery to automatically detect landslides and boulders on the Moon's surface. The system combines rule-based geomorphological analysis with lightweight machine learning models to achieve target performance metrics (IoU ≥ 0.50, AP50 ≥ 0.65) within 20 minutes processing time.

## Core Architecture

### Pipeline Structure
The system is organized as an 11-step sequential pipeline orchestrated by `run_prototype.py`:

1. **Data Acquisition** (`01_data_acquisition.sh`) - Downloads CH-2 TMC Ortho, TMC-DTM, and OHRC imagery via pradan-cli
2. **Raster Preprocessing** (`02_raster_preprocessing.py`) - COG conversion, reprojection to EPSG:104903, grid alignment
3. **Co-registration** (`03_coregistration.sh`) - GCP-based alignment using QGIS georeferencer
4. **Photometric Correction** (`04_hapke_normalisation.py`) - Cosine and Hapke model corrections for illumination
5. **Terrain Analysis** (`05_terrain_derivatives.py`) - RichDEM-based slope/curvature computation with Gaussian smoothing
6. **Rule-based Detection** (`06_rule_based_baseline.py`) - GLCM texture analysis and feature masking
7. **Manual Annotation** (`07_annotation_sprint.md`) - QGIS/LabelMe workflow for training data
8. **ML Training** (`08_light_ml_models.py`) - U-Net landslide segmentation + YOLOv8 boulder detection
9. **Fusion & Filtering** (`09_fusion_and_filter.py`) - Cross-scale validation with physics constraints
10. **Metrics Evaluation** (`10_metrics_audit.py`) - Performance assessment and timing analysis
11. **Visualization** (`11_visuals_packaging.py`) - Static map generation and result packaging

### Key Algorithms
- **Photometric Correction**: Implements both cosine (I_cos = I_raw / cos(i)) and Hapke bidirectional reflectance models
- **Terrain Analysis**: Uses RichDEM with 3×3 Gaussian pre-filtering to compute slope and curvature derivatives
- **Feature Detection**: Combines slope (>25°) + curvature (<-0.15) + GLCM texture contrast thresholds
- **Boulder Detection**: Multi-scale Laplacian-of-Gaussian followed by YOLOv8 validation
- **Physics Filtering**: Shadow geometry constraints (h = L × tan(θ_s)) to validate detections

## Environment Setup

```bash
# Create conda environment
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai

# Configure GDAL for optimal performance
source scripts/00_env_setup/gdal_bashrc_snippet.sh
```

The environment includes Python 3.10 with geospatial stack (GDAL 3.8, rasterio, rioxarray, richdem), ML frameworks (PyTorch, ultralytics), and domain-specific tools (pradan-cli for data access).

## Common Development Commands

### Pipeline Execution
```bash
# Run complete pipeline
python run_prototype.py --all

# Run specific steps
python run_prototype.py --step 1          # Data acquisition only
python run_prototype.py --step 2-5        # Preprocessing through terrain analysis
python run_prototype.py --step 6          # Rule-based detection only

# List all available steps
python run_prototype.py --list

# Use custom AOI
python run_prototype.py --aoi path/to/custom.geojson --all

# Environment setup (Step 0)
python run_prototype.py --step 0          # Shows environment setup instructions
```

### Individual Script Execution
```bash
# Run preprocessing with proper working directory
cd data && python ../scripts/02_raster_preprocessing.py

# Execute terrain analysis
python scripts/05_terrain_derivatives.py

# Generate rule-based baseline
python scripts/06_rule_based_baseline.py

# All scripts can be run directly from project root:
python scripts/04_hapke_normalisation.py
python scripts/08_light_ml_models.py
```

### Debug and Troubleshooting
```bash
# Enable GDAL verbose logging and performance optimization
source scripts/00_env_setup/gdal_bashrc_snippet.sh

# Or manually set GDAL environment variables:
export GDAL_CACHEMAX=1024
export CPL_LOG=/tmp/gdal_log.txt

# Test with smaller AOI for development
python run_prototype.py --aoi test_small.geojson --step 1-6

# Check pipeline interruption handling
# Pipeline supports Ctrl+C graceful termination with progress summary
```

### Environment Management
```bash
# Create environment from scratch
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai

# Verify environment setup
conda list | grep -E "(gdal|pytorch|ultralytics|richdem)"

# Update environment if needed
conda env update -f scripts/00_env_setup/environment.yml
```

## Critical Data Flow

### Input Requirements
- **AOI Definition**: GeoJSON polygon in `data/aoi.geojson` (default: 20km×20km at 6.20°S, 226.40°E)
- **Pradan Access**: Configured pradan-cli with ISRO credentials for CH-2 data download
- **Manual Steps**: GCP collection (Step 3) and annotation work (Step 7) require human intervention

### Key Intermediate Products
- **COG Rasters**: `*_cog.tif` files with optimized tiling and compression
- **Terrain Derivatives**: `slope_degrees.tif`, `curvature.tif` from RichDEM processing
- **Photometric Products**: `tmc_tile_cosine.tif`, `tmc_tile_hapke.tif` for algorithm comparison
- **Feature Maps**: `landslide_polygons.shp`, `boulder_seeds.shp` from rule-based detection

### Final Outputs
- **Validated Features**: `outputs/aoi_landslide_boulder.gpkg` - spatially validated landslide/boulder features
- **Performance Metrics**: `outputs/prototype_metrics.csv` - IoU, precision, recall, AP50, timing data
- **Visualizations**: `reports/figures/` - static map plates for presentation

## Manual Workflow Integration

### Step 3: Co-registration
Requires QGIS interaction for Ground Control Point (GCP) collection:
- Load TMC ortho as reference and OHRC as unreferenced
- Collect 10 GCPs on crater rims near AOI center
- Target RMSE < 0.5 pixels (≈12cm accuracy)
- Save as `gcp_points.txt` for automated processing

### Step 7: Annotation Sprint
Critical 2-3 day manual annotation phase:
- **Landslides**: 30 polygons in QGIS with confidence scoring (1-5 scale)
- **Boulders**: 300 annotations in LabelMe with bounding boxes
- **Dataset Split**: 70% train / 15% val / 15% test with spatial separation
- Generates training data for ML models in steps 8-9

## Performance Constraints

### Resource Limits
- **Memory**: 16GB+ RAM required for large raster processing
- **Storage**: 50GB+ for CH-2 data and intermediate products
- **Processing Time**: Target <20 minutes total pipeline execution
- **Cost**: $86 budget for cloud compute (240 hours @ $0.35/hr)

### Quality Thresholds
- **Landslide Detection**: IoU ≥ 0.50, Precision ≥ 0.70
- **Boulder Detection**: AP50 ≥ 0.65, diameter error <20%
- **Co-registration**: RMSE < 0.5 pixels for geometric accuracy

## Error Handling Patterns

The pipeline uses consistent error handling:
- All scripts include proper shebangs and execute permissions
- GDAL operations wrapped with error checking and logging via `run_command()` utility functions
- Processing steps validate input files before execution
- Failed steps terminate pipeline with informative error messages and non-zero exit codes
- Timing and resource usage tracked for performance optimization
- Graceful interrupt handling (Ctrl+C) with progress summary
- Real-time command output display during execution

## Testing and Validation

This prototype uses integration testing rather than unit tests:
- **End-to-end validation**: Complete pipeline execution on test AOI
- **Performance benchmarking**: Target metrics validation (IoU ≥ 0.50, AP50 ≥ 0.65)
- **Resource monitoring**: Memory usage and execution time tracking in `run_prototype.py`
- **Manual validation**: Visual inspection of intermediate products
- **Metrics audit**: Automated performance evaluation in Step 10

```bash
# Run pipeline on test data
python run_prototype.py --aoi test_small.geojson --step 1-11

# Validate specific processing steps
python run_prototype.py --step 2-6  # Preprocessing validation
python run_prototype.py --step 10   # Metrics evaluation only
```

## Extension Points

### Adding New Processing Steps
1. Create script in `scripts/XX_new_step.py` following naming convention
2. Add step function to `run_prototype.py` step definitions
3. Include in pipeline orchestration logic
4. Update documentation and help text

### Custom Algorithm Integration
- Photometric correction models in `04_hapke_normalisation.py`
- Terrain analysis algorithms in `05_terrain_derivatives.py`
- Feature detection rules in `06_rule_based_baseline.py`
- ML model architectures in `08_light_ml_models.py`

## Technical Architecture Notes

### Working Directory Requirements
- Main pipeline orchestrator (`run_prototype.py`) changes to project root automatically
- Individual scripts expect to be run from project root directory
- Data processing occurs in `data/` subdirectory with relative paths
- All scripts use `Path(__file__).parent` for robust path resolution

### Pipeline State Management
- Each step validates required input files before execution
- Intermediate products use consistent naming conventions (`*_cog.tif`, `*_degrees.tif`)
- Failed steps exit immediately with descriptive error messages
- No automatic retry logic - manual intervention required for failures
- Pipeline execution summary shows completed steps and total runtime

### Resource Optimization
- GDAL caching configured via `gdal_bashrc_snippet.sh` (1GB cache)
- Conda environment uses optimized geospatial stack with CUDA support
- Processing designed for 16GB+ RAM systems
- Cloud-Optimized GeoTIFF (COG) format used throughout for efficient I/O