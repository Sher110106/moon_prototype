# Lunar Landslide Detection Prototype

A six-week prototype for automated detection of landslides and boulders on the lunar surface using Chandrayaan-2 data.

## Overview

This prototype combines rule-based methods with lightweight machine learning to detect:
- **Landslides**: Using terrain analysis (slope, curvature) and texture features
- **Boulders**: Using multi-scale blob detection and deep learning validation

**Target Area**: 20 km × 20 km window centered at 6.20°S, 226.40°E  
**Data Sources**: CH-2 TMC Ortho, TMC-DTM, and OHRC imagery  
**Performance Goals**: IoU ≥ 0.50, AP50 ≥ 0.65, < 20 min processing time

## Quick Start

```bash
# 1. Set up environment
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai
source scripts/00_env_setup/gdal_bashrc_snippet.sh

# 2. Run complete pipeline
python run_prototype.py --all
```

## Project Structure

```
moon_prototype/
├── data/                     # Input data and AOI definition
│   └── aoi.geojson          
├── scripts/                  # Processing pipeline scripts
│   ├── 00_env_setup/        # Environment configuration
│   ├── 01_data_acquisition.sh
│   ├── 02_raster_preprocessing.py
│   ├── 03_coregistration.sh
│   ├── 04_hapke_normalisation.py
│   ├── 05_terrain_derivatives.py
│   ├── 06_rule_based_baseline.py
│   ├── 07_annotation_sprint.md
│   ├── 08_light_ml_models.py
│   ├── 09_fusion_and_filter.py
│   ├── 10_metrics_audit.py
│   └── 11_visuals_packaging.py
├── notebooks/               # Analysis and reporting
│   └── Prototype_Report.ipynb
├── reports/                 # Documentation and figures
│   └── PROTOTYPE_WORKFLOW.md
├── outputs/                 # Final results
│   ├── prototype_metrics.csv
│   └── aoi_landslide_boulder.gpkg
└── run_prototype.py         # Main CLI entry point
```

## Prerequisites

### System Requirements
- **CPU**: 8+ cores recommended
- **RAM**: 16+ GB 
- **Storage**: 50+ GB available space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Dependencies
- **Python**: 3.10
- **Conda**: For environment management
- **GDAL**: 3.8+ for geospatial processing
- **QGIS**: For manual co-registration and annotation
- **LabelMe**: For boulder annotation

### Data Access
- Access to ISRO's Pradan data portal
- `pradan-cli` tool configured with credentials

## Installation

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd moon_prototype

# Create conda environment
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai

# Configure GDAL settings
source scripts/00_env_setup/gdal_bashrc_snippet.sh
```

### 2. Verify Installation
```bash
python run_prototype.py --list
```

## Usage

### Run Individual Steps
```bash
# Data acquisition
python run_prototype.py --step 1

# Preprocessing only
python run_prototype.py --step 2

# Steps 2-5 (preprocessing through terrain analysis)
python run_prototype.py --step 2-5
```

### Run Complete Pipeline
```bash
# Full automated pipeline
python run_prototype.py --all

# With custom AOI
python run_prototype.py --aoi path/to/custom_aoi.geojson --all
```

### Key Manual Steps
- **Step 3**: GCP collection in QGIS for co-registration
- **Step 7**: Manual annotation of landslides and boulders

## Pipeline Steps

| Step | Description | Duration | Manual? |
|------|-------------|----------|---------|
| 0 | Environment setup | 30 min | ✓ |
| 1 | Data acquisition | 2-4 hours | - |
| 2 | Raster preprocessing | 1 hour | - |
| 3 | Co-registration | 2 hours | ✓ |
| 4 | Hapke normalization | 15 min | - |
| 5 | Terrain derivatives | 30 min | - |
| 6 | Rule-based baseline | 1 hour | - |
| 7 | Annotation sprint | 2-3 days | ✓ |
| 8 | ML model training | 4-6 hours | - |
| 9 | Fusion & filtering | 1 hour | - |
| 10 | Metrics audit | 30 min | - |
| 11 | Visuals & packaging | 1 hour | - |

## Key Algorithms

### Photometric Correction
- **Cosine correction**: I_cos = I_raw / cos(i)
- **Hapke model**: Advanced bidirectional reflectance correction

### Terrain Analysis
- **Slope**: Using RichDEM with Gaussian smoothing
- **Curvature**: Profile curvature for landslide detection

### Feature Detection
- **Landslides**: Combined slope (>25°) + curvature (<-0.15) + texture contrast
- **Boulders**: Multi-scale Laplacian-of-Gaussian + YOLOv8 validation

### Machine Learning
- **U-Net**: ResNet18 encoder for landslide segmentation
- **YOLOv8n**: Lightweight object detection for boulders

## Output Files

### Primary Results
- `outputs/aoi_landslide_boulder.gpkg`: Validated features
- `outputs/prototype_metrics.csv`: Performance metrics
- `reports/figures/`: Static map visualizations

### Intermediate Products
- `data/*_cog.tif`: Cloud-optimized rasters
- `data/slope_degrees.tif`, `data/curvature.tif`: Terrain derivatives
- `data/landslide_polygons.shp`: Rule-based detections

## Performance Targets

- **Landslide Detection**: IoU ≥ 0.50, Precision ≥ 0.70
- **Boulder Detection**: AP50 ≥ 0.65, mean diameter error < 20%
- **Processing Time**: < 20 minutes total pipeline time
- **Cost**: < $86 cloud compute (240 hours @ $0.35/hr)

## Troubleshooting

### Common Issues

**GDAL Errors**
```bash
export GDAL_CACHEMAX=1024
export CPL_LOG=/tmp/gdal_log.txt
```

**Memory Issues**
- Reduce tile sizes in processing scripts
- Process smaller AOIs first

**Pradan CLI Issues**
- Check credentials and quota
- Verify AOI intersects with data coverage

### Debug Mode
```bash
# Enable verbose logging
export GDAL_CONFIG_FILE=/tmp/gdal_debug.conf
python run_prototype.py --step X
```

## Development

### Adding New Steps
1. Create script in `scripts/XX_new_step.py`
2. Add to `run_prototype.py` step definitions
3. Update documentation

### Testing
```bash
# Test with small AOI
python run_prototype.py --aoi test_small.geojson --step 1-6
```

## References

- **Chandrayaan-2**: [ISRO Mission Overview](https://www.isro.gov.in/chandrayaan2-home-1)
- **Hapke Model**: Hapke, B. (2012). Theory of reflectance and emittance spectroscopy
- **RichDEM**: Horn, B.K.P. (1981). Hill shading and the reflectance map

## License

This prototype is developed for research purposes. Please cite appropriately if used in publications.

## Contact

For technical issues or questions about the prototype implementation, please refer to the detailed workflow documentation in `reports/PROTOTYPE_WORKFLOW.md`.

---

**Status**: Prototype implementation complete  
**Last Updated**: $(date +%Y-%m-%d)  
**Target Completion**: 30 calendar days, $86 cloud spend