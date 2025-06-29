# Lunar Landslide Detection Prototype - Complete Execution Guide

## Overview

This document provides comprehensive step-by-step instructions for executing the lunar landslide detection prototype. The system processes Chandrayaan-2 satellite imagery through an 11-step pipeline to automatically detect landslides and boulders on the Moon's surface.

**Target Performance**: IoU ≥ 0.50, AP50 ≥ 0.65, processing time < 20 minutes  
**Target Area**: 20 km × 20 km window at 6.20°S, 226.40°E  
**Data Sources**: CH-2 TMC Ortho, TMC-DTM, OHRC imagery

## Prerequisites and System Requirements

### Hardware Requirements
- **CPU**: 8+ cores (recommended)
- **RAM**: 16+ GB (minimum)
- **Storage**: 50+ GB available space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for ML training)

### Software Dependencies
- **Python**: 3.10
- **Conda**: For environment management
- **GDAL**: 3.8+ for geospatial processing
- **QGIS**: For manual co-registration (Step 3) and annotation (Step 7)
- **LabelMe**: For boulder annotation (Step 7)

### Data Access
- Access to ISRO's Pradan data portal
- `pradan-cli` tool configured with valid credentials
- Sufficient quota for downloading ~4GB of satellite data

### Cloud Infrastructure (Optional)
- GCP `n1-standard-8` + NVIDIA T4 GPU
- Cost: ~$0.35/hour, capped at 240 hours = $84
- Storage: 50GB standard disk (~$1.50)

## File Structure Overview

```
moon_prototype/
├── data/                          # Input data and intermediate products
│   ├── aoi.geojson               # Area of Interest definition
│   ├── *_cog.tif                 # Cloud-Optimized GeoTIFF files
│   ├── slope_degrees.tif         # Terrain slope analysis
│   ├── curvature.tif            # Terrain curvature analysis
│   ├── landslide_polygons.shp   # Rule-based landslide detections
│   └── boulder_seeds.shp        # Boulder detection seeds
├── scripts/                      # Processing pipeline scripts
│   ├── 00_env_setup/            # Environment configuration
│   │   ├── environment.yml      # Conda environment specification
│   │   └── gdal_bashrc_snippet.sh # GDAL optimization settings
│   ├── 01_data_acquisition.sh   # Download CH-2 satellite data
│   ├── 02_raster_preprocessing.py # Convert to COG, reproject, align
│   ├── 03_coregistration.sh     # Co-register OHRC with TMC
│   ├── 04_hapke_normalisation.py # Photometric correction
│   ├── 05_terrain_derivatives.py # Compute slope and curvature
│   ├── 06_rule_based_baseline.py # Rule-based feature detection
│   ├── 07_annotation_sprint.md  # Manual annotation instructions
│   ├── 08_light_ml_models.py    # Train U-Net and YOLOv8 models
│   ├── 09_fusion_and_filter.py  # Cross-scale validation
│   ├── 10_metrics_audit.py      # Performance evaluation
│   └── 11_visuals_packaging.py  # Generate visualizations
├── notebooks/                   # Analysis and reporting
│   └── Prototype_Report.ipynb   # Jupyter notebook for results
├── reports/                     # Documentation and figures
│   ├── PROTOTYPE_WORKFLOW.md    # Workflow documentation
│   └── figures/                 # Generated visualizations
├── outputs/                     # Final results
│   ├── prototype_metrics.csv    # Performance metrics
│   └── aoi_landslide_boulder.gpkg # Validated feature detections
└── run_prototype.py            # Main CLI orchestrator
```

## Step-by-Step Execution Guide

### STEP 0: Environment Setup and Configuration

**Purpose**: Create conda environment and configure GDAL settings  
**Duration**: 30 minutes  
**Manual**: Yes

#### Prerequisites
- Conda installed on system
- Git repository cloned locally

#### Execution Steps

1. **Navigate to project directory**:
```bash
cd /path/to/moon_prototype
```

2. **Create conda environment**:
```bash
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai
```

3. **Configure GDAL optimization**:
```bash
source scripts/00_env_setup/gdal_bashrc_snippet.sh
```

4. **Verify installation**:
```bash
python run_prototype.py --list
conda list | grep -E "(gdal|pytorch|ultralytics|richdem)"
```

#### Expected Environment Packages
- Python 3.10
- GDAL 3.8+ with rasterio, rioxarray
- RichDEM for terrain analysis
- PyTorch with CUDA support
- Ultralytics YOLOv8
- GeoPandas for vector processing
- Scikit-image for texture analysis
- JupyterLab for analysis

#### Troubleshooting
- **GDAL errors**: Ensure conda-forge channel priority
- **CUDA issues**: Verify GPU drivers and CUDA toolkit version
- **Memory issues**: Set `GDAL_CACHEMAX=1024` in environment

#### Output Files
- Activated `moonai` conda environment
- GDAL settings configured in shell

---

### STEP 1: Data Acquisition

**Purpose**: Download Chandrayaan-2 satellite imagery for target AOI  
**Duration**: 2-4 hours  
**Manual**: No (automated)

#### Prerequisites
- `pradan-cli` installed and configured with ISRO credentials
- Valid AOI definition in `data/aoi.geojson`
- Sufficient download quota (~4GB)

#### Input Files Required
- `data/aoi.geojson` (Area of Interest polygon)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 1

# Or direct execution
bash scripts/01_data_acquisition.sh
```

#### What This Step Does
1. Creates AOI polygon (20km × 20km at 6.20°S, 226.40°E)
2. Searches for matching CH-2 products:
   - TMC Ortho imagery
   - TMC-DTM elevation data
   - OHRC high-resolution imagery
3. Downloads intersecting data clips (~8% of full strips)
4. Validates data completeness and coverage

#### Expected Output Files
```
data/
├── aoi.geojson                    # Area of Interest definition
├── ch2_tmc_*_oth_*.tif           # TMC Ortho GeoTIFF imagery
├── ch2_tmc_*_dtm_*.tif           # TMC-DTM GeoTIFF elevation data
└── ch2_ohrc_*_img_*.img          # OHRC high-resolution imagery
```

#### Validation Checks
- All three data products downloaded successfully
- Files contain valid raster data with proper geospatial metadata
- AOI coverage confirmed for each dataset

#### Troubleshooting
- **Authentication errors**: Verify pradan-cli credentials
- **Quota exceeded**: Check data portal usage limits
- **Coverage issues**: Verify AOI intersects with available data strips
- **Network timeouts**: Retry download commands

---

### STEP 2: Raster Preprocessing

**Purpose**: Convert to Cloud-Optimized GeoTIFF, reproject, and align raster grids  
**Duration**: 1 hour  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 1 (raw satellite data downloaded)
- GDAL properly configured with optimization settings

#### Input Files Required
- `data/ch2_tmc_*_oth_*.tif` (TMC Ortho)
- `data/ch2_tmc_*_dtm_*.tif` (TMC-DTM)
- `data/ch2_ohrc_*_img_*.img` (OHRC)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 2

# Or direct execution
cd data && python ../scripts/02_raster_preprocessing.py
```

#### What This Step Does
1. **COG Conversion**: Converts source GeoTIFF (.tif) or .img files to Cloud-Optimized GeoTIFF
   - Enables tiled access and compression
   - Preserves 16-bit radiometric depth
   - Adds overviews for efficient visualization

2. **OHRC Reprojection**: 
   - Reprojects from polar stereographic to EPSG:104903 (selenographic)
   - Resamples to 0.25m pixel resolution using cubic interpolation
   - Preserves radiometric accuracy

3. **Grid Alignment**:
   - Snaps all rasters to common 5m grid lattice
   - Ensures pixel-perfect alignment for analysis
   - Maintains TMC ortho as spatial reference

#### Expected Output Files
```
data/
├── tmc_ortho_cog.tif             # TMC Ortho (5m resolution, EPSG:104903)
├── tmc_dtm_cog.tif               # TMC-DTM (5m resolution, EPSG:104903)
├── ohrc_eq.tif                   # OHRC reprojected (0.25m resolution)
└── dtm_snap.tif                  # DTM aligned to TMC grid
```

#### Quality Validation
- All COG files pass `gdalinfo -checksum` validation
- Spatial extents overlap properly
- Pixel alignment confirmed with `gdalinfo -proj4`
- No data loss during reprojection

#### Troubleshooting
- **Memory errors**: Reduce tile sizes, increase GDAL cache
- **Projection issues**: Verify EPSG:104903 definition
- **Alignment failures**: Check input raster extents and CRS

---

### STEP 3: Co-registration (MANUAL STEP)

**Purpose**: Collect Ground Control Points (GCPs) to align OHRC with TMC ortho  
**Duration**: 2 hours  
**Manual**: Yes (requires QGIS interaction)

#### Prerequisites
- Completed Step 2 (preprocessed rasters available)
- QGIS installed with Georeferencer plugin
- Visual identification skills for crater landmarks

#### Input Files Required
- `data/tmc_ortho_cog.tif` (reference image)
- `data/ohrc_eq.tif` (image to be co-registered)

#### Manual Procedure

1. **Open QGIS Georeferencer**:
```bash
# Launch QGIS and navigate to:
# Raster → Georeferencer
```

2. **Load Reference and Target Images**:
   - Reference (target): Load `tmc_ortho_cog.tif`
   - Unreferenced: Load `ohrc_eq.tif`

3. **Collect Ground Control Points**:
   - Identify 10+ common features (crater rims, prominent rocks)
   - Focus on features near AOI center to minimize edge errors
   - Record GCP coordinates in both images
   - Target RMSE < 0.5 pixels (≈12cm accuracy)

4. **Save GCP File**:
   - Export GCPs as `data/gcp_points.txt`
   - Format: `x_pixel y_pixel x_geo y_geo`

5. **Automated Co-registration**:
```bash
# Via pipeline orchestrator
python run_prototype.py --step 3

# Or direct execution
bash scripts/03_coregistration.sh
```

#### What the Automated Part Does
1. Injects GCPs into OHRC metadata using `gdal_translate`
2. Performs polynomial warping with 1st order transformation
3. Validates co-registration accuracy
4. Generates quality assessment report

#### Expected Output Files
```
data/
├── gcp_points.txt                # Ground Control Points
├── ohrc_gcp.tif                 # OHRC with GCP metadata
├── ohrc_coreg.tif               # Co-registered OHRC
└── coregistration_report.txt    # Quality assessment
```

#### Quality Validation
- RMSE < 0.5 pixels for acceptable accuracy
- Visual inspection of landmark alignment
- No systematic geometric distortions
- Edge effects minimized within AOI

#### Troubleshooting
- **High RMSE**: Collect additional GCPs, avoid edge features
- **Distortion artifacts**: Use lower order polynomial transformation
- **Missing landmarks**: Use subtle terrain features like small craters

---

### STEP 4: Hapke Photometric Normalization

**Purpose**: Apply photometric corrections for illumination effects  
**Duration**: 15 minutes  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 3 (co-registered imagery)
- Solar illumination parameters from TMC metadata

#### Input Files Required
- `data/tmc_ortho_cog.tif` (for solar elevation extraction)
- `data/ohrc_coreg.tif` (co-registered OHRC)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 4

# Or direct execution
python scripts/04_hapke_normalisation.py
```

#### What This Step Does
1. **Extract Solar Parameters**:
   - Reads `SUN_ELEVATION = 41.3°` from TMC header
   - Calculates incidence angle: i = 90° - 41.3° = 48.7°

2. **Cosine Correction** (primary method):
   ```
   I_cos = I_raw / cos(i)
   ```

3. **Hapke Model Correction** (optional validation):
   ```
   I_hapke = I_raw × (μ₀ + H(μ₀,w)) / (μ + H(μ,w))
   ```
   Where:
   - w = 0.11 (mare regolith albedo)
   - H(x,w) = (1+2x)/(1+2x√(1-w))

4. **Performance Benchmarking**:
   - Tests corrections on 512×512 tile for runtime analysis
   - Compares cosine vs Hapke model accuracy

#### Expected Output Files
```
data/
├── tmc_tile_cosine.tif          # Cosine-corrected TMC tile
├── tmc_tile_hapke.tif           # Hapke-corrected TMC tile
└── photometric_comparison.png    # Visual quality comparison
```

#### Algorithm Details
- **Cosine correction**: Fast, suitable for bulk processing
- **Hapke model**: Physically accurate, computationally intensive
- **Validation**: Side-by-side comparison on test tile

#### Quality Validation
- Reduced illumination gradients across scene
- Preserved radiometric relationships
- No over-correction artifacts

#### Troubleshooting
- **Metadata errors**: Manually specify solar elevation
- **Over-correction**: Check for extreme incidence angles
- **Performance issues**: Use cosine correction for bulk processing

---

### STEP 5: Terrain Derivatives

**Purpose**: Compute slope and curvature from DTM for landslide detection  
**Duration**: 30 minutes  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 2 (DTM preprocessing)
- RichDEM library properly installed

#### Input Files Required
- `data/dtm_snap.tif` (aligned DTM)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 5

# Or direct execution
python scripts/05_terrain_derivatives.py
```

#### What This Step Does
1. **Gaussian Pre-filtering**:
   - Applies 3×3 Gaussian filter to DTM
   - Reduces stair-step artifacts from discrete elevation data
   - Smooths noise while preserving major terrain features

2. **Slope Calculation**:
   - Uses RichDEM `slope_degrees` algorithm
   - Computes: slope = arctan(|∇z|)
   - Output in degrees for interpretability

3. **Curvature Analysis**:
   - Calculates profile curvature using RichDEM
   - Identifies concave areas (negative curvature)
   - Critical for landslide scarp detection

4. **Quality Control**:
   - Validates output ranges and statistics
   - Generates terrain analysis summary

#### Expected Output Files
```
data/
├── slope_degrees.tif            # Slope in degrees (0-90°)
├── curvature.tif               # Profile curvature
└── terrain_summary.txt         # Statistical summary
```

#### Algorithm Parameters
- **Gaussian filter**: 3×3 kernel, σ=1.0
- **Slope range**: 0-90 degrees
- **Curvature units**: 1/meters (elevation units)
- **NoData handling**: Preserves original DTM mask

#### Quality Validation
- Slope values within expected range (0-90°)
- Curvature statistics consistent with lunar terrain
- No edge artifacts or processing anomalies

#### Troubleshooting
- **Memory errors**: Process in tiles for large areas
- **Edge effects**: Ensure proper NoData handling
- **Unrealistic values**: Check DTM units and vertical datum

---

### STEP 6: Rule-Based Baseline Detection

**Purpose**: Detect landslide and boulder candidates using rule-based methods  
**Duration**: 1 hour  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 4 (photometric correction)
- Completed Step 5 (terrain derivatives)

#### Input Files Required
- `data/tmc_tile_cosine.tif` (photometrically corrected imagery)
- `data/slope_degrees.tif` (terrain slope)
- `data/curvature.tif` (terrain curvature)
- `data/ohrc_coreg.tif` (for boulder detection)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 6

# Or direct execution
python scripts/06_rule_based_baseline.py
```

#### What This Step Does

1. **Texture Analysis (GLCM)**:
   - Computes Grey-Level Co-occurrence Matrix contrast
   - Uses 32×32 pixel windows
   - Identifies areas with high textural variation
   - Threshold at 90th percentile within AOI

2. **Landslide Detection Rules**:
   ```
   landslide_mask = (slope > 25°) AND (curvature < -0.15) AND (contrast > P90)
   ```
   - **Slope criterion**: Steep terrain (>25°)
   - **Curvature criterion**: Concave areas (<-0.15)
   - **Texture criterion**: High contrast regions

3. **Boulder Seed Detection**:
   - Resamples OHRC to 0.5m for efficiency
   - Applies multi-scale Laplacian-of-Gaussian (σ = 1,2,3 pixels)
   - Identifies local maxima > 3σ threshold
   - Filters by size constraints (diameter > 1m)

4. **Vectorization**:
   - Converts raster masks to polygon features
   - Applies morphological operations (opening/closing)
   - Removes small artifacts and noise

#### Expected Output Files
```
data/
├── texture_contrast.tif         # GLCM contrast values
├── landslide_mask.tif          # Binary landslide detection mask
├── landslide_polygons.shp      # Vectorized landslide features
├── boulder_seeds.shp           # Boulder detection points
└── rule_based_summary.txt      # Detection statistics
```

#### Algorithm Parameters
- **GLCM window**: 32×32 pixels
- **Slope threshold**: 25 degrees
- **Curvature threshold**: -0.15 (1/m)
- **LoG scales**: σ = 1, 2, 3 pixels
- **Boulder size**: minimum diameter 1m

#### Quality Validation
- Landslide polygons overlap with visually identified features
- Boulder seeds concentrate in rough terrain areas
- False positive rate within acceptable limits
- Detection statistics consistent with lunar geology

#### Troubleshooting
- **High false positives**: Adjust threshold parameters
- **Missing detections**: Lower detection thresholds
- **Processing time**: Reduce window sizes or image resolution

---

### STEP 7: Manual Annotation Sprint (MANUAL STEP)

**Purpose**: Create training dataset through manual annotation of landslides and boulders  
**Duration**: 2-3 days  
**Manual**: Yes (intensive manual work)

#### Prerequisites
- Completed Step 6 (rule-based detections for guidance)
- QGIS for landslide annotation
- LabelMe for boulder annotation
- Domain expertise in lunar geology

#### Input Files Required
- `data/landslide_polygons.shp` (rule-based candidates)
- `data/boulder_seeds.shp` (boulder candidates)
- `data/tmc_ortho_cog.tif` (reference imagery)
- `data/ohrc_coreg.tif` (high-resolution imagery)

#### Manual Annotation Workflow

#### Landslide Annotation (QGIS)
1. **Setup QGIS Project**:
   - Load TMC ortho as base layer
   - Load slope and curvature as overlays
   - Load rule-based landslide polygons as guidance

2. **Create Annotation Layer**:
   - New shapefile: `landslide_training.shp`
   - Attributes: `id`, `confidence` (1-5 scale), `notes`

3. **Annotation Guidelines**:
   - Target: 30 landslide polygons minimum
   - Focus on clear, unambiguous examples
   - Include confidence scoring (1=uncertain, 5=definite)
   - Diverse size range and terrain types

#### Boulder Annotation (LabelMe)
1. **Setup LabelMe**:
   - Export OHRC tiles (1024×1024 pixels)
   - Launch LabelMe annotation tool

2. **Annotation Guidelines**:
   - Target: 300 boulder annotations minimum
   - Bounding boxes around clear boulder features
   - Class label: "boulder"
   - Size range: 1-10m diameter

#### Dataset Split Strategy
```python
# Spatial split to avoid data leakage
train_split = 70%  # Training data
val_split = 15%    # Validation data  
test_split = 15%   # Test data (hold-out)
```

#### Expected Output Files
```
data/annotations/
├── landslide_training.shp       # Manual landslide polygons
├── boulder_training/            # LabelMe boulder annotations
│   ├── images/                  # OHRC image tiles
│   ├── annotations/             # JSON annotation files
│   └── dataset_split.json       # Train/val/test split
└── annotation_statistics.txt    # Dataset summary
```

#### Quality Control
- Inter-annotator agreement validation
- Systematic coverage across AOI
- Balanced representation of feature sizes
- Documentation of annotation criteria

#### Troubleshooting
- **Annotation fatigue**: Take regular breaks, rotate annotators
- **Ambiguous features**: Use confidence scoring, add notes
- **Class imbalance**: Ensure diverse feature representation

---

### STEP 8: Machine Learning Model Training

**Purpose**: Train U-Net landslide segmentation and YOLOv8 boulder detection models  
**Duration**: 4-6 hours  
**Manual**: No (automated training)

#### Prerequisites
- Completed Step 7 (manual annotations)
- GPU with CUDA support (recommended)
- Sufficient disk space for model checkpoints

#### Input Files Required
- `data/annotations/landslide_training.shp` (landslide training data)
- `data/annotations/boulder_training/` (boulder training data)
- `data/tmc_tile_cosine.tif`, `data/slope_degrees.tif`, `data/curvature.tif` (input channels)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 8

# Or direct execution
python scripts/08_light_ml_models.py
```

#### What This Step Does

#### U-Net Landslide Segmentation
1. **Model Architecture**:
   - Encoder: ResNet18 with ImageNet weights
   - Input channels: 3 [TMC_cosine, slope, curvature]
   - Output: Single channel probability map

2. **Training Configuration**:
   - Loss: 0.5 × BCE + 0.5 × Dice Loss
   - Optimizer: Adam with lr=1e-3
   - Scheduler: Cosine annealing
   - Epochs: 40 with early stopping
   - Batch size: 8

3. **Data Augmentation**:
   - Random rotation, flip, brightness
   - Elastic deformation for geological realism

#### YOLOv8 Boulder Detection
1. **Model Architecture**:
   - YOLOv8n-seg (nano segmentation model)
   - Input size: 1024×1024 pixels
   - Pre-trained on COCO dataset

2. **Fine-tuning Configuration**:
   - Epochs: 5 (transfer learning)
   - Batch size: 8
   - Learning rate: 1e-4
   - Data augmentation: HSV, rotation, scaling

#### Expected Output Files
```
models/
├── unet_landslide_best.pth      # Best U-Net checkpoint
├── yolov8_boulder_best.pt       # Best YOLOv8 checkpoint
├── training_curves.png          # Loss/metric curves
├── unet_training_log.txt        # U-Net training history
└── yolo_training_log.txt        # YOLO training history
```

#### Training Validation
- Validation IoU > 0.50 for landslides
- Validation AP50 > 0.65 for boulders
- No overfitting (stable validation metrics)
- Training convergence within time limits

#### Performance Targets
- **Landslide IoU**: ≥ 0.50
- **Boulder AP50**: ≥ 0.65
- **Training time**: < 6 hours on T4 GPU
- **Model size**: < 100MB for deployment

#### Troubleshooting
- **Out of memory**: Reduce batch size, use gradient accumulation
- **Slow convergence**: Adjust learning rate, check data quality
- **Poor performance**: Increase annotation quality/quantity

---

### STEP 9: Cross-Scale Fusion and Physics Filtering

**Purpose**: Combine U-Net and YOLO predictions with physics-based validation  
**Duration**: 1 hour  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 8 (trained ML models)
- Model checkpoints available

#### Input Files Required
- `models/unet_landslide_best.pth` (trained U-Net model)
- `models/yolov8_boulder_best.pt` (trained YOLO model)
- `data/tmc_tile_cosine.tif`, `data/slope_degrees.tif`, `data/curvature.tif`
- `data/ohrc_coreg.tif` (high-resolution imagery)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 9

# Or direct execution
python scripts/09_fusion_and_filter.py
```

#### What This Step Does

1. **U-Net Inference**:
   - Applies trained U-Net to full 20×20 km TMC tile
   - Generates landslide probability maps
   - Converts to binary masks with optimal threshold

2. **Raster-to-Vector Conversion**:
   - Converts landslide masks to polygon features
   - Applies 30m buffer for spatial tolerance
   - Filters by minimum area threshold

3. **Cross-Scale Validation**:
   For each landslide polygon:
   - Crops corresponding 512×512 OHRC window
   - Runs YOLO inference for boulder detection
   - Validates landslide if either:
     - ≥ 1 boulder detected within polygon, OR
     - Mean slope > 18° within polygon

4. **Physics-Based Filtering**:
   - Boulder diameter: d = 2√(A/π)
   - Shadow length calculation: h = L × tan(θs)
   - Rejects features if h/d > 3 (implausible shadow geometry)
   - Solar elevation θs = 41.3° from metadata

#### Algorithm Details
```python
# Validation logic
for polygon in landslide_candidates:
    ohrc_crop = crop_ohrc(polygon.centroid, size=512)
    yolo_results = yolo_model(ohrc_crop)
    
    if yolo_results.masks.any():
        polygon.validated = True  # Boulder evidence
    elif polygon.mean_slope > 18:
        polygon.validated = True  # Steep terrain
    else:
        polygon.validated = False # Reject
```

#### Expected Output Files
```
outputs/
├── landslide_probability.tif    # U-Net probability map
├── landslide_binary.tif        # Binary landslide mask
├── landslide_polygons_raw.shp  # Before validation
├── boulder_detections.shp      # YOLO boulder results
└── validated_features.shp      # Final validated features
```

#### Validation Metrics
- Cross-validation accuracy between U-Net and YOLO
- Physics filter rejection rate
- Spatial consistency of detections

#### Quality Control
- Visual inspection of validated features
- Physics filter false positive/negative analysis
- Comparison with rule-based baseline

#### Troubleshooting
- **High rejection rate**: Adjust physics filter thresholds
- **Inconsistent results**: Check model loading and preprocessing
- **Performance issues**: Optimize inference batch sizes

---

### STEP 10: Metrics and Runtime Audit

**Purpose**: Evaluate model performance and measure processing efficiency  
**Duration**: 30 minutes  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 9 (validated features)
- Test dataset from Step 7 annotation split

#### Input Files Required
- `outputs/validated_features.shp` (model predictions)
- `data/annotations/landslide_training.shp` (ground truth, test split)
- `data/annotations/boulder_training/` (ground truth annotations)

#### Execution Command
```bash
# Via pipeline orchestrator with timing
/usr/bin/time -v python run_prototype.py --step 10

# Or direct execution
python scripts/10_metrics_audit.py
```

#### What This Step Does

1. **Landslide Performance Metrics**:
   - Intersection over Union (IoU)
   - Precision and Recall
   - F1-Score
   - Average Precision (AP)

2. **Boulder Detection Metrics**:
   - AP50 (Average Precision at IoU=0.5)
   - Mean diameter error vs manual measurements
   - Detection rate by boulder size class

3. **Runtime Performance**:
   - Wall-clock time for complete pipeline
   - Memory usage profiling
   - Processing time per step
   - Throughput metrics (features/minute)

4. **Comparison Analysis**:
   - Rule-based vs ML model performance
   - Cross-validation with manual annotations
   - Error analysis and failure modes

#### Expected Output Files
```
outputs/
├── prototype_metrics.csv        # Complete performance summary
├── confusion_matrix.png         # Landslide classification results
├── precision_recall_curve.png   # Model performance curves
├── boulder_size_analysis.png    # Boulder detection by size
├── runtime_profile.txt          # Timing and memory usage
└── error_analysis.txt           # Failure mode analysis
```

#### Performance Targets
- **Landslide IoU**: ≥ 0.50
- **Landslide Precision**: ≥ 0.70
- **Boulder AP50**: ≥ 0.65
- **Boulder diameter error**: < 20%
- **Total processing time**: < 20 minutes
- **Memory usage**: < 16GB peak

#### Metrics Calculation
```python
# Key metrics computed
iou = intersection_area / union_area
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
ap50 = average_precision_at_iou_threshold(0.5)
```

#### Troubleshooting
- **Poor metrics**: Review model training and validation
- **Timing issues**: Profile bottlenecks, optimize algorithms
- **Memory problems**: Implement incremental processing

---

### STEP 11: Visualization and Packaging

**Purpose**: Generate final visualizations and package results for delivery  
**Duration**: 1 hour  
**Manual**: No (automated)

#### Prerequisites
- Completed Step 10 (performance evaluation)
- All intermediate results available

#### Input Files Required
- `outputs/validated_features.shp` (final detections)
- `outputs/prototype_metrics.csv` (performance results)
- `data/tmc_ortho_cog.tif` (base imagery)
- `data/ohrc_coreg.tif` (high-resolution context)

#### Execution Command
```bash
# Via pipeline orchestrator
python run_prototype.py --step 11

# Or direct execution
python scripts/11_visuals_packaging.py
```

#### What This Step Does

1. **Static Map Generation**:
   - Creates overview map of entire AOI
   - Overlays detected landslides and boulders
   - Includes scale bar, north arrow, coordinates
   - Multiple zoom levels for detail inspection

2. **Results Visualization**:
   - Before/after comparison maps
   - Detection confidence heat maps
   - Boulder size distribution plots
   - Performance metric dashboard

3. **Report Generation**:
   - Updates Jupyter notebook with results
   - Embeds figures and performance tables
   - Generates executive summary

4. **Final Packaging**:
   - Exports final GeoPackage with all features
   - Creates compressed deliverable archive
   - Generates metadata documentation

#### Expected Output Files
```
reports/figures/
├── aoi_overview_map.png         # Complete AOI with detections
├── landslide_detail_maps.png    # Zoomed views of key features
├── boulder_distribution.png     # Boulder size and spatial analysis
├── performance_dashboard.png    # Metrics visualization
└── methodology_diagram.png      # Pipeline workflow diagram

outputs/
├── aoi_landslide_boulder.gpkg   # Final GeoPackage
├── prototype_deliverable.zip    # Complete package
└── metadata_report.xml          # ISO 19115 metadata
```

#### Visualization Specifications
- **Map style**: Clean, publication-ready cartography
- **Color scheme**: Colorblind-friendly palette
- **Resolution**: 300 DPI for publication quality
- **Format**: PNG for figures, PDF for reports
- **Coordinate system**: EPSG:104903 with lat/lon labels

#### Final Deliverables
1. **Technical Results**:
   - `aoi_landslide_boulder.gpkg` with validated features
   - `prototype_metrics.csv` with performance data
   - `Prototype_Report.ipynb` with complete analysis

2. **Documentation**:
   - Updated README.md with results summary
   - Workflow documentation with timing data
   - Metadata compliant with spatial data standards

#### Quality Assurance
- All figures properly labeled and captioned
- GeoPackage validates with GDAL
- Deliverable archive < 1GB size limit
- Documentation completeness check

#### Troubleshooting
- **Figure generation errors**: Check matplotlib backend
- **GeoPackage corruption**: Validate with ogrinfo
- **Package size issues**: Optimize raster compression

---

## Pipeline Integration and Dependencies

### Step Dependencies

```
Step 0 (Environment) → Step 1 (Data Acquisition)
                    ↓
Step 2 (Preprocessing) → Step 3 (Co-registration)
                      ↓
Step 4 (Photometric) → Step 5 (Terrain) → Step 6 (Rule-based)
                                        ↓
Step 7 (Annotation) → Step 8 (ML Training) → Step 9 (Fusion)
                                           ↓
                    Step 10 (Metrics) → Step 11 (Visualization)
```

### Critical Path Analysis
- **Longest path**: Steps 7-8 (annotation + training) = 3-9 days
- **Parallel opportunities**: Steps 4-6 can run concurrently after Step 3
- **Manual bottlenecks**: Steps 3 and 7 require human intervention

### Complete Pipeline Execution

#### Single Command Execution
```bash
# Complete automated pipeline (skips manual steps)
python run_prototype.py --all

# With custom AOI
python run_prototype.py --aoi custom_aoi.geojson --all

# Specific step ranges
python run_prototype.py --step 1-6    # Through rule-based detection
python run_prototype.py --step 8-11   # ML training through visualization
```

#### Manual Step Integration
```bash
# Phase 1: Automated preprocessing
python run_prototype.py --step 1-2

# Phase 2: Manual co-registration (requires QGIS)
python run_prototype.py --step 3

# Phase 3: Automated analysis
python run_prototype.py --step 4-6

# Phase 4: Manual annotation (requires 2-3 days)
# Follow instructions in scripts/07_annotation_sprint.md

# Phase 5: ML training and final results
python run_prototype.py --step 8-11
```

#### Progress Monitoring
```bash
# Check pipeline status
python run_prototype.py --status

# List completed steps
ls -la data/ outputs/ models/

# Monitor resource usage
htop
nvidia-smi  # For GPU monitoring
```

## Performance Optimization

### Memory Management
- **GDAL Cache**: Set `GDAL_CACHEMAX=1024` (1GB)
- **Python**: Use generators for large raster processing
- **Batch sizes**: Adjust ML batch sizes based on available GPU memory

### Processing Speed
- **Parallel processing**: Use all available CPU cores
- **GPU acceleration**: Enable CUDA for ML training and inference
- **I/O optimization**: Use COG format for efficient tile access

### Storage Optimization
- **Compression**: LZW compression for intermediate files
- **Cleanup**: Remove intermediate files after pipeline completion
- **Archiving**: Compress final deliverables

## Troubleshooting Guide

### Common Issues

#### Environment Problems
```bash
# GDAL not found
conda install -c conda-forge gdal=3.8

# CUDA issues
conda install cudatoolkit=11.8
export CUDA_VISIBLE_DEVICES=0

# Memory errors
export GDAL_CACHEMAX=512
ulimit -m 16777216  # 16GB limit
```

#### Data Issues
```bash
# Corrupted downloads
pradan-cli verify --file downloaded_file.img

# Projection errors
gdalinfo -proj4 input_file.tif
gdalsrsinfo EPSG:104903

# Missing files
python run_prototype.py --validate-inputs
```

#### Processing Errors
```bash
# Debug mode with verbose logging
export CPL_LOG=/tmp/gdal_debug.log
export GDAL_CONFIG_FILE=/tmp/gdal_debug.conf
python run_prototype.py --step X --verbose

# Memory profiling
mprof run python run_prototype.py --step X
mprof plot

# Performance timing
/usr/bin/time -v python run_prototype.py --step X
```

### Recovery Procedures

#### Pipeline Interruption
- Pipeline supports Ctrl+C graceful termination
- Progress saved in step-specific output files
- Resume from last completed step

#### Partial Failures
```bash
# Rerun specific failed step
python run_prototype.py --step X --force

# Validate outputs before proceeding
python run_prototype.py --validate-step X

# Clean and restart from step
python run_prototype.py --clean-step X --step X
```

## Success Criteria and Validation

### Technical Performance
- [ ] Landslide detection IoU ≥ 0.50
- [ ] Boulder detection AP50 ≥ 0.65
- [ ] Total processing time < 20 minutes
- [ ] Memory usage < 16GB peak
- [ ] No data corruption or loss

### Deliverable Completeness
- [ ] All 11 pipeline steps completed successfully
- [ ] Final GeoPackage contains validated features
- [ ] Performance metrics documented
- [ ] Visualizations generated
- [ ] Documentation updated

### Quality Assurance
- [ ] Manual validation of sample detections
- [ ] Cross-validation with rule-based methods
- [ ] Error analysis completed
- [ ] Reproducibility confirmed

## Project Timeline Summary

| Phase | Duration | Steps | Key Activities |
|-------|----------|-------|----------------|
| Setup | Day 0 | 0 | Environment configuration |
| Data Acquisition | Days 1-2 | 1-2 | Download and preprocess data |
| Registration | Day 3 | 3 | Manual GCP collection |
| Analysis | Days 4-6 | 4-6 | Photometric correction, terrain analysis, rule-based detection |
| Annotation | Days 7-10 | 7 | Manual feature labeling |
| ML Training | Days 11-16 | 8 | Model training and validation |
| Integration | Days 17-19 | 9-10 | Fusion, filtering, evaluation |
| Finalization | Days 20-21 | 11 | Visualization and packaging |

**Total Duration**: 21 days  
**Manual Effort**: ~5 days (Steps 3, 7)  
**Automated Processing**: ~16 days  
**Target Budget**: $86 cloud compute cost

---

*This comprehensive execution guide provides step-by-step instructions for the complete lunar landslide detection prototype pipeline. Follow the sequential steps while paying special attention to manual intervention points and validation checkpoints.*