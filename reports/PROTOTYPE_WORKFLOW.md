# Lunar Landslide Detection Prototype Workflow

**Complete 6-week implementation guide for automated lunar landslide detection**

AOI: 20 km × 20 km centred at 6.20°S, 226.40°E  
Target: IoU ≥ 0.50, AP50 ≥ 0.65, Runtime ≤ 20 minutes  
Budget: $86 cloud compute (GCP n1-standard-8 + T4)

---

## Prerequisites

### Environment Setup
```bash
# Create conda environment
conda create -n moonai python=3.10 -y
conda activate moonai

# Install dependencies
conda install -c conda-forge \
    gdal=3.8 rasterio rioxarray richdem scikit-image \
    pytorch pytorch-lightning cudatoolkit=11.8 albumentations \
    ultralytics geopandas scikit-learn jupyterlab -y

# Configure GDAL
echo 'export GDAL_CACHEMAX=1024' >> ~/.bashrc
source ~/.bashrc
```

### Project Structure Verification
```bash
# Ensure project structure exists
ls -la moon_prototype/
# Expected: data/, scripts/, notebooks/, reports/, outputs/, run_prototype.py, README.md
```

---

## Step-by-Step Execution

### Step 0: Environment Setup (Day 0)
**Script**: `scripts/00_env_setup/`
**Purpose**: Configure development environment and validate setup

```bash
# Source GDAL configuration
source scripts/00_env_setup/gdal_bashrc_snippet.sh

# Verify conda environment
conda env export > environment_check.yml
echo "Environment setup completed successfully"
```

---

### Step 1: Data Acquisition (Days 1-2)
**Script**: `scripts/01_data_acquisition.sh`
**Purpose**: Download and clip Chandrayaan-2 data to AOI

```bash
# Execute data acquisition
bash scripts/01_data_acquisition.sh

# Expected outputs:
# - data/aoi.geojson
# - data/ch2_tmc_*oth*.img (TMC ortho)
# - data/ch2_tmc_*dtm*.img (TMC DTM)
# - data/ch2_ohrc_*img*.img (OHRC)
```

**Validation**:
```bash
# Check downloaded data
ls -lah data/
du -sh data/  # Should be ~4GB with clipping
```

---

### Step 2: Raster Preprocessing (Days 3-4)
**Script**: `scripts/02_raster_preprocessing.py`
**Purpose**: Convert to COGs, reproject, and align grids

```bash
# Execute preprocessing
python scripts/02_raster_preprocessing.py data/ outputs/

# Expected outputs:
# - outputs/tmc_ortho_cog.tif (5m resolution)
# - outputs/tmc_dtm_cog.tif (5m resolution)
# - outputs/ohrc_eq.tif (0.25m, reprojected to EPSG:104903)
# - outputs/dtm_snap.tif (grid-aligned DTM)
```

**Validation**:
```bash
# Verify raster properties
gdalinfo outputs/tmc_ortho_cog.tif | head -20
gdalinfo outputs/ohrc_eq.tif | grep "Pixel Size"
```

---

### Step 3: Co-registration (Day 5)
**Script**: `scripts/03_coregistration.sh`
**Purpose**: Achieve sub-pixel alignment between TMC and OHRC

```bash
# Interactive GCP collection (use QGIS)
# 1. Load TMC ortho as reference
# 2. Load OHRC as unreferenced layer
# 3. Collect 10 GCPs on crater rims
# 4. Save as gcp_points.txt

# Execute co-registration
bash scripts/03_coregistration.sh

# Expected outputs:
# - outputs/ohrc_coreg.tif (co-registered OHRC)
# - outputs/gcp_points.txt (GCP coordinates)
# - outputs/coregistration_report.txt (RMSE statistics)
```

**Target**: RMSE < 0.5 pixels (≈12 cm)

---

### Step 4: Photometric Normalization (Day 6)
**Script**: `scripts/04_hapke_normalisation.py`
**Purpose**: Apply illumination corrections for consistent radiometry

```bash
# Execute normalization
python scripts/04_hapke_normalisation.py outputs/tmc_ortho_cog.tif outputs/

# Expected outputs:
# - outputs/tmc_cosine_corrected.tif
# - outputs/tmc_hapke_corrected.tif (single tile demo)
# - outputs/normalization_stats.json
```

**Parameters**:
- Solar elevation: 41.3° (incidence angle: 48.7°)
- Hapke albedo: w = 0.11 (mare regolith)

---

### Step 5: Terrain Derivatives (Day 7)
**Script**: `scripts/05_terrain_derivatives.py`
**Purpose**: Compute slope and curvature from DTM

```bash
# Execute terrain analysis
python scripts/05_terrain_derivatives.py outputs/dtm_snap.tif outputs/

# Expected outputs:
# - outputs/slope.tif (degrees)
# - outputs/curvature.tif (m⁻¹)
# - outputs/terrain_stats.json
```

**Processing**: 3×3 Gaussian smoothing applied to reduce stair-step artifacts

---

### Step 6: Rule-based Baseline (Days 8-9)
**Script**: `scripts/06_rule_based_baseline.py`
**Purpose**: Generate initial landslide candidates and boulder seeds

```bash
# Execute baseline detection
python scripts/06_rule_based_baseline.py \
    outputs/tmc_cosine_corrected.tif \
    outputs/slope.tif \
    outputs/curvature.tif \
    outputs/ohrc_coreg.tif \
    outputs/

# Expected outputs:
# - outputs/landslide_candidates.gpkg
# - outputs/boulder_seeds.gpkg
# - outputs/glcm_contrast.tif
# - outputs/baseline_stats.json
```

**Thresholds**:
- Slope > 25°
- Curvature < -0.15
- GLCM contrast > 90th percentile

---

### Step 7: Manual Annotation (Days 10-13)
**Instructions**: `scripts/07_annotation_sprint.md`
**Purpose**: Create training/validation datasets

```bash
# Review annotation instructions
cat scripts/07_annotation_sprint.md

# Expected outputs (manual creation):
# - annotations/landslide_polygons.gpkg (30 polygons)
# - annotations/boulder_boxes/ (300 bounding boxes)
# - annotations/train_test_split.json (70/15/15 split)
```

**Tools**: QGIS for landslides, LabelMe for boulders

---

### Step 8: ML Model Training (Days 14-22)
**Script**: `scripts/08_light_ml_models.py`
**Purpose**: Train U-Net and YOLOv8 models

```bash
# Execute training
python scripts/08_light_ml_models.py annotations/ outputs/models/

# Expected outputs:
# - outputs/models/best_landslide_unet.pth
# - outputs/models/yolo_boulder/weights/best.pt
# - outputs/models/training_metrics.json
# - outputs/models/training_logs/
```

**Training Configuration**:
- U-Net: 40 epochs, batch 8, lr=1e-3, cosine schedule
- YOLO: 15 epochs, batch 8, lr=1e-4, early stopping

**Target Performance**:
- Landslide IoU ≥ 0.50
- Boulder AP50 ≥ 0.65

---

### Step 9: Fusion and Filtering (Days 23-25)
**Script**: `scripts/09_fusion_and_filter.py`
**Purpose**: Combine models with physics-based validation

```bash
# Execute fusion pipeline
python scripts/09_fusion_and_filter.py \
    outputs/models/ \
    outputs/tmc_cosine_corrected.tif \
    outputs/slope.tif \
    outputs/curvature.tif \
    outputs/ohrc_coreg.tif \
    outputs/

# Expected outputs:
# - outputs/landslide_prediction.tif (U-Net inference)
# - outputs/aoi_landslide_boulder.gpkg (validated features)
# - outputs/fusion_stats.json
```

**Validation Logic**:
1. U-Net detects landslide candidates
2. YOLO searches for boulders in OHRC crops
3. Accept if: (≥1 boulder detected) OR (mean slope >18°)
4. Apply shadow geometry filter

---

### Step 10: Metrics Audit (Day 26)
**Script**: `scripts/10_metrics_audit.py`
**Purpose**: Comprehensive performance evaluation

```bash
# Execute metrics audit
python scripts/10_metrics_audit.py \
    outputs/models/ \
    annotations/ \
    outputs/aoi_landslide_boulder.gpkg \
    outputs/

# Expected outputs:
# - outputs/prototype_metrics.csv
# - outputs/comprehensive_metrics_report.json
# - outputs/runtime_benchmark.json
```

**Runtime Measurement**:
```bash
# Full pipeline benchmark
/usr/bin/time -v python run_prototype.py --all --aoi data/aoi.geojson
```

**Target**: <20 minutes wall-clock time

---

### Step 11: Visualization & Packaging (Days 27-30)
**Script**: `scripts/11_visuals_packaging.py`
**Purpose**: Create final deliverables and package

```bash
# Generate visualizations and package
python scripts/11_visuals_packaging.py . outputs/

# Expected outputs:
# - reports/figures/*.png (visualization plates)
# - outputs/lunar_landslide_prototype.zip (final package)
# - outputs/package_info.json
```

**Deliverables**:
- Static map plates with detection overlays
- Performance plots and confusion matrices
- Packaged codebase (<1GB)

---

## Quality Control Checkpoints

### After Each Step
```bash
# Verify outputs exist
ls -la outputs/
echo $?  # Should be 0 (success)

# Check log files
tail -20 outputs/logs/step_XX.log
```

### Critical Validations

**Step 2 - Preprocessing**:
```bash
# Verify raster alignment
python -c "
import rasterio
with rasterio.open('outputs/tmc_ortho_cog.tif') as src1, \
     rasterio.open('outputs/dtm_snap.tif') as src2:
    print('TMC shape:', src1.shape)
    print('DTM shape:', src2.shape)
    print('Bounds match:', src1.bounds == src2.bounds)
"
```

**Step 3 - Co-registration**:
```bash
# Check RMSE
grep "RMSE" outputs/coregistration_report.txt
# Should be < 0.5 pixels
```

**Step 8 - Training**:
```bash
# Verify model performance
python -c "
import json
with open('outputs/models/training_metrics.json') as f:
    metrics = json.load(f)
print('Landslide IoU:', metrics.get('unet_iou', 0))
print('Boulder AP50:', metrics.get('yolo_map50', 0))
"
```

**Step 10 - Final Metrics**:
```bash
# Check success criteria
python -c "
import json
with open('outputs/comprehensive_metrics_report.json') as f:
    report = json.load(f)
assessment = report['overall_assessment']
print('Overall Success:', assessment['overall_success'])
print('All targets met:', all(assessment.values()))
"
```

---

## Troubleshooting

### Common Issues

**Memory Errors**:
```bash
# Reduce batch sizes in scripts
export CUDA_VISIBLE_DEVICES=0
export GDAL_CACHEMAX=512
```

**GDAL Errors**:
```bash
# Verify GDAL installation
gdalinfo --version
python -c "import rasterio; print('OK')"
```

**Missing Dependencies**:
```bash
# Reinstall conda environment
conda env remove -n moonai
# Repeat environment setup
```

**Co-registration Issues**:
```bash
# Manual GCP verification in QGIS
# Ensure 10+ well-distributed points
# Check for systematic offsets
```

### Performance Optimization

**GPU Memory**:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Reduce tile sizes if needed
export TILE_SIZE=256  # Default: 512
```

**Disk Space**:
```bash
# Clean intermediate files
rm outputs/*_temp.tif
rm -rf outputs/cache/
```

---

## Final Deliverables Checklist

- [ ] `run_prototype.py` - One-click driver
- [ ] `notebooks/Prototype_Report.ipynb` - Analysis notebook
- [ ] `outputs/prototype_metrics.csv` - Quantitative results
- [ ] `outputs/aoi_landslide_boulder.gpkg` - Validated features
- [ ] `reports/figures/` - Visualization plates
- [ ] `outputs/lunar_landslide_prototype.zip` - Final package
- [ ] Runtime ≤ 20 minutes demonstrated
- [ ] All success criteria documented

---

## Success Criteria Summary

| Criterion | Target | Status |
|-----------|--------|--------|
| Landslide IoU | ≥ 0.50 | ⟨to be filled⟩ |
| Boulder AP50 | ≥ 0.65 | ⟨to be filled⟩ |
| Runtime | ≤ 20 min | ⟨to be filled⟩ |
| Cloud Budget | ≤ $86 | ⟨to be filled⟩ |
| Package Size | < 1 GB | ⟨to be filled⟩ |

**Finish Line**: 30 calendar days, all targets achieved, prototype operational.

---

*Generated by lunar landslide detection prototype v1.0*  
*For questions or issues, refer to README.md or project documentation*