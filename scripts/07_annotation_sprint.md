# Annotation Sprint - Manual Data Labeling

This step requires manual annotation of landslides and boulders for training machine learning models. Plan for **2-3 days** of focused annotation work.

## Overview

**Target Annotations:**
- **30 landslide polygons** (QGIS)
- **300 boulder annotations** (LabelMe)
- **Dataset split**: 70% train, 15% validation, 15% test

**Required Software:**
- QGIS (for landslide polygon annotation)
- LabelMe (for boulder bounding box annotation)

## Part 1: Landslide Polygon Annotation (QGIS)

### Setup
1. Open QGIS
2. Load base layers:
   - `tmc_ortho_cog.tif` (TMC orthophoto)
   - `slope_degrees.tif` (slope analysis)
   - `curvature.tif` (curvature analysis)
   - `landslide_polygons.shp` (rule-based detections as reference)

### Annotation Process
1. **Create new vector layer:**
   - Geometry type: Polygon
   - CRS: EPSG:104903
   - Fields: `id` (integer), `confidence` (integer 1-5), `notes` (text)

2. **Annotation guidelines:**
   - Focus on areas with clear landslide morphology
   - Look for: steep scarps, depositional fans, irregular topography
   - Use rule-based detections as starting points, but refine boundaries
   - Annotate both fresh and degraded landslides

3. **Quality criteria:**
   - Minimum area: 100 m²
   - Clear morphological evidence
   - Confidence scoring:
     - 5: Definite landslide with clear scarp and deposit
     - 4: Probable landslide with good evidence
     - 3: Possible landslide, some uncertainty
     - 2: Questionable, marginal evidence
     - 1: Very uncertain

### Workflow Checklist
- [ ] Load all base layers in QGIS
- [ ] Create landslide annotation layer
- [ ] Review rule-based detections for candidates
- [ ] Annotate 30 high-quality landslide polygons
- [ ] Assign confidence scores (use mostly 4-5)
- [ ] Add descriptive notes for unusual cases
- [ ] Save as `landslide_annotations.shp`
- [ ] Export coordinates to CSV for dataset splitting

## Part 2: Boulder Annotation (LabelMe)

### Setup
1. Install LabelMe: `pip install labelme`
2. Prepare image tiles:
   - Extract 512×512 pixel tiles from OHRC imagery
   - Focus on landslide areas and boulder fields
   - Save as PNG files for LabelMe compatibility

### Annotation Process
1. **Launch LabelMe:**
   ```bash
   labelme
   ```

2. **Annotation guidelines:**
   - Create bounding boxes around individual boulders
   - Class name: `boulder`
   - Minimum size: ~2 pixels diameter
   - Include both isolated and clustered boulders
   - Avoid annotation of small rocks or unclear features

3. **Quality criteria:**
   - Clear boulder shadows (when illuminated)
   - Roughly circular/elliptical shape
   - Size range: 0.5-20 meters diameter
   - Avoid obvious artifacts or noise

### Workflow Checklist
- [ ] Extract 512×512 OHRC tiles covering landslide areas
- [ ] Convert tiles to PNG format
- [ ] Launch LabelMe annotation tool
- [ ] Annotate ~300 boulders across multiple tiles
- [ ] Use consistent `boulder` class label
- [ ] Focus on clear, unambiguous boulder features
- [ ] Save annotations in YOLO format
- [ ] Review and quality-check annotations

## Part 3: Dataset Splitting

### Strategy
- **Spatial splitting**: Ensure train/val/test splits are spatially separated
- **Stratified sampling**: Maintain class balance across splits
- **Random seed**: Use seed=42 for reproducibility

### Implementation
```python
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split

# Load annotations
landslides = gpd.read_file('landslide_annotations.shp')
boulders = gpd.read_file('boulder_annotations.shp')

# Set random seed
np.random.seed(42)

# Split landslides
train_ls, temp_ls = train_test_split(landslides, test_size=0.3, random_state=42)
val_ls, test_ls = train_test_split(temp_ls, test_size=0.5, random_state=42)

# Split boulders
train_bl, temp_bl = train_test_split(boulders, test_size=0.3, random_state=42)
val_bl, test_bl = train_test_split(temp_bl, test_size=0.5, random_state=42)

# Save splits
train_ls.to_file('train_landslides.shp')
val_ls.to_file('val_landslides.shp')
test_ls.to_file('test_landslides.shp')

train_bl.to_file('train_boulders.shp')
val_bl.to_file('val_boulders.shp')
test_bl.to_file('test_boulders.shp')
```

### Checklist
- [ ] Implement spatial dataset splitting
- [ ] Verify split ratios: 70/15/15%
- [ ] Check for spatial separation between splits
- [ ] Save split datasets with clear naming
- [ ] Generate summary statistics for each split
- [ ] Create YOLO format files for boulder training

## Part 4: Quality Assurance

### Landslide QA
- [ ] Review all 30 polygons for geometric validity
- [ ] Check for topology errors (self-intersections)
- [ ] Verify confidence scores are reasonable
- [ ] Ensure good spatial distribution across AOI

### Boulder QA
- [ ] Review random sample of 50 boulder annotations
- [ ] Check bounding box accuracy
- [ ] Verify class labels are consistent
- [ ] Look for missed boulders or false positives

### Dataset QA
- [ ] Verify train/val/test split percentages
- [ ] Check for data leakage between splits
- [ ] Ensure class balance across splits
- [ ] Generate summary statistics

## Deliverables

### Files to Create
```
data/
├── landslide_annotations.shp      # All landslide polygons
├── train_landslides.shp           # Training set
├── val_landslides.shp             # Validation set
├── test_landslides.shp            # Test set
├── boulder_annotations/           # LabelMe annotation files
│   ├── tile_001.json
│   ├── tile_002.json
│   └── ...
├── train_boulders.shp             # Training boulder points
├── val_boulders.shp               # Validation boulder points
├── test_boulders.shp              # Test boulder points
└── annotation_summary.txt         # Summary statistics
```

### Summary Report
Create `annotation_summary.txt` with:
- Total annotations per class
- Dataset split statistics
- Quality metrics and confidence scores
- Time spent on annotation
- Notes on challenging cases

## Tips for Efficient Annotation

1. **Use shortcuts**: Learn QGIS and LabelMe keyboard shortcuts
2. **Batch processing**: Group similar annotations together
3. **Reference data**: Keep geology references handy
4. **Take breaks**: Avoid annotation fatigue
5. **Consistency**: Develop personal annotation standards early
6. **Documentation**: Note unusual cases for future reference

## Common Challenges

- **Degraded landslides**: Use slope/curvature data to identify subtle features
- **Boulder shadows**: Some boulders may lack clear shadows
- **Scale variation**: Maintain consistent annotation scale
- **Ambiguous features**: When in doubt, mark with lower confidence

## Success Criteria

- [ ] 30 high-quality landslide polygons annotated
- [ ] 300 boulder annotations completed
- [ ] Dataset properly split (70/15/15%)
- [ ] All files saved in correct formats
- [ ] Quality assurance completed
- [ ] Summary documentation created

**Estimated Time**: 2-3 days (16-24 hours total)
- Landslides: 1-1.5 days
- Boulders: 1-1.5 days  
- QA and documentation: 2-4 hours

---

**Note**: This annotation work is critical for model training quality. Take time to ensure annotations are accurate and consistent.