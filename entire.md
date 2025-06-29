DETAILED SIX-WEEK PROTOTYPE PLAN  
AOI: 20 km × 20 km window centred at 6.20 °S, 226.40 °E  
(overlap confirmed for CH-2 TMC Ortho, TMC-DTM and OHRC strip 20240215T0311467211)

Timezone assumption: GMT+5 h 30 m

────────────────────────────────────────
0. ENVIRONMENT & COST GUARDRAILS  (Day 0)
────────────────────────────────────────
- Cloud VM: GCP `n1-standard-8` + NVIDIA T4  
  - \$0.35 / h on-demand, capped at 240 h → \$84
- Storage: 50 GB standard disk (~\$1.5)  
- Conda env creation

```bash
conda create -n moonai python=3.10 -y
conda activate moonai
conda install -c conda-forge \
    gdal=3.8 rasterio rioxarray richdem scikit-image \
    pytorch pytorch-lightning cudatoolkit=11.8 albumentations \
    ultralytics geopandas scikit-learn jupyterlab -y
```

Best practice  
- Add `GDAL_CACHEMAX=1024` in `.bashrc` (speeds resampling).  
- Enable conda-forge channel first to avoid binary clashes.

────────────────────────────────────────
1. DATA ACQUISITION  (Days 1–2)
────────────────────────────────────────
1-A  Build an AOI polygon

```bash
echo '{"type":"Polygon",
       "coordinates":[[[226.3,-6.1],[226.5,-6.1],
                       [226.5,-6.3],[226.3,-6.3],
                       [226.3,-6.1]]]
      }' > aoi.geojson
```

1-B  Discover matching product IDs

```bash
pradan-cli search --aoi aoi.geojson --instrument TMC2   # ortho+DTM
pradan-cli search --aoi aoi.geojson --instrument OHRC
```

1-C  Download only intersecting rows (≈ 8 % of full strips)

```bash
pradan-cli fetch ch2_tmc_*oth* --clip aoi.geojson
pradan-cli fetch ch2_tmc_*dtm* --clip aoi.geojson
pradan-cli fetch ch2_ohrc_*img* --clip aoi.geojson
```

Keep in mind  
- OHRC native polar-stereo pixels shrink near the equator; reproject later.  
- Use `--clip` so you are billed for < 4 GB instead of > 2 GB.

────────────────────────────────────────
2. RASTER PRE-PROCESSING  (Days 3–4)
────────────────────────────────────────
2-A  Convert to Cloud-Optimised GeoTIFF

```bash
for f in *.img *.tif; do
  gdal_translate "$f" "${f%.*}_cog.tif" \
      -co TILED=YES -co COMPRESS=LZW -co COPY_SRC_OVERVIEWS=YES
done
```

2-B  Reproject OHRC to selenographic equirectangular  
EPSG 104903, cubic resampling, target pixel ≈ 0.25 m

```bash
gdalwarp -t_srs EPSG:104903 -tr 0.25 0.25 \
         -r cubic -of COG ohrc_*_cog.tif ohrc_eq.tif
```

2-C  Snap grids (align all rasters to a common 5 m lattice)

```bash
gdal_translate -a_srs EPSG:104903 \
               -a_ullr $(gdalinfo tmc_ortho_cog.tif | grep "Upper Left") \
               -tr 5 5 tmc_dtm_cog.tif dtm_snap.tif
```

────────────────────────────────────────
3. ONE-AFTERNOON CO-REGISTRATION (GCPs)  (Day 5)
────────────────────────────────────────
3-A  Open QGIS → Raster → Georeferencer  
- Load `tmc_ortho_cog.tif` as target.  
- Load `ohrc_eq.tif` as “unreferenced.”  
- Drop 10 GCPs on common crater rims, save as `gcp_points.txt`.

3-B  Inject GCPs and warp

```bash
gdal_translate -of GTiff -gcpfile gcp_points.txt \
    ohrc_eq.tif ohrc_gcp.tif
gdalwarp -r cubic -t_srs EPSG:104903 -order 1 \
         ohrc_gcp.tif ohrc_coreg.tif
```

Target RMSE < 0.5 px (≈ 12 cm).  
Best practice: pick GCPs near AOI centre to minimise edge extrapolation error.

────────────────────────────────────────
4. MICRO-HAPKE NORMALISATION  (Day 6)
────────────────────────────────────────
We prototype on **one** 512 × 512 TMC tile.

4-A  Incidence angle \(i\)  
- Extract solar elevation from TMC header (`SUN_ELEVATION = 41.3`).  
- \(i = 90^{\circ} - 41.3^{\circ} = 48.7^{\circ}\).

4-B  Cosine-only normalisation

\(I_{\text{cos}} = I_{\text{raw}} / \cos i\).

4-C  Tiny-Hapke layer (optional)

\[
I_{\text{hapke}} =
I_{\text{raw}}\,
\frac{\mu_0 + H(\mu_0, w)}{\mu + H(\mu, w)},\quad
H(x,w)=\frac{1+2x}{1+2x\sqrt{1-w}}
\]

Assume  
- \(w = 0.11\) (albedo for mature mare regolith)  
- \(\mu_0 = \cos i\), \(\mu = \cos e\) with emission \(e≈0\).

Python snippet

```python
import numpy as np, rasterio

with rasterio.open("tmc_tile.tif") as src:
    im = src.read(1).astype('float32')
i = np.deg2rad(48.7)
mu0 = np.cos(i); mu = 1.0
w = 0.11
def H(x, w): return (1 + 2 * x) / (1 + 2 * x * np.sqrt(1 - w))
I_hapke = im * (mu0 + H(mu0, w)) / (mu + H(mu, w))
```

Keep this tile for runtime benchmarking; use cosine correction for the bulk runs.

────────────────────────────────────────
5. TERRAIN DERIVATIVES (Day 7)
────────────────────────────────────────
RichDEM slope & curvature

```python
import richdem as rd, rioxarray as rxr
dem = rxr.open_rasterio("dtm_snap.tif").squeeze().astype('float32')
slope = rd.TerrainAttribute(dem.values, attrib="slope_degrees")
curv = rd.TerrainAttribute(dem.values, attrib="curvature")
rxr.open_rasterio("tmc_ortho_cog.tif").rio.write_crs(104903)
```

Best practice  
- Apply a 3×3 Gaussian filter to the DEM before derivatives to reduce stair-step artefacts.

────────────────────────────────────────
6. RULE-BASED BASELINE (Days 8–9)
────────────────────────────────────────
6-A  Texture contrast (GLCM)

```python
from skimage.feature import greycomatrix, greycoprops
def glcm_contrast(img, win=32):
    pad = win // 2
    out = np.zeros_like(img, dtype='float32')
    for i in range(pad, img.shape[0]-pad):
        for j in range(pad, img.shape[1]-pad):
            w = img[i-pad:i+pad, j-pad:j+pad]
            g = greycomatrix(w, [1], [0], 256, symmetric=True)
            out[i,j] = greycoprops(g, 'contrast')[0,0]
    return out
```

Threshold at 90th percentile within AOI.

6-B  Landslide mask  

\[
\text{mask}_L = (slope>25^{\circ}) \land (curv < -0.15) \land
(\text{contrast} > P_{90})
\]

Convert to polygons with `rasterio.features.shapes`.

6-C  Boulder seeds  
- Resample OHRC to 0.5 m (faster).  
- Convolve 3-scale Laplacian-of-Gaussian (`sigma = 1,2,3` px); pick peaks > 3 σ.

────────────────────────────────────────
7. ANNOTATION SPRINT  (Days 10–13)
────────────────────────────────────────
- 30 landslide polygons: QGIS → “Edit” → “Add Feature.”  
- 300 boulders: Labelme → bounding-box or polygon class `boulder`.

Dataset split  
- Train 70 %, val 15 %, test 15 % (random but stratified).

────────────────────────────────────────
8. LIGHT ML MODELS (Days 14–22)
────────────────────────────────────────
8-A  Landslide U-Net

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet18", encoder_weights="imagenet",
    in_channels=3, classes=1)
```

Input channels = `[TMC_cos, slope, curvature]`.  
Loss \(L = 0.5\,L_{\text{BCE}} + 0.5\,L_{\text{Dice}}\).

Training: 40 epochs, batch 8, `lr=1e-3` cosine schedule.  
Early-stop on val IoU for ≥ 5 epochs plateau.

8-B  YOLOv8 tiny fine-tune

```bash
yolo task=segment model=yolov8n-seg.pt data=boulder.yaml \
     imgsz=1024 epochs=5 batch=8 lr0=1e-4 \
     box=4,8,16,32 hsv_h=0.02 hsv_s=0.2 hsv_v=0.2
```

Expect AP50 ≈ 0.65 after 15 min on T4.

────────────────────────────────────────
9. CROSS-SCALE FUSION & PHYSICS FILTER (Days 23–25)
────────────────────────────────────────
Algorithm  
1. Run U-Net on full 20 × 20 km TMC tile.  
2. Raster-to-vector buffer 30 m.  
3. For each polygon, window-crop OHRC 512 × 512; infer YOLO masks.  
4. Accept polygon if either  
   - ≥ 1 boulder mask inside, or  
   - mean slope \(>\,18^{\circ}\).  
5. Boulder diameter \(d = 2\sqrt{A/\pi}\).  
6. Shadow sanity: with solar elevation \(θ_s = 41.3^{\circ}\),

\[
h = L \tan θ_s,\quad \text{discard if } h/d > 3.
\]

Python glue (simplified)

```python
import geopandas as gpd, shapely.geometry as sgeom
gdf = gpd.read_file("landslide_polygons.shp")
for idx,row in gdf.iterrows():
    crop = crop_ohrc(row.geometry.centroid, size=512)
    results = yolo_model(crop)
    if (results.masks).any():
        gdf.at[idx, 'validated']=True
    elif row['mean_slope']>18:
        gdf.at[idx, 'validated']=True
    else:
        gdf.at[idx, 'validated']=False
gdf = gdf[gdf.validated]
```

────────────────────────────────────────
10. METRICS & RUNTIME AUDIT  (Day 26)
────────────────────────────────────────
- IoU, Precision, Recall on the hold-out 15 % test set.  
- Boulder AP50, mean diameter error vs. manual caliper.  
- Pipeline wall-clock timer:

```bash
/usr/bin/time -v python run_prototype.py --aoi aoi.geojson
```

Target < 20 min including inference.

────────────────────────────────────────
11. VISUALS & PACKAGING  (Days 27–30)
────────────────────────────────────────
- Static map plates: `matplotlib` + `contextily`.  
- Jupyter notebook `Prototype_Report.ipynb` sections  
  - Data → Pre-proc → ML → Fusion → Results  
  - Embed confusion matrices.  
- Zip: code, three sample COGs (< 1 GB), `README.md`.

────────────────────────────────────────
MATHEMATICAL/PHYSICAL INGREDIENTS RECAP
────────────────────────────────────────
- Photometry: cosine and mini-Hapke correction  
- Terrain: slope \(s = \tan^{-1}\|\nabla z\|\); curvature \(κ\) from second-order derivatives  
- GLCM texture contrast  
- Laplacian-of-Gaussian for blob radii \(σ\) ↔ boulder diameter  
- Shadow geometry \(h = L \tan θ_s\)  
- Losses: BCE, Dice (segmentation); CIoU + BCE (YOLOv8)

────────────────────────────────────────
BEST-PRACTICE CHECKLIST
────────────────────────────────────────
- Always keep the native 16-bit depth for radiance; do not scale to 8-bit.  
- Use `rio clip` rather than `gdal_translate` when shaving AOIs (preserves tags).  
- Log every GDAL call with `--config CPL_LOG` for reproducibility.  
- Document every manual GCP (id, x, y, tie-point residual) in `gcp.csv`.  
- Seed every random generator (`torch`, `numpy`, `random`) with `42`.

────────────────────────────────────────
DELIVERABLES
────────────────────────────────────────
- `run_prototype.py` one-click driver  
- `Prototype_Report.ipynb` with inline figures  
- `prototype_metrics.csv`  
- `aoi_landslide_boulder.gpkg` (validated features)  
- Time-stamped bash history (`bash-script-log.txt`)

Finish line: 30 calendar days, \$86 cloud spend, prototype IoU ≥ 0.50, AP50 ≥ 0.65.