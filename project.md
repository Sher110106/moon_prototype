Below is a two‑part deliverable:

---

## 1. AI‑Automatable Components

From your six‑week prototype plan, these are the pieces an AI can generate for you:

| Component                     | Description                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Folder structure scaffold** | Create the top‑level folders and subfolders (e.g. `data/`, `scripts/`, `notebooks/`, `reports/`).              |
| **Environment files**         | `environment.yml` (Conda), `.bashrc` snippet with GDAL settings.                                               |
| **CLI wrapper**               | `run_prototype.py` – orchestrates each step via command‑line flags.                                            |
| **Step scripts**              | Individual Python and shell scripts for each numbered step (e.g. `01_acquire_data.sh`, `02_preproc.py`, etc.). |
| **Markdown driver**           | `PROTOTYPE_WORKFLOW.md` containing clear instructions to run each step in order.                               |
| **Jupyter notebook stub**     | `Prototype_Report.ipynb` with empty sections/data‑placeholders.                                                |
| **Metrics CSV template**      | `prototype_metrics.csv` with header row for IoU, Precision, Recall, AP50, timing, etc.                         |
| **GPKG placeholder**          | Empty `aoi_landslide_boulder.gpkg` (or script that bootstraps it).                                             |
| **README**                    | Top‑level `README.md` summarizing the project, usage, folder layout, and prerequisites.                        |
| **Logging setup**             | A small script or config to enable `CPL_LOG` for every GDAL call.                                              |

---

## 2. Detailed AI Prompt / Plan

You can feed the following to an AI agent (e.g. a code‑generation model) to have it scaffold the entire prototype repository:

```
You are going to scaffold a six‑week lunar‑landslide prototype project.  
Use the following high‑level folder structure:  

moon_prototype/
├── data/  
│   └── aoi.geojson  
├── scripts/  
│   ├── 00_env_setup/  
│   │   ├── environment.yml  
│   │   └── gdal_bashrc_snippet.sh  
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
├── notebooks/  
│   └── Prototype_Report.ipynb  
├── reports/  
│   └── PROTOTYPE_WORKFLOW.md  
├── outputs/  
│   ├── prototype_metrics.csv  
│   └── aoi_landslide_boulder.gpkg  
├── run_prototype.py  
└── README.md  

For each file, do the following:  

1. **environment.yml**  
   - Include Python 3.10 and all conda‑forge packages: gdal=3.8, rasterio, rioxarray, richdem, scikit-image, pytorch, pytorch-lightning, cudatoolkit=11.8, albumentations, ultralytics, geopandas, scikit-learn, jupyterlab.  

2. **gdal_bashrc_snippet.sh**  
   - Export `GDAL_CACHEMAX=1024` and add `conda-forge` priority.  

3. **01_data_acquisition.sh**  
   - Generate `aoi.geojson` as per spec.  
   - Use `pradan-cli` commands to search and fetch the three products, clipping to AOI.  

4. **02_raster_preprocessing.py**  
   - Loop over raw images to create COGs.  
   - Reproject OHRC to EPSG:104903 at 0.25 m.  
   - Snap DTM to the TMC ortho grid at 5 m.  

5. **03_coregistration.sh**  
   - Skeleton commands for QGIS georeferencer, plus `gdal_translate` and `gdalwarp` with GCP file.  

6. **04_hapke_normalisation.py**  
   - Python script implementing both cosine and Hapke corrections on a single tile.  

7. **05_terrain_derivatives.py**  
   - Python script using richdem + rioxarray to compute slope and curvature, applying a 3×3 Gaussian filter.  

8. **06_rule_based_baseline.py**  
   - Python script computing GLCM contrast, thresholding, masking, and boulder seed detection.  

9. **07_annotation_sprint.md**  
   - Markdown checklist/instructions for manual QGIS and LabelMe annotation and dataset split.  

10. **08_light_ml_models.py**  
    - Python script that defines the U‑Net and YOLOv8 tiny training routines with specified hyperparameters.  

11. **09_fusion_and_filter.py**  
    - Python glue code to read polygons, crop OHRC, run YOLO inference, apply physics filter, and flag validated features.  

12. **10_metrics_audit.py**  
    - Script to compute IoU, Precision, Recall, AP50 and log runtime via `/usr/bin/time -v`. Output to `outputs/prototype_metrics.csv`.  

13. **11_visuals_packaging.py**  
    - Python script (using matplotlib + contextily) to generate static map plates and export them into `reports/figures/`.  

14. **run_prototype.py**  
    - A CLI entry point (e.g. with `argparse`) that calls each `scripts/` module in order when passed flags `--step 1` … `--step 11` or `--all`.  

15. **Prototype_Report.ipynb**  
    - Notebook stub with headings: Data → Pre‑proc → ML → Fusion → Results, and empty code cells ready for figures and metrics tables.  

16. **PROTOTYPE_WORKFLOW.md**  
    - Top‑level Markdown that lists each step with the corresponding `scripts/` filename and the exact shell command to run it.  

17. **README.md**  
    - Project overview, prerequisites, how to set up the environment, and a single‑line command to bootstrap the entire pipeline.  

18. **Outputs placeholders**  
    - Create empty `prototype_metrics.csv` with header row: `step,iou,precision,recall,ap50,wall_time_sec`.  
    - Create an empty GeoPackage `aoi_landslide_boulder.gpkg` (can be touched by a shell script).  

Make sure each script has a proper shebang (`#!/usr/bin/env bash` or `#!/usr/bin/env python3`), is executable, and includes basic docstrings or comments indicating its role.  

End of prompt.
```
