# Deployment & Test Guide for Lunar-Landslide Prototype

Below you find two execution blueprints:

1. **Single VM (all-in-one)** – good for development and small AOIs.
2. **GPU Cluster (batch / Slurm style)** – for large-scale runs or grid-search retraining.

Both guides assume the repository layout/function names described in `run.md` and that you have pushed the project to a private Git repo (or zipped tarball).

---
## 1. Single VM Workflow  (≈ 3 h set-up / 30 min end-to-end run)

### 1.1  Provision machine
| Cloud | Machine type | vCPU | RAM | GPU | Boot Disk | Spot? |
|-------|--------------|------|-----|-----|-----------|-------|
| GCP   | `n1-standard-8` | 8 | 30 GB | NVIDIA T4 (1×) | 100 GB SSD | optional |
| AWS   | `g4dn.xlarge`   | 4 | 16 GB | NVIDIA T4 (1×) | 100 GB gp3 | optional |
| Azure | `Standard_NC4as_T4_v3` | 4 | 28 GB | NVIDIA T4 (1×) | 100 GB SSD | optional |

*Enable the GPU drivers (CUDA 11.8) via the vendor's marketplace image or `nvidia-driver-installer` script.*

### 1.2  Clone & bootstrap
```bash
sudo apt-get update && sudo apt-get install git build-essential wget -y
# (Or yum/dnf on RHEL-based images)

git clone https://github.com/<your_org>/moon_prototype.git
cd moon_prototype

# Install Miniconda (quiet)
wget -qO miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"

conda init bash && source ~/.bashrc
conda env create -f scripts/00_env_setup/environment.yml
conda activate moonai

# Add GDAL tuning flags
source scripts/00_env_setup/gdal_bashrc_snippet.sh
```

### 1.3  Data acquisition (Step 1)
You need ISRO PRADAN credentials **inside** the VM:
```bash
pradan-cli login   # one-time; stores token at ~/.config/pradan
python run_prototype.py --step 1
```
Expected output (see *run.md*): `data/ch2_tmc_*_oth_*.tif`, `data/ch2_tmc_*_dtm_*.tif`, `data/ch2_ohrc_*_img_*.img`, `<4 GB` total.

### 1.4  End-to-end execution
For a first smoke test run everything **except** manual steps (3 & 7):
```bash
# Automated chain: 0,2,4,5,6,8-11
python run_prototype.py --step 2-6   # preprocessing → rule-based

# -> do co-registration in QGIS on your laptop or via X11 SSH (Step 3)
#    copy the resulting ohrc_coreg.tif + gcp_points.txt back into data/

# Continue ML & fusion (Step 8-11)
python run_prototype.py --step 8-11
```

Wall-clock targets (small AOI, T4 GPU):
| Phase | Time |
|-------|------|
| 2-6 Pre-proc & rules | 4-5 min |
| 8 Training (U-Net 40 ep) | 10-12 min |
| 8 YOLO fine-tune 5 ep | 3-4 min |
| 9 Fusion + physics | 2 min |
| 10 Metrics | 1 min |
| 11 Visuals & ZIP | 1 min |

Outputs appear in `outputs/` (GeoPackage, metrics CSV, figures).

### 1.5  Quick validation
```bash
ogrinfo outputs/aoi_landslide_boulder.gpkg -so | head -n 20
cat outputs/prototype_metrics.csv
```
Look for IoU ≥ 0.50 and AP50 ≥ 0.65.

---
## 2. GPU-Cluster Workflow  (SLURM example)

> The cluster will host the *ML-heavy* steps (8 & 9). Steps 0-6 & 11 can stay on the login node or a lightweight CPU node.

### 2.1  Shared storage layout
```
/scratch/$USER/moon_prototype
├── data/               # copied from VM or downloaded directly
├── env/                # conda envs live here
├── outputs/            # job artefacts
└── logs/
```
Make sure `/scratch` (or `$WORK`) is visible from all GPU nodes.

### 2.2  Build the conda env once per cluster
```bash
module load cuda/11.8  # if required by site
cd /scratch/$USER/moon_prototype
conda env create -p env/moonai -f scripts/00_env_setup/environment.yml
```
Package caches are shared, subsequent jobs start instantly.

### 2.3  Slurm job scripts
#### 2.3.1  UNet + YOLO training (`train.slurm`)
```bash
#!/bin/bash
#SBATCH -A lunar
#SBATCH -J moon-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/train_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/moon_prototype/env/moonai
cd /scratch/$USER/moon_prototype

python run_prototype.py --step 8   # only training
```

#### 2.3.2  Fusion & filter (`fusion.slurm`)
```bash
#!/bin/bash
#SBATCH -A lunar
#SBATCH -J moon-fuse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/fusion_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/moon_prototype/env/moonai
cd /scratch/$USER/moon_prototype

# assumes training has produced outputs/models/
srun python run_prototype.py --step 9
```

#### 2.3.3  Metrics & visuals (`metrics.slurm`)
```bash
#!/bin/bash
#SBATCH -A lunar
#SBATCH -J moon-metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/metrics_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/moon_prototype/env/moonai
cd /scratch/$USER/moon_prototype

python run_prototype.py --step 10-11
```

### 2.4  Dependency graph
```
step2-6 (CPU)  →  step8 (GPU)  →  step9 (GPU)  →  step10-11 (CPU)
```
Use Slurm `--dependency=afterok:<jobid>` to chain them:
```bash
jid_pre=$(sbatch preproc.slurm | awk '{print $4}')
jid_train=$(sbatch --dependency=afterok:$jid_pre train.slurm | awk '{print $4}')
jid_fuse=$(sbatch --dependency=afterok:$jid_train fusion.slurm | awk '{print $4}')
sbatch --dependency=afterok:$jid_fuse metrics.slurm
```

### 2.5  Large-scale retraining
For N different AOIs:
```bash
for aoi in $(cat aoi_list.txt); do
   cp templates/train.slurm jobs/train_${aoi}.slurm
   sed -i "s|AOI_PLACEHOLDER|$aoi|g" jobs/train_${aoi}.slurm
   sbatch jobs/train_${aoi}.slurm
done
```
Each job writes into `outputs/$AOI/` to avoid collisions.

### 2.6  Monitoring
```bash
squeue -u $USER
watch nvidia-smi       # inside a GPU node via ssh -L
```
Failure log path: `logs/<job>.out` → always inspect tail.

### 2.7  Collecting artefacts
At the end, sync minimal outputs back to a workstation:
```bash
rsync -avzP gpu-cluster:/scratch/$USER/moon_prototype/outputs/ ./outputs_cluster/
```

---
## 3. Smoke-Test Checklist
| Item | Expected |
|------|----------|
| `data/*.tif` sizes | >10 MB each |
| `outputs/slope_degrees.tif` CRS | EPSG:104903 |
| `outputs/models/best_landslide_unet.pth` | ≈ 45 MB |
| `outputs/landslide_prediction.tif` | values 0-1 |
| `aoi_landslide_boulder.gpkg` | ≥ 1 feature |
| `prototype_metrics.csv` | IoU ≥ 0.50, AP50 ≥ 0.65 |

---
### Troubleshooting Tips (superset of *run.md*)
* **OOM on training** → lower `batch_size` in `config.yaml` and re-submit.
* **GDAL "out of memory"** → export `GDAL_CACHEMAX=2048` if RAM allows, or slice AOI.
* **Slurm job stuck in `CG`** → node out of GPUs; add `#SBATCH --constraint=T4` or your site tag.
* **GeoTIFF > 4 GB fails** → confirm `BIGTIFF=IF_SAFER` flag present (added in Step-2 scripts).

---
*Last updated:* $(date +%Y-%m-%d)
