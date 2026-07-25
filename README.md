# coastal-waste-3d

3D detection and segmentation pipeline for coastal waste objects from
uncalibrated multi-view images (YOLOv8 → SAM → MASt3R → consensus
segmentation modules).

This is the perception stage of a research pipeline aimed at detecting and
reconstructing waste objects in 3D for downstream robotic collection. It
takes a set of unposed images of a scene and produces, per detected object,
a segmented 3D point cloud plus geometric measurements (volume, dimensions).

> **Scale note**: reconstruction is done with MASt3R, which recovers geometry
> up to an arbitrary scale factor. All "cm³ / m" quantities in the output are
> **relative units** (no metric calibration). Ratios and comparisons between
> objects in the same scene remain valid.

## Pipeline overview

1. **Detection** — YOLOv8 detector locates candidate waste objects in each view
2. **Segmentation** — SAM (Segment Anything, ViT-H) produces per-object masks
3. **3D reconstruction** — MASt3R reconstructs the scene from the multi-view
   images without requiring camera calibration or poses
4. **Consensus segmentation** — per-object 2D masks are lifted and fused
   across views into a single 3D point cloud per object (module `E`,
   the main approach), with an ablation framework (`A`–`G`) to compare
   alternative consensus/cleaning strategies

## Project structure

```
coastal-waste-3d/
├── pipeline.py            # end-to-end pipeline (CLI: --scene, --approach, skip flags)
├── coastal_waste_3d/       # configuration and ablation framework
│   ├── config.py           # typed configuration + env_overrides()
│   ├── ablation.py         # generation of experimental configurations (A-G)
│   ├── runner.py           # experiment execution + statistical aggregation
│   └── minitest_e2.py      # targeted test: consensus (E2) x cleaning (DBSCAN)
├── jobs/job_scene_example.sh  # example SLURM job script
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
# MASt3R has its own requirements.txt in its cloned repo — install those too
```

Expected data layout (not included in this repo):

```
data/
├── scenes/<scene_name>/     # input images (.jpg/.png) for one scene
├── models/
│   ├── sam_vit_h.pth        # SAM ViT-H checkpoint (~2.4 GB)
│   └── yolo_taco/best.pt    # YOLOv8 detector trained on TACO
├── mast3r/                  # cloned MASt3R repo + checkpoint
└── hf_cache/                 # HuggingFace cache (created on first run)
```

## Running a scene

```bash
python pipeline.py --scene <scene_name> --approach E
```

See `jobs/job_scene_example.sh` for an example SLURM submission script
(GPU, 8 CPUs, 32 GB RAM recommended; runtime ~30-60 min per scene).

## Status

This is a work-in-progress research pipeline. The perception stage
(detection → segmentation → 3D reconstruction → consensus) is functional;
downstream registration/alignment against reference models is developed
separately and not part of this repository.
