#!/bin/bash
#SBATCH --job-name=coastal-waste-3d-example
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=./outputs/slurm-%j.out

# ═══════════════════════════════════════════════════════════════
#  Coastal-Waste-3D — traitement d'une scène complète (détection YOLO,
#  masques SAM, reconstruction MASt3R, segmentation 3D approche E)
#
#  Prérequis (voir README.md) :
#    - images de la scène dans   $COASTAL_SCENES/canette_coca/
#    - dépôt MASt3R + checkpoint dans $MAST3R_DIR
#    - poids SAM ViT-H et YOLO dans  ./data/models/
#    - venv "coastal_waste_3d" créé dans      ./venv
# ═══════════════════════════════════════════════════════════════

# ── Chemins de la plateforme (adapter si votre arborescence diffère) ──
export COASTAL_BASE=./data
export COASTAL_SCENES=./data/scenes
export COASTAL_OUTPUTS=./outputs
export HF_HOME=./data/hf_cache
export MAST3R_DIR=./data/mast3r
export SAM_CKPT=./data/models/sam_vit_h.pth
export YOLO_MODEL=./data/models/yolo_taco/best.pt

# ── Paramètres d'exécution ──
# 34 images max = calibré pour un GPU 16 Go (P100). Sur 32 Go : 60. Sur 40 Go+ : 80.
export COASTAL_MAX_IMAGES=34
# Tout est préchargé dans /data : aucune installation ni téléchargement au run
# (les nœuds de calcul n'ont pas nécessairement d'accès internet).
export SKIP_INSTALL_DEPS=1
export SKIP_NUMPY_FIX=1

# ── Environnement Python ──
source ./venv/bin/activate

mkdir -p "$COASTAL_OUTPUTS"

# ── Exécution : une scène, approche E (méthode complète) ──
# Remplacer "canette_coca" par le nom exact du dossier d'images.
# --approach all ajouterait la baseline S0 (comparaison), au prix d'un run plus long.
python ./pipeline.py \
    --scene canette_coca \
    --approach E
