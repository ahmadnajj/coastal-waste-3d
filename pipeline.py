#!/usr/bin/env python3
"""
Coastal-Waste-3D -- Pipeline de detection, reconstruction et segmentation 3D
de dechets cotiers a partir d images multi-vues non calibrees.
Author, 2026

Chaine de traitement
====================

Etage 1 -- perception et reconstruction (outils integres) :
  - YOLOv8 (entraine sur TACO)  : detection 2D des dechets (boites).
  - SAM ViT-H (prompte par boites) : masques 2D precis par objet.
  - MASt3R : reconstruction 3D multi-vues non calibree ; produit pour
    chaque vue un pointmap (point 3D par pixel, repere commun), des
    cartes de confiance et le nuage dense global.

Etage 2 -- segmentation 3D par consensus multi-vues (approche E) :
  E1 association native (indexation des pointmaps par les masques,
     sans reprojection) -> E2 consensus multi-vues (seuil en nombre
     absolu de vues) -> E3 separation objet/sol (garde-fou de
     verticalite anti-amputation) -> E4 nettoyage robuste
     (DBSCAN -> plan residuel -> compactage) -> E5 mesure
     (OBB par SVD + surface de Poisson, ratio qualite) ->
     E6 filtrage geometrique des faux positifs 3D.

Baseline S0 : association directe pixel -> point 3D natif, sans les
modules E (point de comparaison).
Approches A a D : variantes experimentales conservees pour les etudes
d ablation (selection via --approach).

Points techniques
=================
  - Repere de scene : detection automatique de l axe vertical par PCA
    globale (direction de plus petite variance du nuage) ; ne suppose
    pas la convention Z-up.
  - NMS 2D intra-image : deduplication des detections YOLO redondantes
    dans une meme vue (IoU > 0.8).
  - ECHELLE : MASt3R reconstruit a un facteur d echelle arbitraire
    pres. Toutes les grandeurs "cm3" / "m" du pipeline sont des
    UNITES RELATIVES (pas de calibration metrique). Les comparaisons
    inter-configurations et les ratios restent valides (le facteur
    d echelle se simplifie).

Configuration par variables d environnement (voir bloc ci-dessous).
Skip flags : --skip_mast3r, --skip_detection
"""

# ═══════════════════════════════════════════════════
# FIX NUMPY -- doit etre AVANT tout autre import
# Evite un conflit numpy conda vs pip selon l environnement
# ═══════════════════════════════════════════════════
import subprocess, sys, os

def _fix_numpy():
    if os.environ.get("SKIP_NUMPY_FIX","0") == "1":
        return
    try:
        import numpy as _np
        _ = _np.dtype('float64').itemsize
        import numpy.random.mtrand
    except (ValueError, ImportError, AttributeError):
        print("[FIX] Conflit numpy -- reinstallation numpy 1.26.4...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "numpy==1.26.4", "--force-reinstall", "-q",
                        "--break-system-packages"], check=True)
        mods = [k for k in sys.modules if k.startswith("numpy")]
        for m in mods: del sys.modules[m]
        print("[FIX] OK")

_fix_numpy()
# ═══════════════════════════════════════════════════

import argparse, os, json, gc, time, cv2, zipfile
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════
# Chemins -- override possibles via variables d environnement
# Le script de soumission exporte les chemins, le code les lit.
#
# Variables : COASTAL_BASE, COASTAL_SCENES, COASTAL_OUTPUTS, HF_HOME,
#             MAST3R_DIR, SAM_CKPT, YOLO_MODEL, COASTAL_MAX_IMAGES
# ═══════════════════════════════════════════════════
BASE_DIR    = os.environ.get("COASTAL_BASE", "./data")
SCENES_DIR  = os.environ.get("COASTAL_SCENES",
                             f"{BASE_DIR}/scenes")
OUT_DIR     = os.environ.get("COASTAL_OUTPUTS",
                             f"{BASE_DIR}/outputs")
HF_CACHE    = os.environ.get("HF_HOME",
                             f"{BASE_DIR}/hf_cache")
MAST3R_DIR  = os.environ.get("MAST3R_DIR", f"{BASE_DIR}/mast3r")
SAM_CKPT    = os.environ.get("SAM_CKPT",
                             f"{BASE_DIR}/models/sam_vit_h.pth")
YOLO_MODEL  = os.environ.get("YOLO_MODEL",
                             f"{BASE_DIR}/models/yolo_taco/best.pt")
IMG_EXT     = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

# Sous-echantillonnage des images (eviter OOM MASt3R global_alignment)
# Sur P100 16GB : ~34 images max. Sur V100 32GB : 60. Sur A100 40GB : 80+.
# Override via env COASTAL_MAX_IMAGES.
MAX_IMAGES = int(os.environ.get("COASTAL_MAX_IMAGES", "34"))

def subsample_images(all_files, max_n=None):
    """
    Sous-echantillonne regulierement une liste d images triees.
    Garde N images espacees uniformement sur la sequence d origine,
    de maniere a preserver la couverture spatiale (debut/milieu/fin).

    Critique pour MASt3R global_alignment qui charge tous les depthmaps
    simultanement : sa VRAM scale lineairement avec le nombre d images.
    """
    if max_n is None:
        max_n = MAX_IMAGES
    if max_n <= 0 or len(all_files) <= max_n:
        return all_files
    idx = np.linspace(0, len(all_files)-1, max_n).round().astype(int)
    idx = np.unique(idx)
    return [all_files[i] for i in idx]

# MASt3R -- anti-superposition
SCENE_GRAPH  = "swin-5"   # P100 16GB : swin-5 evite OOM (~510 paires sur 51 img)
LOOP_CLOSURE = 4          # paires debut<->fin uniquement, evite paires eloignees bruitees
MAST3R_ITER  = 200        # GPU : 200 suffisant si swin-5 (etait 300 pour swin-7)
CONF_THR     = 1.5

# Vote SAM
# 0 = adaptatif 70% du max (recommande)
# >0 = seuil fixe
VOTE_THRESH_A = 0

# GSCE-C
GSCE_EXPANSION      = 3.0   # rayon = dist_moy x facteur
GSCE_ANGLE          = 45.0  # angle max normales (degres)
GSCE_DEPTH          = 3.0   # tolerance saut profondeur
GSCE_CONF_THR       = 0.1   # score confiance minimum
GSCE_K_NEIGHBORS    = 5     # voisins pour calcul scores
GSCE_MAX_ITER       = 50    # iterations max (augmente de 15 a 50)
GSCE_CONV_RATIO     = 0.001 # convergence si n_new/n_total < 0.1%

# Poids scores GSCE-C
W_NORMAL     = 0.35
W_PROFONDEUR = 0.30
W_DISTANCE   = 0.20
W_DENSITE    = 0.15

# Approche C
CONF_THRESH_C = 0.3
VOTE_RATIO_C  = 0.15

# NMS 3D -- elimination doubles detections
NMS_3D_DIST = 0.15   # 2 dechets < 15cm = meme objet -> garder le plus haut vote

# ═══════════════════════════════════════════════════
# FILTRE FAUX POSITIFS
# ═══════════════════════════════════════════════════

# Filtre 2D multi-vues (avant SAM)
FP2D_MIN_VIEWS_RATIO = 0.10  # Une detection doit apparaitre dans
                              # au moins 10% des images pour etre gardee
FP2D_IOU_THRESHOLD   = 0.30   # IoU minimum pour considerer 2 boites
                              # comme la meme detection entre images

# Filtre geometrique 3D (apres reconstruction)
# Un candidat doit passer >= MIN_RULES_PASS / 6 regles pour etre garde
FP3D_MIN_RULES_PASS  = 5

# Regle 1 -- Coplanarite avec le sol
FP3D_GROUND_DIST_MAX     = 0.02    # 2 cm
FP3D_GROUND_RATIO_MAX    = 0.80    # 80% pts coplanaires = FP

# Regle 2 -- Epaisseur OBB minimale
FP3D_MIN_THICKNESS       = 0.015   # 1.5 cm
FP3D_MIN_RATIO_DIMS      = 0.05    # ratio min/max dim

# Regle 3 -- Vote multi-vues
FP3D_MIN_VIEW_RATIO      = 0.25    # 25% des cameras qui voient le candidat

# Regle 4 -- Densite locale
FP3D_MIN_DENSITY_RATIO   = 0.10    # 10% de la densite mediane scene

# Regle 5 -- Volume realiste
FP3D_VOLUME_MIN_CM3      = 5.0      # 5 cm3 (briquet, capuchon)
FP3D_VOLUME_MAX_CM3      = 50000.0  # 50000 cm3 (carton pizza max)

# Regle 6 -- Position relative au sol
FP3D_Z_MIN_M             = -0.05    # Pas sous le sol (5cm tolerance)
FP3D_Z_MAX_M             = 1.50     # Pas au dessus de 1.5m

# Sauvegarde des FP pour pseudo-HNM
SAVE_FP_FOR_HNM          = True

# ═══════════════════════════════════════════════════
# PRE-TRAITEMENTS 2D
# ═══════════════════════════════════════════════════

# NMS 2D intra-image (avant filtre multi-vues)
# Si 2 detections dans la meme image ont IoU > seuil, on garde
# celle avec la confiance YOLO max
NMS_2D_INTRA_IOU         = 0.80

# Approche D adaptative
# Volume attendu par defaut pour un dechet cotier moyen (cm3)
EXPECTED_VOLUME_CM3      = 500.0     # canette ~400, bouteille ~600
# Seuils de declenchement du raffinement
D_VOLUME_BRUIT_RATIO     = 3.0       # > 3x volume attendu = trop bruite
D_VOLUME_MAIGRE_RATIO    = 0.3       # < 0.3x volume attendu = trop maigre
# Mode adaptatif : 'auto' choisit B ou A selon diagnostic
# Force 'B' ou 'A' pour forcer le mode
D_MODE                   = "auto"

# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════

# Poisson
POISSON_SCALE      = 1.3
FOND_SUBSAMPLE     = 20
DBSCAN_EPS         = 0.02
DBSCAN_MIN_SAMPLES = 30
COLORS_DECHETS = [
    [255, 50,  50],
    [50,  150, 255],
    [50,  255, 50],
    [255, 165, 0],
    [200, 50,  255],
]
# ═══════════════════════════════════════════════════


def log(msg, level=0):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]" + "  "*level + " " + msg, flush=True)


def clear_gpu():
    """Libere la memoire GPU."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except: pass
    gc.collect()


# ───────────────────────────────────────────────────
# SETUP
# ───────────────────────────────────────────────────
def install_deps():
    """
    Installe les dependances manquantes.
    Ne reinstalle pas torch/torchvision/numpy (deja dans conda).
    Skip si SKIP_INSTALL_DEPS=1 (venv fige, sans installation au run).
    """
    if os.environ.get("SKIP_INSTALL_DEPS","0") == "1":
        log("install_deps SKIPPED (SKIP_INSTALL_DEPS=1)")
        return
    log("Installation dependances...")
    deps = ["ultralytics","segment-anything","open3d",
            "scikit-learn","opencv-python","tqdm","einops","roma"]
    for dep in deps:
        check = subprocess.run([sys.executable,"-c",
            f"import {dep.replace('-','_').split('[')[0]}"],
            capture_output=True)
        if check.returncode == 0:
            log(f"  OK {dep} (present)", 1)
            continue
        r = subprocess.run([sys.executable,"-m","pip","install",
            "-q", dep, "--no-deps", "--break-system-packages"],
            capture_output=True)
        log(f"  {'OK' if r.returncode==0 else 'ERREUR'} {dep}", 1)
    log("OK dependances")


def setup_mast3r():
    """Clone et configure MASt3R."""
    if not os.path.exists(MAST3R_DIR):
        log("Clonage MASt3R...")
        subprocess.run(["git","clone","--recursive",
            "https://github.com/naver/mast3r.git", MAST3R_DIR], check=True)
        subprocess.run([sys.executable,"-m","pip","install","-q","-r",
            f"{MAST3R_DIR}/dust3r/requirements.txt",
            "--break-system-packages"], check=True)
    for m in [k for k in sys.modules if "mast3r" in k or "dust3r" in k]:
        del sys.modules[m]
    sys.path = [p for p in sys.path if "mast3r" not in p]
    sys.path.insert(0, MAST3R_DIR)
    sys.path.insert(0, f"{MAST3R_DIR}/dust3r")
    os.environ["HF_HOME"] = HF_CACHE
    log("OK MASt3R")


def download_sam():
    """Telecharge le checkpoint SAM ViT-H si absent."""
    os.makedirs(os.path.dirname(SAM_CKPT), exist_ok=True)
    if not os.path.exists(SAM_CKPT):
        log("Telechargement SAM ViT-H (~2.5 GB)...")
        subprocess.run(["wget","-q","--show-progress","-O", SAM_CKPT,
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"], check=True)
    log("OK SAM")


# ───────────────────────────────────────────────────
# UTILS PLY
# ───────────────────────────────────────────────────
def write_ply(path, pts, cols):
    """
    Sauvegarde un nuage de points au format PLY ASCII.
    pts  : (N,3) float -- coordonnees x,y,z
    cols : (N,3) float [0,1] ou uint8 [0,255]
    """
    cols = np.array(cols)
    if cols.dtype != np.uint8:
        cols = (np.clip(cols,0,1)*255).astype(np.uint8)
    with open(path,"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt,col in zip(pts,cols):
            f.write(f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f} "
                    f"{col[0]} {col[1]} {col[2]}\n")


def write_ply_confidence(path, pts, scores):
    """
    Sauvegarde un nuage avec coloration par score de confiance GSCE-C.
    Rouge (score=0) -> Jaune (score=0.5) -> Vert (score=1.0)
    Permet de visualiser dans CloudCompare quels points sont fiables.
    """
    with open(path,"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt,s in zip(pts,scores):
            if s < 0.5:
                r,g,b = 255, int(s*2*255), 0
            else:
                r,g,b = int((1-s)*2*255), 255, 0
            f.write(f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f} {r} {g} {b}\n")


# ───────────────────────────────────────────────────
# REPERE NORMALISE
# ───────────────────────────────────────────────────
def compute_scene_frame(points):
    """
    Calcule un repere normalise robuste, sans supposer Z-up.

    Le Z monde de MASt3R n est PAS toujours vertical.
    Detection : l axe vertical est celui de plus PETITE variance
    dans le nuage (les scenes sont generalement etalees horizontalement
    et concentrees verticalement).

    Etapes :
    1. PCA globale sur tout le nuage -> 3 axes principaux
    2. L axe de plus petite variance = candidat vertical
    3. Verifier : etendue le long de cet axe << etendue X,Y
    4. Selectionner les points "bas" le long de cet axe pour sol
    5. PCA locale sur les pts bas pour affiner la normale sol
    6. Construire repere : Z=normale_sol, X=axe horizontal principal

    Retourne : origin (3,), R (3,3 matrice rotation monde->scene),
               T (3,) translation -R.T @ origin
    """
    if len(points) < 100:
        origin = points.mean(axis=0)
        return origin, np.eye(3), -origin

    # Etape 1 : PCA globale pour identifier l axe vertical candidat
    centered = points - points.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # S = valeurs singulieres ordonnees par taille decroissante
    # Vt = axes principaux (Vt[0] = grande etendue, Vt[2] = petite etendue)

    # Candidat vertical = axe de plus petite variance
    vertical_candidate = Vt[-1]
    # S'assurer qu il pointe vers le haut (composante z monde positive
    # si elle n est pas trop faible, sinon composante y monde positive)
    if abs(vertical_candidate[2]) > abs(vertical_candidate[1]):
        if vertical_candidate[2] < 0:
            vertical_candidate = -vertical_candidate
    else:
        if vertical_candidate[1] < 0:
            vertical_candidate = -vertical_candidate

    # Etape 2 : projeter tous les points sur l axe vertical
    heights = (points - points.mean(axis=0)) @ vertical_candidate

    # Etape 3 : selectionner les 20% les plus bas comme candidats sol
    h_low  = np.percentile(heights, 10)
    h_high = np.percentile(heights, 90)
    sol_mask = heights < (h_low + 0.2*(h_high - h_low))
    sol_pts  = points[sol_mask]

    if len(sol_pts) < 100:
        origin = points.mean(axis=0)
        R_default = np.stack([Vt[0], Vt[1], vertical_candidate], axis=1)
        return origin, R_default, -R_default.T @ origin

    # Etape 4 : PCA locale sur les pts bas pour affiner la normale sol
    sol_center = sol_pts.mean(axis=0)
    sol_centered = sol_pts - sol_center
    _, _, Vt_sol = np.linalg.svd(sol_centered, full_matrices=False)
    normal_sol = Vt_sol[-1]

    # Aligner avec le candidat vertical global (eviter inversion)
    if np.dot(normal_sol, vertical_candidate) < 0:
        normal_sol = -normal_sol

    z_axis = normal_sol / np.linalg.norm(normal_sol)

    # Etape 5 : axes X et Y dans le plan horizontal
    # X = direction principale du sol (premiere PCA des pts sol)
    x_axis = Vt_sol[0]
    # Gram-Schmidt : retirer composante verticale
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    origin = sol_center.copy()
    return origin, R, -R.T @ origin


def estimate_ground_plane(points):
    """
    RANSAC sur le plan sol apres detection robuste de l axe vertical.

    Prendre les points de z monde minimum n est pas fiable si l axe
    vertical n est pas Z.

    Methode : utilise compute_scene_frame pour identifier l axe vertical,
    puis RANSAC sur les points bas le long de cet axe.

    Retourne (normale, d) tels que normale.dot(point) + d = 0
    pour les points sur le plan sol.
    """
    import open3d as o3d

    if len(points) < 100:
        return None

    # Reutiliser compute_scene_frame pour identifier l axe vertical
    try:
        _, R, _ = compute_scene_frame(points)
        vertical_axis = R[:, 2]  # 3e colonne = axe Z scene = vertical
    except Exception:
        return None

    # Projeter les points sur l axe vertical
    centered = points - points.mean(axis=0)
    heights = centered @ vertical_axis

    # Prendre les 30% les plus bas comme candidats sol
    h_thresh = np.percentile(heights, 30)
    low_mask = heights <= h_thresh
    low_pts = points[low_mask]

    if len(low_pts) < 100:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(low_pts)

    try:
        model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000)
        a, b, c, d = model
        normal = np.array([a, b, c])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            return None
        normal = normal / norm_len
        d = d / norm_len
        # Aligner avec l axe vertical detecte (et pas avec Z monde)
        if np.dot(normal, vertical_axis) < 0:
            normal = -normal
            d = -d
        return normal, d
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# SUPPRESSION DU SOL AU NIVEAU SCENE -- Priorite 1 (contribution)
# ═══════════════════════════════════════════════════════════════════
#
# Probleme : les objets (canettes, sacs) reposent SUR le sol. Aucune
# methode basee distance/continuite geometrique (vote 3D, GSCE-C) ne
# peut separer l objet du sol coplanaire qu il touche -> "galette"
# d herbe/sol importee autour de l objet.
#
# Solution : retirer le plan-sol de TOUTE la scene (un seul plan estime
# globalement, robuste) AVANT l extraction des seeds objet. Le sol ne
# peut alors plus etre importe par aucune approche.
#
# Avantage vs per-object plane (clean_seeds) : le plan est estime sur
# l ensemble du nuage (millions de pts), donc tres stable, contrairement
# au RANSAC par-objet qui peut confondre objet plat et sol.
#
# Override : COASTAL_GROUND_EPS (m, defaut 0.015 = 1.5cm)
#            COASTAL_GROUND_REMOVE=0 pour desactiver (ablation)
# ═══════════════════════════════════════════════════════════════════

def remove_ground_from_scene(points, colors, ground_plane,
                              scene_origin=None, scene_R=None,
                              eps=None, verbose=True):
    """
    Retire les points du sol -- VERSION ROBUSTE en repere normalise.

    Choix critique : PAS de seuil metrique absolu. Le repere MASt3R
    est a une echelle arbitraire ; un seuil exprime en centimetres n y
    a pas de sens et peut classer 100% des points comme "sol".

    Approche : on travaille dans le REPERE SCENE normalise
    (Z = hauteur reelle au-dessus du sol, calcule par compute_scene_frame).
    Le seuil devient RELATIF : on retire les points sous un percentile
    bas des hauteurs. Robuste a toute echelle/orientation.

    Args:
        points, colors : nuage complet
        ground_plane   : (normal, d) -- utilise en fallback
        scene_origin, scene_R : repere scene (Z vertical) -- PRIORITAIRE
        eps : si fourni, seuil de hauteur en unites scene (sinon auto)
        verbose : log

    Returns:
        points_filt, colors_filt, keep_mask
    """
    if os.environ.get("COASTAL_GROUND_REMOVE", "1") != "1":
        if verbose:
            log("Ground removal DESACTIVE (ablation)", 1)
        return points, colors, np.ones(len(points), dtype=bool)

    # Methode prioritaire : repere scene normalise
    if scene_origin is not None and scene_R is not None:
        # Z dans le repere scene = hauteur au-dessus du sol
        heights = (points - scene_origin) @ scene_R[:, 2]
        h_lo = np.percentile(heights, 5)
        h_hi = np.percentile(heights, 95)
        span = max(h_hi - h_lo, 1e-6)
        # FIX A : epaisseur de coupe QUASI-CONSTANTE au lieu de 3% du span.
        # Probleme avant : 3% du span variait selon la scene (0.001 a 0.005),
        # coupant la base des objets DEBOUT (canette scene00 -> anneau).
        # Solution : on prend le MINIMUM entre une petite fraction (1%) et
        # un plafond absolu, pour ne retirer que la fine nappe de sol.
        # MASt3R metric donne une echelle ~ metrique : 1% du span sur une
        # scene de ~0.3-0.5 d etendue = ~3-5mm, ce qui preserve les objets.
        margin_frac = float(os.environ.get("COASTAL_GROUND_MARGIN", "0.01"))
        thresh = h_lo + margin_frac * span
        keep_mask = heights > thresh
        n_removed = int((~keep_mask).sum())
        frac = n_removed / max(len(points), 1)
        # GARDE-FOU : si on retire > 90% ou < 1%, c'est suspect.
        if frac > 0.90 or frac < 0.01:
            if verbose:
                log(f"Ground removal ANNULE : {frac*100:.0f}% serait "
                    f"retire (suspect) -> on garde tout le nuage", 1)
            return points, colors, np.ones(len(points), dtype=bool)
        if verbose:
            log(f"Ground removal (repere normalise) : {n_removed:,} pts "
                f"sol retires ({frac*100:.1f}%) | seuil hauteur="
                f"{thresh:.4f} | span={span:.3f} | "
                f"reste {int(keep_mask.sum()):,} pts", 1)
        return points[keep_mask], colors[keep_mask], keep_mask

    # Fallback : plan RANSAC (avec garde-fou)
    if ground_plane is None:
        if verbose:
            log("Ground removal SKIP : ni repere ni plan disponibles", 1)
        return points, colors, np.ones(len(points), dtype=bool)

    normal, d = ground_plane
    signed_dist = points @ normal + d
    # Seuil RELATIF base sur la distribution des distances
    s_lo = np.percentile(signed_dist, 5)
    s_hi = np.percentile(signed_dist, 95)
    span = max(s_hi - s_lo, 1e-6)
    thresh = s_lo + 0.03 * span
    keep_mask = signed_dist > thresh
    frac = (~keep_mask).sum() / max(len(points), 1)
    if frac > 0.90 or frac < 0.02:
        if verbose:
            log(f"Ground removal (plan) ANNULE : {frac*100:.0f}% suspect "
                f"-> on garde tout", 1)
        return points, colors, np.ones(len(points), dtype=bool)
    if verbose:
        log(f"Ground removal (plan) : {int((~keep_mask).sum()):,} pts "
            f"retires ({frac*100:.1f}%)", 1)
    return points[keep_mask], colors[keep_mask], keep_mask


def transform_to_scene_frame(points, origin, R):
    """
    Transforme des points du repere monde vers le repere scene.
    Dans le repere scene : Z=0 sur le sol, Z>0 au-dessus du sol.
    """
    return (points - origin) @ R


def format_coordinates(obb_center_scene):
    """
    Formate les coordonnees dans le repere scene de maniere lisible.
    x = horizontal gauche-droite
    y = horizontal avant-arriere
    z = vertical (hauteur au-dessus du sol)
    """
    x, y, z = obb_center_scene
    return {
        "x_m":  round(float(x), 4),
        "y_m":  round(float(y), 4),
        "z_m":  round(float(z), 4),
        "description": (
            f"x={x:.3f}m (horizontal), "
            f"y={y:.3f}m (profondeur), "
            f"z={z:.3f}m (hauteur sol)"
        )
    }


# ───────────────────────────────────────────────────
# ANNOTATION LISIBLE
# ───────────────────────────────────────────────────
def write_annotation(path, scene_name, dechets_info):
    """
    Ecrit un fichier d annotation lisible en JSON structure.
    Contient toutes les informations necessaires pour le robot.

    Format :
    {
      "scene": "scene00",
      "repere": "sol = Z=0, X=horizontal, Y=profondeur",
      "dechets": [
        {
          "id": 0,
          "classe": "metal (canette)",
          "approche": "B",
          "position": {
            "x_m": 0.013,  // horizontal gauche-droite
            "y_m": -0.047, // profondeur avant-arriere
            "z_m": 0.041,  // hauteur au-dessus du sol
          },
          "dimensions": {
            "longueur_m": 0.285,
            "largeur_m":  0.072,
            "hauteur_m":  0.068,
          },
          "volume_cm3":   389.4,
          "surface_cm2":  312.1,
          "score_gsce":   0.887,
          "collecte": "POSSIBLE" ou "HORS_PORTEE"
        }
      ]
    }
    """
    annotation = {
        "scene": scene_name,
        "repere": {
            "description": "Repere normalise sol",
            "origine":     "Centroide du nuage projete sur le plan sol",
            "axe_X":       "Horizontal gauche-droite (metres)",
            "axe_Y":       "Horizontal avant-arriere / profondeur (metres)",
            "axe_Z":       "Vertical, Z=0 sur le sol, Z>0 au-dessus (metres)",
        },
        "n_dechets": len(dechets_info),
        "dechets": dechets_info
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


# ───────────────────────────────────────────────────
# NMS 3D -- suppression doubles detections
# ───────────────────────────────────────────────────
def nms_3d(detections_dict, points, votes_dict, threshold_m=NMS_3D_DIST):
    """
    Non-Maximum Suppression en 3D, adaptatif a la taille des objets.

    FIX bug multi-objets (scene08) : avant, threshold_m fixe causait la
    suppression d objets distincts mais proches (canette + carton a 14cm).
    Maintenant : on considere 2 detections comme "doublons" SI ET SEULEMENT SI
    leurs bounding spheres se recouvrent suffisamment, ce qui depend de
    leur taille reelle.

    Principe :
    - Pour chaque detection, calculer le rayon de la bounding sphere
      depuis les seeds (extent des pts autour du centre)
    - Deux detections (i, j) sont "doublons" si :
         dist(center_i, center_j) < max(r_i, r_j) * IOU_FACTOR
      OU
         dist(center_i, center_j) < threshold_m  (legacy fallback)
    - On garde le critere le plus restrictif (objets distincts pas fusionnes)

    Args:
        detections_dict : {det_idx: {"center":..., "vote_max":..., "classe":...,
                                     "conf":..., "pts_seeds":... [optional]}}
        threshold_m     : seuil minimum (legacy, plancher de securite)

    Returns:
        Liste des indices a garder
    """
    if not detections_dict:
        return []

    indices  = list(detections_dict.keys())
    centers  = np.array([detections_dict[i]["center"] for i in indices])
    votes    = np.array([detections_dict[i]["vote_max"] for i in indices])

    # Calcul du rayon par detection (depuis seeds si dispo, sinon defaut)
    radii = np.zeros(len(indices))
    for k, i in enumerate(indices):
        info = detections_dict[i]
        if "pts_seeds" in info and len(info["pts_seeds"]) >= 5:
            ps = np.asarray(info["pts_seeds"])
            ctr = ps.mean(axis=0)
            radii[k] = float(np.linalg.norm(ps - ctr, axis=1).mean())
        elif "keep_mask" in info:
            # Cas approche A : on a un masque sur le nuage global
            mask = info["keep_mask"]
            if mask.sum() >= 5:
                ps = points[mask]
                ctr = ps.mean(axis=0)
                radii[k] = float(np.linalg.norm(ps - ctr, axis=1).mean())
            else:
                radii[k] = 0.05  # 5cm defaut
        else:
            radii[k] = 0.05  # 5cm defaut

    # Critere adaptatif : facteur sur somme des rayons
    # < 1 = objets recouvrants franchement
    # = 1 = bounding spheres tangentes
    # > 1 = objets clairement separes
    IOU_FACTOR = 0.6   # 60% de la somme des rayons

    keep  = []
    used  = set()
    order = np.argsort(-votes)

    for i in order:
        idx = indices[i]
        if idx in used:
            continue
        keep.append(idx)
        dists = np.linalg.norm(centers - centers[i], axis=1)
        for j in range(len(indices)):
            if i == j:
                used.add(indices[j])
                continue
            # Seuil adaptatif : la somme des rayons fois facteur
            adaptive_thresh = (radii[i] + radii[j]) * IOU_FACTOR
            # Plancher : threshold_m / 4 pour eviter fusion accidentelle
            # de petits objets meme tres proches mais distincts
            effective_thresh = max(adaptive_thresh, threshold_m * 0.25)
            if dists[j] < effective_thresh:
                used.add(indices[j])

    log(f"NMS 3D : {len(indices)} detections -> {len(keep)} gardees "
        f"({len(indices)-len(keep)} doublons supprimes)", 2)
    return keep


# ───────────────────────────────────────────────────
# FILTRES FAUX POSITIFS
# ───────────────────────────────────────────────────
def bbox_iou_2d(box1, box2):
    """
    Calcule l IoU entre deux bounding boxes 2D.
    box format : [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms_2d_intra_image(all_det, iou_threshold=None):
    """
    NMS 2D intra-image.

    Si 2 detections YOLO dans la MEME image ont IoU > seuil,
    elles representent probablement le meme dechet detecte avec
    2 classes differentes (ex: 'plastic 78%' et 'metal 65%').
    On garde celle avec la confiance YOLO la plus haute.

    Different du filtre multi-vues qui compare ENTRE images.

    Args:
        all_det : dict {img_name: {image_size, detections}}
        iou_threshold : seuil IoU (defaut NMS_2D_INTRA_IOU = 0.80)

    Returns:
        all_det_nms : dict filtre
        stats : {n_before, n_after, n_rejected}
    """
    if iou_threshold is None:
        iou_threshold = NMS_2D_INTRA_IOU

    log("[NMS 2D intra-image] Suppression doublons intra-image...", 1)
    total_before = sum(len(v.get("detections", [])) for v in all_det.values())
    rejected = 0
    all_det_nms = {}

    for img_name, entry in all_det.items():
        dets = entry.get("detections", [])
        W, H = entry.get("image_size", [1, 1])

        if len(dets) <= 1:
            all_det_nms[img_name] = entry
            continue

        # Trier par confiance decroissante : on traite d abord les fortes
        order = sorted(range(len(dets)),
                       key=lambda i: -dets[i].get("conf", 0))

        keep_mask = [True] * len(dets)
        for i_idx, i in enumerate(order):
            if not keep_mask[i]:
                continue
            box_i = dets[i]["bbox_2d"]
            for j in order[i_idx+1:]:
                if not keep_mask[j]:
                    continue
                box_j = dets[j]["bbox_2d"]
                if bbox_iou_2d(box_i, box_j) >= iou_threshold:
                    # j est supprime (i a une conf >= j car ordre decroissant)
                    keep_mask[j] = False
                    rejected += 1

        kept_dets = [dets[i] for i in range(len(dets)) if keep_mask[i]]
        all_det_nms[img_name] = {
            "image_size": entry.get("image_size", [W, H]),
            "detections": kept_dets
        }

    total_after = total_before - rejected
    log(f"NMS intra : {total_before} -> {total_after} "
        f"({rejected} doublons intra-image supprimes)", 2)

    return all_det_nms, {
        "n_before": total_before,
        "n_after":  total_after,
        "n_rejected": rejected
    }


def filter_fp_2d_multiview(all_det, camera_data=None):
    """
    Filtre 2D multi-vues : elimine les detections qui n apparaissent
    que dans une seule image (artefact textural local).

    Principe simple : compter pour chaque image combien de detections
    elle contient. Si une detection est isolee (sans equivalent IoU
    dans les images voisines), c est probablement un FP.

    Note : ce filtre est conservateur -- il ne se base pas encore sur
    la geometrie 3D. Le vrai filtrage se fait au niveau 3D apres MASt3R.

    Retourne all_det_filtered avec les FP isoles marques.
    Les statistiques sont retournees pour log.
    """
    log("[FILTRE FP 2D] Analyse multi-vues...", 1)

    img_names = sorted(all_det.keys())
    n_imgs = len(img_names)
    if n_imgs < 3:
        log("Trop peu d images pour filtre multi-vues -- skip", 2)
        return all_det, {"n_before": 0, "n_after": 0, "n_rejected": 0}

    # Compter combien de fois chaque detection a un equivalent dans les
    # images suivantes (par IoU sur position normalisee)
    total_before = sum(len(v.get("detections", [])) for v in all_det.values())
    rejected_count = 0
    all_det_filtered = {}

    for img_name in img_names:
        entry = all_det[img_name]
        dets = entry.get("detections", [])
        W, H = entry.get("image_size", [1, 1])
        if not dets:
            all_det_filtered[img_name] = entry
            continue

        # Pour chaque detection, compter les detections "similaires"
        # dans les autres images (normalisees au range [0,1])
        kept_dets = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox_2d"]
            box_norm = [x1/W, y1/H, x2/W, y2/H]

            # Compter combien d images ont une detection similaire
            n_supports = 1  # cette image elle-meme
            for other_name in img_names:
                if other_name == img_name: continue
                other_entry = all_det[other_name]
                other_W, other_H = other_entry.get("image_size", [1, 1])
                for other_det in other_entry.get("detections", []):
                    ox1, oy1, ox2, oy2 = other_det["bbox_2d"]
                    other_norm = [ox1/other_W, oy1/other_H,
                                   ox2/other_W, oy2/other_H]
                    # Note : IoU sur coordonnees normalisees est approximatif
                    # mais suffit comme proxy de coherence multi-vues
                    if bbox_iou_2d(box_norm, other_norm) >= FP2D_IOU_THRESHOLD:
                        n_supports += 1
                        break  # une support par image suffit

            # Si la detection est supportee par >= seuil d images, on la garde
            if n_supports >= max(2, int(n_imgs * FP2D_MIN_VIEWS_RATIO)):
                kept_dets.append(det)
            else:
                rejected_count += 1

        all_det_filtered[img_name] = {
            "image_size": entry.get("image_size", [W, H]),
            "detections": kept_dets
        }

    total_after = sum(len(v.get("detections", [])) for v in all_det_filtered.values())
    log(f"Detections 2D : {total_before} -> {total_after} "
        f"({rejected_count} rejetees comme FP isoles)", 2)

    return all_det_filtered, {
        "n_before": total_before,
        "n_after":  total_after,
        "n_rejected": rejected_count
    }


def point_to_plane_distance(points, plane):
    """
    Distance signee de chaque point au plan.
    plane = (normal, d) tel que normal.dot(p) + d = 0.
    """
    if plane is None:
        return np.zeros(len(points))
    normal, d = plane
    return np.abs(points @ normal + d)


def filter_fp_3d_geometric(candidate_info, ground_plane, scene_density,
                            n_views_visible, save_dir=None):
    """
    Filtre geometrique 3D applique a un candidat dechet.
    Applique les 6 regles -- doit passer >= FP3D_MIN_RULES_PASS / 6.

    Args:
        candidate_info : dict {
            "pts": array (N,3) des points 3D du candidat,
            "obb_dims": [L, l, h],
            "obb_center": [x, y, z],
            "volume_cm3": float,
            "z_sol": float (hauteur au-dessus du sol)
        }
        ground_plane    : (normal, d) ou None
        scene_density   : densite moyenne du nuage scene (pts/m3)
        n_views_visible : ratio views ou ce candidat est confirme (0-1)
        save_dir        : si non None, sauve les FP pour HNM

    Returns:
        (is_valid, dict_regles) : True/False + detail par regle
    """
    pts        = candidate_info["pts"]
    dims       = np.array(candidate_info["obb_dims"])
    center     = np.array(candidate_info["obb_center"])
    volume_cm3 = candidate_info["volume_cm3"]
    z_sol      = candidate_info.get("z_sol", center[2])

    rules = {}

    # FIX B : detection d un OBJET PLAT LEGITIME (papier, emballage,
    # sac aplati). Ces dechets sont intrinsequement coplanaires et fins,
    # donc ils echouent naturellement les regles 1/2/5 concues pour
    # filtrer le sol. On les exempte SI :
    #   - leur empreinte au sol est petite (< seuil, ce n est pas le sol
    #     qui s etend sur toute la scene)
    #   - ils ont une densite/coherence suffisante (vrai objet, pas bruit)
    # Override : FP3D_FLAT_FOOTPRINT_MAX (m, defaut 0.35 = 35cm de cote max)
    dims_sorted = np.sort(dims)[::-1]  # [L, l, h] decroissant
    footprint = dims_sorted[0] * dims_sorted[1]  # surface au sol (m2)
    FLAT_FOOTPRINT_MAX = float(os.environ.get("FP3D_FLAT_FOOTPRINT_MAX", "0.35"))
    # Un objet est "plat et petit" si sa plus grande dimension < seuil
    # ET son epaisseur est faible devant son empreinte
    is_small_flat = (dims_sorted[0] < FLAT_FOOTPRINT_MAX and
                     dims_sorted[2] < dims_sorted[0] * 0.5)

    # ── Regle 1 : Coplanarite avec sol ──
    if ground_plane is None or len(pts) < 10:
        rules["1_coplanar"] = True   # pas de jugement
    elif is_small_flat:
        rules["1_coplanar"] = True   # objet plat legitime exempte
    else:
        dists = point_to_plane_distance(pts, ground_plane)
        ratio_coplanar = np.mean(dists < FP3D_GROUND_DIST_MAX)
        rules["1_coplanar"] = ratio_coplanar < FP3D_GROUND_RATIO_MAX
        # FP si trop coplanaire avec le sol

    # ── Regle 2 : Epaisseur OBB ──
    min_dim = dims.min()
    max_dim = dims.max() if dims.max() > 0 else 1
    ratio_dims = min_dim / max_dim
    if is_small_flat:
        rules["2_thickness"] = True  # objet plat legitime exempte
    else:
        rules["2_thickness"] = (min_dim >= FP3D_MIN_THICKNESS
                                 and ratio_dims >= FP3D_MIN_RATIO_DIMS)

    # ── Regle 3 : Vote multi-vues ──
    rules["3_multiview"] = n_views_visible >= FP3D_MIN_VIEW_RATIO

    # ── Regle 4 : Densite locale ──
    if len(pts) < 10 or scene_density <= 0:
        rules["4_density"] = True
    else:
        vol_m3 = max(dims[0] * dims[1] * dims[2], 1e-9)
        local_density = len(pts) / vol_m3
        rules["4_density"] = local_density >= scene_density * FP3D_MIN_DENSITY_RATIO

    # ── Regle 5 : Volume realiste ──
    if is_small_flat:
        # Pour un objet plat, on autorise un volume plus faible
        # (un papier fin a un petit volume OBB legitime)
        rules["5_volume"] = (volume_cm3 <= FP3D_VOLUME_MAX_CM3)
    else:
        rules["5_volume"] = (FP3D_VOLUME_MIN_CM3 <= volume_cm3 <= FP3D_VOLUME_MAX_CM3)

    # ── Regle 6 : Position relative au sol ──
    rules["6_z_sol"] = (FP3D_Z_MIN_M <= z_sol <= FP3D_Z_MAX_M)

    # ── Combinaison ──
    n_pass = sum(1 for v in rules.values() if v)
    is_valid = n_pass >= FP3D_MIN_RULES_PASS

    return is_valid, rules, n_pass


def save_false_positive(out_dir, scene_name, det_idx, candidate_info,
                          source_images, rules_failed):
    """
    Sauvegarde un faux positif confirme pour pseudo-HNM futur.

    Structure :
      out_dir/false_positives/scene_X_det_Y/
          crops/IMG_NNNN_crop.jpg  -- crops 2D des images sources
          info.json                -- raisons du rejet + metadata
    """
    if not SAVE_FP_FOR_HNM:
        return

    fp_dir = Path(out_dir) / "false_positives" / f"{scene_name}_det_{det_idx}"
    fp_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = fp_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    # Sauvegarder info.json
    info = {
        "scene": scene_name,
        "det_idx": int(det_idx),
        "rejected_rules": [k for k, v in rules_failed.items() if not v],
        "passed_rules":   [k for k, v in rules_failed.items() if v],
        "obb_dims":   candidate_info.get("obb_dims", []),
        "obb_center": candidate_info.get("obb_center", []),
        "volume_cm3": candidate_info.get("volume_cm3", 0),
        "n_points":   len(candidate_info.get("pts", [])),
    }
    with open(fp_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Sauvegarder crops (si source_images fourni)
    if source_images:
        for img_path, bbox in source_images[:5]:  # max 5 crops
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                x1, y1, x2, y2 = bbox
                pad = 10
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(img.shape[1], x2 + pad)
                y2 = min(img.shape[0], y2 + pad)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_name = Path(img_path).stem + "_crop.jpg"
                    cv2.imwrite(str(crops_dir / crop_name), crop)
            except Exception:
                continue


def compute_scene_density(points):
    """
    Estime la densite moyenne du nuage scene (points/m3).
    Utilise un sous-echantillon pour aller vite.
    """
    from sklearn.neighbors import KDTree
    if len(points) < 100:
        return 0.0
    sample = points[::max(1, len(points) // 5000)]
    tree = KDTree(sample)
    # Distance moyenne au 5eme voisin -> rayon caracteristique
    dists, _ = tree.query(sample, k=6)  # k=6 car le 1er est le point lui-meme
    r_avg = dists[:, 5].mean()
    if r_avg <= 0:
        return 0.0
    # Densite = nb points / volume sphere de rayon r_avg
    vol = (4.0 / 3.0) * np.pi * r_avg**3
    return 1.0 / vol  # densite locale moyenne


def count_views_visible(candidate_center, candidate_radius,
                         camera_data, view_threshold=0.01):
    """
    Compte dans combien de cameras le candidat est geometriquement visible
    (projete dans l image valide).

    Retourne (n_visible, n_total, ratio).
    """
    n_total = len(camera_data)
    if n_total == 0:
        return 0, 0, 0.0
    n_visible = 0
    for img_name, cam in camera_data.items():
        K    = np.array(cam["K"])
        pose = np.array(cam["pose_4x4"])
        W, H = cam["image_size"]
        # Centre + 8 points autour pour test plus robuste
        pt_h = np.array([candidate_center[0], candidate_center[1],
                          candidate_center[2], 1.0])
        pc = (pose @ pt_h)[:3]
        if pc[2] <= view_threshold:
            continue
        u = pc[0] * K[0, 0] / pc[2] + K[0, 2]
        v = pc[1] * K[1, 1] / pc[2] + K[1, 2]
        if 0 <= u < W and 0 <= v < H:
            n_visible += 1
    ratio = n_visible / n_total if n_total > 0 else 0.0
    return n_visible, n_total, ratio


# ───────────────────────────────────────────────────
# Fin filtres FP
# ───────────────────────────────────────────────────


# ───────────────────────────────────────────────────
# UTILS GEOMETRIE
# ───────────────────────────────────────────────────
def get_classe_from_det(all_det, det_idx):
    """
    Recupere la classe YOLOv8 du dechet det_idx.
    Retourne la classe la plus frequente parmi toutes les images.
    """
    class_votes = {}
    for img_data in all_det.values():
        dets = img_data.get("detections", [])
        if det_idx < len(dets):
            cl   = dets[det_idx].get("class", "dechet")
            conf = dets[det_idx].get("conf", 0)
            if cl not in class_votes or conf > class_votes[cl]:
                class_votes[cl] = conf
    if not class_votes:
        return "dechet", 0.0
    best = max(class_votes, key=class_votes.get)
    return best, class_votes[best]


def get_best_cluster(cands, cols, center):
    """
    Isole le cluster de points le plus proche du centre de vote.
    Utilise DBSCAN pour separer les dechets dans les scenes multi-dechets.
    """
    from sklearn.cluster import DBSCAN
    if len(cands) < DBSCAN_MIN_SAMPLES:
        return cands, cols
    db  = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(cands)
    lbs = db.labels_
    ul, ct = np.unique(lbs[lbs>=0], return_counts=True)
    if len(ul)==0: return cands, cols
    if len(ul)==1:
        return cands[lbs==ul[0]], cols[lbs==ul[0]]
    best=None; bd=np.inf
    for l,c in zip(ul,ct):
        if c < DBSCAN_MIN_SAMPLES: continue
        d = np.linalg.norm(cands[lbs==l].mean(0)-center)
        if d < bd: bd=d; best=l
    if best is None: best=ul[ct.argmax()]
    return cands[lbs==best], cols[lbs==best]


# ═══════════════════════════════════════════════════════════════════
# DEBRUITAGE ROBUSTE -- contribution scientifique principale du papier
# ═══════════════════════════════════════════════════════════════════
#
# Probleme : meme avec seeds natives (S0), une faible quantite de bruit
# residuel (pixels bord SAM, artefacts MASt3R) peut suffire a contaminer
# l expansion ulterieure (GSCE-C, vote 3D). Si on n'expanse rien, S0 est
# propre mais incomplet. Si on expanse a partir de seeds bruitees, le
# bruit s amplifie (comme on l observe sur B/D).
#
# Solution : pipeline de debruitage robuste applique AVANT toute
# expansion. Trois etapes sans aucune hypothese semantique (donc
# generalisable a toute scene -- canettes, bouteilles, mais aussi sacs,
# emballages, etc.) :
#
#   1. DBSCAN strict : extraire le composant connexe principal
#      (l objet est par definition une masse 3D coherente)
#   2. Filtre RANSAC plan-fond : detecter et eliminer un plan dominant
#      (sol/herbe/sable) parmi les seeds si present
#   3. Compactage spatial : eliminer les points isoles du centre de masse
#      au-dela de la dispersion principale (analogue 3D du whisker IQR)
#
# Contribution : pipeline de nettoyage qui transforme une segmentation
# bruitee en seeds objet pures, robustes a tout type de fond
# (herbe, sable, beton, eau). Permet ensuite l expansion controlee.
# ═══════════════════════════════════════════════════════════════════

def clean_seeds_robust(pts, cols=None,
                        dbscan_eps_factor=2.0,
                        min_inlier_ratio=0.30,
                        plane_inlier_thresh=0.01,
                        compact_z_thresh=2.5,
                        verbose=True):
    """
    Pipeline de debruitage robuste de seeds 3D (objet isole).

    Args:
        pts : (N,3) seeds candidates de l objet
        cols : (N,3) couleurs optionnelles
        dbscan_eps_factor : multiplicateur du knn-distance pour DBSCAN eps
        min_inlier_ratio : ratio min de points dans le plus gros cluster
                           pour considerer le filtrage comme reussi
        plane_inlier_thresh : tolerance RANSAC pour detection de plan-fond
                              (en metres, scenes smartphone metriques)
        compact_z_thresh : seuil de compactage (sigma vs centroide)
        verbose : log les etapes

    Returns:
        pts_clean, cols_clean, info_dict

    ── ABLATION via variables d environnement ──
    Pour faciliter les etudes d ablation sans modifier le code des
    approches, chaque etape peut etre desactivee independamment :
        CLEAN_DBSCAN=0    -> skip etape 1 (DBSCAN)
        CLEAN_PLANE=0     -> skip etape 2 (RANSAC plan-fond)
        CLEAN_COMPACT=0   -> skip etape 3 (compactage MAD)
    Sans variable definie ou =1, toutes les etapes sont actives.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    # ── Lecture des flags ablation ──
    use_dbscan  = os.environ.get("CLEAN_DBSCAN",  "1") == "1"
    use_plane   = os.environ.get("CLEAN_PLANE",   "1") == "1"
    use_compact = os.environ.get("CLEAN_COMPACT", "1") == "1"

    info = {"n_in": int(len(pts)), "plane_detected": False,
            "flags": {"dbscan":use_dbscan, "plane":use_plane,
                      "compact":use_compact}}
    if cols is None:
        cols = np.full((len(pts), 3), 0.5, dtype=np.float32)

    if len(pts) < 30:
        info.update({"n_dbscan": int(len(pts)),
                     "n_no_plane": int(len(pts)),
                     "n_compact": int(len(pts)),
                     "n_out": int(len(pts))})
        return pts, cols, info

    # ─────────────────────────────────────────────────────────────
    # ETAPE 1 -- DBSCAN strict : garder le composant principal
    # ─────────────────────────────────────────────────────────────
    if use_dbscan:
        k = min(8, len(pts) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(pts)
        dists, _ = nbrs.kneighbors(pts)
        knn_med = np.median(dists[:, -1])
        eps = max(knn_med * dbscan_eps_factor, 0.005)

        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(pts)
        valid = labels >= 0
        if valid.sum() == 0:
            info.update({"n_dbscan": 0, "n_no_plane": 0,
                         "n_compact": 0, "n_out": 0})
            return pts[:0], cols[:0], info

        vals, counts = np.unique(labels[valid], return_counts=True)
        main = vals[np.argmax(counts)]
        main_ratio = counts.max() / len(pts)
        keep1 = labels == main
        if main_ratio < min_inlier_ratio:
            if verbose:
                log(f"clean_seeds: composant principal {main_ratio:.0%} "
                    f"< {min_inlier_ratio:.0%}, DBSCAN ignore", 2)
            keep1 = np.ones(len(pts), dtype=bool)
        pts1 = pts[keep1]; cols1 = cols[keep1]
    else:
        pts1 = pts.copy(); cols1 = cols.copy()
        if verbose:
            log(f"clean_seeds: DBSCAN DESACTIVE (ablation)", 2)
    info["n_dbscan"] = int(len(pts1))

    if len(pts1) < 30:
        info.update({"n_no_plane": int(len(pts1)),
                     "n_compact": int(len(pts1)),
                     "n_out": int(len(pts1))})
        return pts1, cols1, info

    # ─────────────────────────────────────────────────────────────
    # ETAPE 2 -- RANSAC plan : detecter et eliminer plan-fond
    # ─────────────────────────────────────────────────────────────
    # FIX bug multi-objets : RANSAC a besoin d assez de seeds pour
    # distinguer "plan-fond" d "objet plat" (carton, canette ecrasee).
    # En dessous de 800 pts, n importe quoi "ressemble" a un plan, et
    # on amputait des petits objets entiers (cas Det 1 scene01,
    # Det 1/2 scene07). On skip RANSAC dans ce regime.
    MIN_PTS_FOR_PLANE = 800
    pts2, cols2 = pts1, cols1
    if use_plane and len(pts1) >= MIN_PTS_FOR_PLANE:
        try:
            import open3d as o3d
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(pts1)
            plane_model, inliers = pcd_tmp.segment_plane(
                distance_threshold=plane_inlier_thresh,
                ransac_n=3, num_iterations=200)
            n_inliers = len(inliers)
            ratio_plane = n_inliers / len(pts1)
            normal = np.array(plane_model[:3])
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            is_horizontal = abs(normal[2]) > 0.6
            # FIX bug "objet plat absorbe" : si > 70% des pts sont
            # sur le plan, c'est probablement l objet lui-meme
            # (carton, canette ecrasee). On refuse d eliminer.
            too_dominant = ratio_plane > 0.70
            do_filter = ((ratio_plane > 0.25 and is_horizontal) or
                         (ratio_plane > 0.50)) and not too_dominant
            if do_filter:
                mask = np.ones(len(pts1), dtype=bool)
                mask[inliers] = False
                pts2 = pts1[mask]; cols2 = cols1[mask]
                info["plane_detected"] = True
                info["plane_normal"] = normal.tolist()
                info["plane_ratio"] = float(ratio_plane)
                if verbose:
                    log(f"clean_seeds: plan-fond detecte "
                        f"({ratio_plane:.0%} points, "
                        f"|n_z|={abs(normal[2]):.2f}) -> elimine", 2)
            else:
                if verbose and ratio_plane > 0.15:
                    reason = "OBJET PLAT" if too_dominant else \
                             "vertical ou peu present"
                    log(f"clean_seeds: plan candidat {ratio_plane:.0%} "
                        f"non elimine ({reason})", 2)
        except Exception as e:
            if verbose:
                log(f"clean_seeds: RANSAC plan skip ({e})", 2)
    elif use_plane:
        if verbose:
            log(f"clean_seeds: PLAN RANSAC skip "
                f"({len(pts1)} pts < {MIN_PTS_FOR_PLANE}, petit objet)", 2)
    else:
        if verbose:
            log(f"clean_seeds: PLANE DESACTIVE (ablation)", 2)
    info["n_no_plane"] = int(len(pts2))

    if len(pts2) < 30:
        info.update({"n_compact": int(len(pts2)),
                     "n_out": int(len(pts2))})
        return pts2, cols2, info

    # ─────────────────────────────────────────────────────────────
    # ETAPE 3 -- Compactage spatial (whisker 3D)
    # ─────────────────────────────────────────────────────────────
    if use_compact:
        center = np.median(pts2, axis=0)
        d = np.linalg.norm(pts2 - center, axis=1)
        d_med = np.median(d)
        mad = np.median(np.abs(d - d_med)) + 1e-9
        keep3 = d < d_med + compact_z_thresh * 1.4826 * mad
        pts3 = pts2[keep3]; cols3 = cols2[keep3]
    else:
        pts3 = pts2; cols3 = cols2
        if verbose:
            log(f"clean_seeds: COMPACT DESACTIVE (ablation)", 2)
    info["n_compact"] = int(len(pts3))
    info["n_out"] = int(len(pts3))

    if verbose:
        log(f"clean_seeds: {info['n_in']} -> {info['n_dbscan']} "
            f"(DBSCAN) -> {info['n_no_plane']} (no-plane) -> "
            f"{info['n_out']} (compact)", 2)

    return pts3, cols3, info


def compute_obb_volume_svd(pts):
    """
    Calcul du volume OBB par decomposition SVD.

    Independant de Poisson : toujours calculable meme sur peu de points
    et meme si Poisson echoue. Donne une borne sup. du volume reel.

    NOTE ECHELLE : les coordonnees MASt3R sont a un facteur d echelle
    arbitraire pres. Les champs volume_m3 / volume_cm3 / volume_litres
    sont donc des UNITES RELATIVES (pas de calibration metrique) ;
    les comparaisons et le ratio Poisson/OBB restent valides.

    Ordre de grandeur (si l echelle etait metrique), canette 33 cl :
        OBB = (D x D x H) avec D ~ 6.6cm, H ~ 11.5cm -> ~500 cm3
        Volume reel = 330 cm3
        Ratio OBB/reel ~ 1.5 (sur-estimation du parallelepipede)

    Le ratio Volume_Poisson / Volume_OBB est lui-meme un score qualite :
        - ratio ~ 0.5-0.8 : Poisson detecte bien la forme creuse (bon)
        - ratio > 0.9     : Poisson capture aussi le fond (mauvais)

    Returns:
        dict {volume_m3, volume_cm3, volume_litres,
              dims_m, centroid_m, axes_R}
    """
    if len(pts) < 4:
        return {"volume_m3": 0.0, "volume_cm3": 0.0,
                "volume_litres": 0.0,
                "dims_m": [0, 0, 0],
                "centroid_m": [0, 0, 0],
                "axes_R": np.eye(3).tolist()}

    pts = np.asarray(pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    pts_c = pts - centroid

    # SVD pour trouver les 3 axes principaux de l objet
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    # V.T pour aligner sur les axes (Vt est deja le V transpose)
    pts_o = pts_c @ Vt.T

    dims = pts_o.max(axis=0) - pts_o.min(axis=0)  # L,l,h
    vol_m3 = float(dims[0] * dims[1] * dims[2])
    return {
        "volume_m3": vol_m3,
        "volume_cm3": vol_m3 * 1e6,
        "volume_litres": vol_m3 * 1000,
        "dims_m": dims.tolist(),
        "centroid_m": centroid.tolist(),
        "axes_R": Vt.tolist(),
    }


def add_svd_volume_to_met(met, pts):
    """
    Enrichit le dict met avec :
    - volume_svd_cm3 : volume OBB par SVD (robuste ; unites relatives)
    - volume_poisson_cm3 : alias du volume_cm3 existant (Poisson, precis)
    - volume_ratio : volume_poisson / volume_svd (score qualite)
       - ~ 0.4-0.7 : forme creuse correctement detectee (canette, bouteille)
       - > 0.85   : nuage probablement bruite (fond inclus)
    - dims_svd_m : dimensions [L,l,h] dans le repere SVD
    """
    svd = compute_obb_volume_svd(pts)
    v_svd_cm3 = svd["volume_cm3"]
    v_poisson_cm3 = met.get("volume_cm3", 0)
    ratio = v_poisson_cm3 / v_svd_cm3 if v_svd_cm3 > 1e-9 else 0.0
    met.update({
        "volume_svd_cm3":     float(v_svd_cm3),
        "volume_svd_litres":  float(svd["volume_litres"]),
        "volume_poisson_cm3": float(v_poisson_cm3),
        "volume_ratio":       float(ratio),
        "dims_svd_m":         svd["dims_m"],
        "centroid_svd_m":     svd["centroid_m"],
    })
    # Enrichir avec metriques intrinseques
    met = add_intrinsic_metrics_to_met(met, pts)
    return met


def add_intrinsic_metrics_to_met(met, pts):
    """
    Calcule des metriques intrinseques de qualite d un nuage segmente.

    Aucune verite terrain requise. Ces metriques sont utiles pour :
    - Comparer differentes configurations d ablation
    - Detecter du bruit (sphericite faible, densite faible, asymetrie)
    - Mesurer la compacite d un cluster d objet

    Metriques :
      - n_pts                : nombre de points
      - density_pts_cm3      : densite (pts par cm3 de volume SVD)
      - sphericity           : 3*lambda3 / (lambda1+lambda2+lambda3)
                                (0=plat/ligne, 1=sphere parfaite)
      - linearity            : (lambda1-lambda2) / lambda1 (1=ligne, 0=plan/sphere)
      - planarity            : (lambda2-lambda3) / lambda1 (1=plan, 0=ligne/sphere)
      - aspect_ratio         : L / l (allongement OBB)
      - flatness             : h / l (aplatissement OBB)
      - bbox_fill            : V_Poisson / V_OBB (compactness, 1=cube plein)
      - centroid_offset_m    : distance centroid moyen vs median
                                (asymetrie : haut = outliers)
      - mean_nn_distance_m   : distance moyenne aux 5 plus proches voisins
                                (= "scale" intrinseque du nuage)
      - convex_hull_volume_cm3 : volume de l enveloppe convexe (vs OBB)
      - convex_hull_ratio    : V_hull / V_OBB (1 = nuage convexe parfait)
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return met

    # PCA pour eigenvalues (linearity / planarity / sphericity)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    cov = (pts_c.T @ pts_c) / max(1, len(pts_c) - 1)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]  # decroissant
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    sum_l = l1 + l2 + l3 + 1e-12
    sphericity = 3 * l3 / sum_l
    linearity  = (l1 - l2) / (l1 + 1e-12)
    planarity  = (l2 - l3) / (l1 + 1e-12)

    # OBB aspect ratios (depuis dims_svd_m deja calcule)
    dims_svd = sorted(met.get("dims_svd_m", [1, 1, 1]), reverse=True)
    L, l, h = float(dims_svd[0]), float(dims_svd[1]), float(dims_svd[2])
    aspect_ratio = L / (l + 1e-9)
    flatness = h / (l + 1e-9)

    # Densite
    v_svd_cm3 = met.get("volume_svd_cm3", 0)
    density = n / v_svd_cm3 if v_svd_cm3 > 1e-9 else 0.0

    # Bounding box fill
    v_poisson = met.get("volume_poisson_cm3", 0)
    bbox_fill = v_poisson / v_svd_cm3 if v_svd_cm3 > 1e-9 else 0.0

    # Centroid offset (asymetrie)
    centroid_med = np.median(pts, axis=0)
    offset = float(np.linalg.norm(centroid - centroid_med))

    # Mean nearest neighbor distance
    try:
        from scipy.spatial import cKDTree
        sample = pts[:min(500, n)]
        tree = cKDTree(pts)
        dists, _ = tree.query(sample, k=min(6, n))
        nn_dist = float(np.mean(dists[:, 1:]))  # exclure self
    except Exception:
        nn_dist = 0.0

    # Convex hull
    hull_vol_cm3 = 0.0
    hull_ratio = 0.0
    try:
        hull = ConvexHull(pts, qhull_options="QJ")
        hull_vol_cm3 = float(hull.volume * 1e6)
        hull_ratio = hull_vol_cm3 / v_svd_cm3 if v_svd_cm3 > 1e-9 else 0.0
    except Exception:
        pass

    met.update({
        "metrics_n_pts":              int(n),
        "metrics_density_pts_cm3":    float(density),
        "metrics_sphericity":         float(sphericity),
        "metrics_linearity":          float(linearity),
        "metrics_planarity":          float(planarity),
        "metrics_aspect_ratio":       float(aspect_ratio),
        "metrics_flatness":           float(flatness),
        "metrics_bbox_fill":          float(bbox_fill),
        "metrics_centroid_offset_m":  float(offset),
        "metrics_mean_nn_dist_m":     float(nn_dist),
        "metrics_hull_volume_cm3":    float(hull_vol_cm3),
        "metrics_hull_ratio":         float(hull_ratio),
        "metrics_eigenvalues":        [l1, l2, l3],
    })
    return met


def filter_seed_outliers(pts, cols, k=10, angle_thresh=60.0):
    """
    Retire les graines GSCE geometriquement incoherentes.
    Sans filtre couleur : generalisable a tout fond.

    3 criteres :
    1. Isolement spatial : voisins trop loin -> point isole
    2. Incoherence normales : angle > 60 deg -> bord ou fond
    3. Outlier positionnel : z-score > 2.5 sigma -> bruit isole
    """
    import open3d as o3d
    from sklearn.neighbors import KDTree
    if len(pts) < k+1: return pts, cols
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    sample = pts[:min(500,len(pts))]
    avg = np.mean(KDTree(sample).query(sample,k=2)[0][:,1])
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=avg*8, max_nn=20))
    pcd.orient_normals_consistent_tangent_plane(k=min(15,len(pts)-1))
    normals = np.asarray(pcd.normals)
    tree = KDTree(pts)
    dists,nn_idx = tree.query(pts, k=min(k+1,len(pts)))
    nn_dists = dists[:,1:]; nn_idx = nn_idx[:,1:]
    isole      = nn_dists[:,0] / (nn_dists.mean()+1e-6) > 3.0
    nn_normals = normals[nn_idx].mean(axis=1)
    dot        = np.clip(np.abs(np.sum(normals*nn_normals,axis=1)),0,1)
    incoherent = np.degrees(np.arccos(dot)) > angle_thresh
    dists_c    = np.linalg.norm(pts-pts.mean(axis=0),axis=1)
    pos_out    = (dists_c-dists_c.mean())/(dists_c.std()+1e-6) > 2.5
    keep = ~(isole | incoherent | pos_out)
    log(f"Filtre graines : {len(pts):,} -> {keep.sum():,} "
        f"({(~keep).sum():,} retires)", 2)
    if keep.sum() < 30: return pts, cols
    return pts[keep], cols[keep]


def make_label_3d(text, position, size=0.008, color=(255,255,255)):
    """
    Genere des points 3D formant un texte lisible dans CloudCompare.
    Place au coin superieur droit de l OBB.
    Chaque caractere est dessine par un bitmap 5x7 pixels.
    """
    CHARS = {
        '0':["01110","10001","10001","10101","10001","10001","01110"],
        '1':["00100","01100","00100","00100","00100","00100","01110"],
        '2':["01110","10001","00001","00110","01000","10000","11111"],
        '3':["11110","00001","00001","01110","00001","00001","11110"],
        '4':["00010","00110","01010","10010","11111","00010","00010"],
        '5':["11111","10000","10000","11110","00001","00001","11110"],
        '6':["01110","10000","10000","11110","10001","10001","01110"],
        '7':["11111","00001","00010","00100","01000","01000","01000"],
        '8':["01110","10001","10001","01110","10001","10001","01110"],
        '9':["01110","10001","10001","01111","00001","00001","01110"],
        '.':["00000","00000","00000","00000","00000","01100","01100"],
        ' ':["00000","00000","00000","00000","00000","00000","00000"],
        '-':["00000","00000","00000","11111","00000","00000","00000"],
        '=':["00000","11111","00000","11111","00000","00000","00000"],
        '/':["00001","00010","00100","01000","10000","00000","00000"],
        'A':["01110","10001","10001","11111","10001","10001","10001"],
        'B':["11110","10001","10001","11110","10001","10001","11110"],
        'C':["01110","10001","10000","10000","10000","10001","01110"],
        'D':["11100","10010","10001","10001","10001","10010","11100"],
        'E':["11111","10000","10000","11110","10000","10000","11111"],
        'G':["01110","10001","10000","10111","10001","10001","01111"],
        'H':["10001","10001","10001","11111","10001","10001","10001"],
        'I':["01110","00100","00100","00100","00100","00100","01110"],
        'L':["10000","10000","10000","10000","10000","10000","11111"],
        'M':["10001","11011","10101","10001","10001","10001","10001"],
        'N':["10001","11001","10101","10011","10001","10001","10001"],
        'O':["01110","10001","10001","10001","10001","10001","01110"],
        'P':["11110","10001","10001","11110","10000","10000","10000"],
        'R':["11110","10001","10001","11110","10100","10010","10001"],
        'S':["01111","10000","10000","01110","00001","00001","11110"],
        'T':["11111","00100","00100","00100","00100","00100","00100"],
        'U':["10001","10001","10001","10001","10001","10001","01110"],
        'V':["10001","10001","10001","10001","01010","01010","00100"],
        'X':["10001","01010","00100","00100","00100","01010","10001"],
        'Z':["11111","00001","00010","00100","01000","10000","11111"],
    }
    DEFAULT = ["11111","10001","10001","10001","10001","10001","11111"]
    pts=[]; col_list=[]
    x0,y0,z0 = position
    char_w = size*6
    for ci,ch in enumerate(text.upper()):
        bitmap = CHARS.get(ch,DEFAULT)
        cx = x0 + ci*char_w
        for row,line in enumerate(bitmap):
            for c2,px in enumerate(line):
                if px=='1':
                    pts.append([cx+c2*size, y0, z0+(6-row)*size])
                    col_list.append(list(color))
    if not pts: return np.zeros((0,3)), []
    return np.array(pts,dtype=np.float32), col_list


def build_obb_with_label(pts_dechet, classe, conf, volume_cm3,
                          obb_dims, obb_center, obb_rotation,
                          col, coords_scene):
    """
    Construit l OBB 3D avec label texte lisible au coin superieur droit.

    Le label contient :
    - La classe du dechet
    - La confiance YOLOv8
    - Les coordonnees dans le repere scene (x, y, z en metres)
    - Le volume en cm3

    Format label : CLASSE(CONF%) x=X.XX y=Y.YY z=Z.ZZ V=VVVcm3
    """
    R=np.array(obb_rotation); c=np.array(obb_center); dims=np.array(obb_dims)
    a=R[:,0]*dims[0]/2; b=R[:,1]*dims[1]/2; cc=R[:,2]*dims[2]/2
    corners=np.array([c-a-b-cc,c+a-b-cc,c+a+b-cc,c-a+b-cc,
                      c-a-b+cc,c+a-b+cc,c+a+b+cc,c-a+b+cc])
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7)]
    edge_pts=np.array([corners[i]*(1-t)+corners[j]*t
                       for i,j in edges for t in np.linspace(0,1,300)])

    # Label au coin superieur droit (corner 6)
    label_pos = corners[6].copy(); label_pos[2] += dims[2]*0.15
    xs = coords_scene["x_m"]; ys = coords_scene["y_m"]; zs = coords_scene["z_m"]
    conf_pct = int(conf*100)
    # Format lisible : METAL(87%) x=0.01 y=-0.05 z=0.04 V=389cm3
    label_txt = (f"{classe.upper()[:8]}({conf_pct}%) "
                 f"x={xs:.2f} y={ys:.2f} z={zs:.2f} "
                 f"V={volume_cm3:.0f}cm3")
    label_size = np.clip(min(dims)*0.15, 0.003, 0.015)
    lpts,lcols = make_label_3d(label_txt, label_pos,
                                size=label_size, color=(255,255,255))
    all_pts  = np.vstack([edge_pts,lpts]) if len(lpts)>0 else edge_pts
    all_cols = (np.tile(col,(len(edge_pts),1)).tolist()
                + [[255,255,255]]*len(lpts))
    return all_pts, all_cols, corners


def reconstruct_fond(points, colors, dechet_masks_list, out_dir):
    """
    Reconstruit le fond de scene en excluant les zones de dechets.
    Utilise Poisson depth=6 (basse qualite = rapide).
    Sert de contexte visuel dans CloudCompare.
    """
    import open3d as o3d
    log("[FOND] Maillage fond...", 1)
    fond_mask = np.ones(len(points),dtype=bool)
    for mask in dechet_masks_list: fond_mask &= ~mask
    fp = points[fond_mask][::max(1,fond_mask.sum()//50000)]
    fc = colors[fond_mask][::max(1,fond_mask.sum()//50000)]
    if len(fp)<1000: log("Pas assez de points fond",2); return
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(fp)
    pcd.colors=o3d.utility.Vector3dVector(np.clip(fc,0,1).astype(np.float64))
    pcd,_=pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=3.0)
    avg=np.mean(pcd.compute_nearest_neighbor_distance())
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg*5,max_nn=20))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0,0,10]))
    mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,depth=6,width=0,scale=1.1,linear_fit=False)
    dens=np.asarray(dens)
    mesh.remove_vertices_by_mask(dens<np.percentile(dens,20))
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(Path(out_dir)/"fond_mesh.ply"),mesh)
    log(f"OK fond_mesh.ply : {len(mesh.triangles):,} triangles",2)


def make_dechet_mask(points, pts_d):
    """
    Cree un masque booleen indiquant quels points du nuage coarse
    appartiennent au dechet. Utilise pour exclure les dechets du fond.
    """
    from sklearn.neighbors import KDTree
    tree=KDTree(points); _,idx=tree.query(pts_d,k=1)
    mask=np.zeros(len(points),dtype=bool); mask[idx.flatten()]=True
    return mask


# ───────────────────────────────────────────────────
# GSCE-C
# ───────────────────────────────────────────────────
def compute_confidence_scores(pts_cands, pts_coarse, normals, depths,
                               center, nn_idx, angle_r, max_dist):
    """
    Calcule le score de confiance geometrique pour chaque point candidat.

    4 scores combines en score global :

    Score 1 -- Coherence normales (poids 35%) :
        Mesure si la normale du candidat est alignee avec ses voisins.
        Score=1 si parfaitement aligne (meme surface continue).
        Score=0 si perpendiculaire (bord ou discontinuite).

    Score 2 -- Continuite profondeur (poids 30%) :
        Mesure si le saut de profondeur est faible par rapport
        a la variabilite locale. Score=1 si continuite parfaite.
        Score=0 si saut brusque (bord du dechet).

    Score 3 -- Proximite centre graines (poids 20%) :
        Les points proches du centre du dechet sont plus fiables.
        Score=1 au centre, Score=0 a max_dist.

    Score 4 -- Densite locale (poids 15%) :
        Les zones denses (bien reconstruites par MASt3R) sont
        plus fiables. Score=1 dans les zones denses.
    """
    nn_n   = normals[nn_idx].mean(axis=1)
    cn     = normals[nn_idx[:,0]]
    dot    = np.clip(np.abs(np.sum(cn*nn_n,axis=1)),0,1)
    s_norm = np.clip(1.0-np.arccos(dot)/angle_r,0,1)

    cd     = depths[nn_idx[:,0]]; nd=depths[nn_idx]
    s_prof = np.clip(1.0-np.abs(cd-nd.mean(1))/(GSCE_DEPTH*(nd.std(1)+1e-6)),0,1)

    dc     = np.linalg.norm(pts_cands-center,axis=1)
    s_dist = np.clip(1.0-dc/max_dist,0,1)

    nn_d   = np.linalg.norm(pts_coarse[nn_idx]-pts_cands[:,np.newaxis,:],axis=2)
    avg_d  = nn_d.mean(axis=1)
    s_dens = np.clip(1.0-avg_d/(3*avg_d.mean()+1e-6),0,1)

    return W_NORMAL*s_norm + W_PROFONDEUR*s_prof + W_DISTANCE*s_dist + W_DENSITE*s_dens


def gsce_c(pts_seeds, pts_coarse, cols_coarse, verbose=True):
    """
    GSCE-C : Geometric Semantic Cloud Expansion with Confidence
    Author, 2026

    Expansion geometrique iterative du nuage partiel labellise.

    Ameliorations v2 vs v1 :
    - Convergence par ratio (n_new/n_total < 0.1%) au lieu de seuil fixe
    - Max iterations augmente a 50 (etait 15)
    - Plus de points recuperes sur les dechets aux bords complexes

    Etapes :
    1. Estimer normales + profondeurs sur le nuage coarse complet
    2. Mapper les graines dans le coarse via KDTree
    3. Boucle d expansion :
       a. Trouver les voisins des points labellises dans le rayon
       b. Appliquer 4 filtres geometriques binaires (F1 F2 F3 F4)
       c. Calculer le score de confiance pour les candidats valides
       d. Accepter si score >= GSCE_CONF_THR
       e. Arreter si ratio < GSCE_CONV_RATIO ou iterations > GSCE_MAX_ITER
    4. Retourner nuage etendu + scores de confiance
    """
    import open3d as o3d
    from sklearn.neighbors import KDTree

    def lg(m):
        if verbose: log(f"[GSCE-C] {m}",2)

    lg(f"Debut -- {len(pts_seeds):,} graines | {len(pts_coarse):,} pts coarse")

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts_coarse)
    sample=pts_coarse[:min(2000,len(pts_coarse))]
    avg_g=np.mean(KDTree(sample).query(sample,k=2)[0][:,1])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg_g*5,max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    normals=np.asarray(pcd.normals)
    centroid=pts_coarse.mean(axis=0)
    depths=np.linalg.norm(pts_coarse-centroid,axis=1)

    tree=KDTree(pts_coarse)
    labeled=set()
    _,idx_s=tree.query(pts_seeds,k=1)
    for idx in idx_s.flatten(): labeled.add(idx)
    conf_map={idx:1.0 for idx in labeled}

    s2=pts_coarse[list(labeled)[:min(1000,len(labeled))]]
    radius=np.mean(tree.query(s2,k=2)[0][:,1])*GSCE_EXPANSION
    center=pts_seeds.mean(axis=0)
    max_dist=np.linalg.norm(pts_seeds.max(0)-pts_seeds.min(0))*1.2
    angle_r=np.deg2rad(GSCE_ANGLE)
    in_range=np.linalg.norm(pts_coarse-center,axis=1)<max_dist

    # PRIORITE 2 : critere d arret par volume.
    # Un objet ne grossit pas indefiniment. Quand l expansion fait
    # exploser l OBB au-dela de GSCE_MAX_VOL_FACTOR x le volume des
    # seeds initiales, c'est qu'elle deborde sur le fond -> on stoppe.
    # Override : GSCE_MAX_VOL_FACTOR (defaut 3.0), 0 = desactive.
    GSCE_MAX_VOL_FACTOR = float(os.environ.get("GSCE_MAX_VOL_FACTOR", "3.0"))
    def _obb_vol(pts):
        if len(pts) < 4: return 0.0
        c = pts.mean(axis=0); pc = pts - c
        try:
            _,_,Vt = np.linalg.svd(pc, full_matrices=False)
            po = pc @ Vt.T
            dims = po.max(0) - po.min(0)
            return float(dims[0]*dims[1]*dims[2])
        except Exception:
            return 0.0
    vol_seeds = _obb_vol(pts_seeds)
    vol_limit = vol_seeds * GSCE_MAX_VOL_FACTOR if GSCE_MAX_VOL_FACTOR > 0 else np.inf
    if vol_seeds > 0:
        lg(f"Volume seeds={vol_seeds*1e6:.0f}cm3 | "
           f"limite expansion={vol_limit*1e6:.0f}cm3 "
           f"(x{GSCE_MAX_VOL_FACTOR})")

    lg(f"Rayon={radius:.5f}m | dist_max={max_dist:.4f}m")

    n_iter=0
    for iteration in range(GSCE_MAX_ITER):
        cur=pts_coarse[list(labeled)]; n_b=len(labeled)
        nbrs=tree.query_radius(cur,r=radius)
        cands=set()
        for nb in nbrs:
            for idx in nb:
                if idx not in labeled: cands.add(idx)
        if not cands: break

        cands=np.array(list(cands))
        f1=in_range[cands]
        _,nn=tree.query(pts_coarse[cands],k=GSCE_K_NEIGHBORS)
        nn_n=normals[nn].mean(axis=1); cn=normals[nn[:,0]]
        dot=np.clip(np.abs(np.sum(cn*nn_n,axis=1)),0,1)
        f2=np.arccos(dot)<angle_r
        cd=depths[nn[:,0]]; nd=depths[nn]
        f3=np.abs(cd-nd.mean(1))<GSCE_DEPTH*(nd.std(1)+1e-6)

        valid=cands[f1&f2&f3]
        if not len(valid): n_iter=iteration+1; break

        scores=compute_confidence_scores(
            pts_coarse[valid],pts_coarse,normals,depths,
            center,nn[f1&f2&f3],angle_r,max_dist)

        acc=valid[scores>=GSCE_CONF_THR]; acc_sc=scores[scores>=GSCE_CONF_THR]
        for i,idx in enumerate(acc): labeled.add(idx); conf_map[idx]=float(acc_sc[i])

        n_new=len(labeled)-n_b; n_iter=iteration+1
        # Convergence par ratio : n_new/n_total < 0.1%
        ratio=n_new/len(labeled) if len(labeled)>0 else 0
        sc_str=f"score_moy={acc_sc.mean():.3f}" if len(acc_sc) else ""
        lg(f"Iter {n_iter:2d} : +{n_new:,} ({ratio*100:.2f}%) | "
           f"{sc_str} | total={len(labeled):,}")

        # PRIORITE 2 : arret si l OBB depasse la limite volumique.
        # On verifie tous les 3 iters (le calcul SVD a un cout).
        if vol_seeds > 0 and (n_iter % 3 == 0):
            vol_cur = _obb_vol(pts_coarse[list(labeled)])
            if vol_cur > vol_limit:
                lg(f"STOP volume : OBB={vol_cur*1e6:.0f}cm3 > "
                   f"limite={vol_limit*1e6:.0f}cm3 (expansion deborde)")
                break

        if ratio < GSCE_CONV_RATIO:
            lg(f"Convergence (ratio={ratio*100:.3f}% < {GSCE_CONV_RATIO*100}%)")
            break

    idx_f=np.array(list(labeled))
    pts_out=pts_coarse[idx_f]; cols_out=cols_coarse[idx_f]
    sc_out=np.array([conf_map.get(i,1.0) for i in idx_f])
    lg(f"OK {len(pts_seeds):,} -> {len(pts_out):,} pts | "
       f"score_moy={sc_out.mean():.3f} | {n_iter} iterations")
    return pts_out,cols_out,sc_out,n_iter


# ───────────────────────────────────────────────────
# POISSON
# ───────────────────────────────────────────────────
def poisson_weighted(pts, cols, scores, pts_c, cols_c):
    """
    Reconstruction Poisson ponderee par les scores GSCE-C.

    Principe :
    Les normales de chaque point sont multipliees par son score.
    Points fiables (score=1) -> normales amplifiees -> plus d influence.
    Points incertains (score=0.1) -> normales attenuees -> moins d influence.
    Resultat : surface plus precise sur les zones fiables.

    Depth Poisson adaptatif selon densite :
    - < 500 points  -> depth=6 (peu de details)
    - < 5000 points -> depth=8 (details moyens)
    - >= 5000 pts   -> depth=9 (haute resolution)
    """
    import open3d as o3d
    from sklearn.neighbors import KDTree
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd.colors=o3d.utility.Vector3dVector(np.clip(cols,0,1).astype(np.float64))
    pcd,_=pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    avg=np.mean(pcd.compute_nearest_neighbor_distance())
    pcd,_=pcd.remove_radius_outlier(nb_points=5,radius=avg*5)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg*5,max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    _,ic=KDTree(pts).query(np.asarray(pcd.points),k=1)
    sc=scores[ic.flatten()]
    nw=np.asarray(pcd.normals)*sc[:,np.newaxis]
    norms=np.linalg.norm(nw,axis=1,keepdims=True)
    pcd.normals=o3d.utility.Vector3dVector(nw/np.where(norms<1e-8,1,norms))
    n=len(pcd.points); d=6 if n<500 else 8 if n<5000 else 9
    mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,depth=d,width=0,scale=POISSON_SCALE,linear_fit=False)
    dens=np.asarray(dens)
    mesh.remove_vertices_by_mask(dens<np.percentile(dens,10))
    obb_o=pcd.get_oriented_bounding_box(); obb_o.scale(1.15,obb_o.center)
    mesh=mesh.crop(obb_o)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    tc,cn,_=mesh.cluster_connected_triangles()
    tc=np.asarray(tc); cn=np.asarray(cn)
    if len(cn)>1:
        m=cn.argmax(); mesh.remove_triangles_by_mask(tc!=m)
        mesh.remove_unreferenced_vertices()
    mesh=mesh.filter_smooth_laplacian(number_of_iterations=2)
    mesh.compute_vertex_normals()
    if len(pts_c)>0:
        _,ir=KDTree(pts_c).query(np.asarray(mesh.vertices),k=1)
        mesh.vertex_colors=o3d.utility.Vector3dVector(cols_c[ir.flatten()])
    pcd_m=o3d.geometry.PointCloud(); pcd_m.points=mesh.vertices
    obb_m=pcd_m.get_oriented_bounding_box()
    dims=np.sort(np.array(obb_m.extent))[::-1]
    is_wt=mesh.is_watertight()
    vol=abs(mesh.get_volume()) if is_wt else float(dims[0]*dims[1]*dims[2]*0.5)
    return mesh,{"obb_dims":dims.tolist(),"obb_center":np.array(obb_m.center).tolist(),
                 "obb_rotation":np.array(obb_m.R).tolist(),"volume_cm3":vol*1e6,
                 "surface_cm2":mesh.get_surface_area()*1e4,
                 "n_triangles":len(mesh.triangles),"watertight":is_wt,
                 "score_moyen":float(sc.mean()),"poisson_depth":d}


def poisson_standard(pts, cols, pts_c, cols_c):
    """
    Reconstruction Poisson standard sans ponderation.
    Utilisee pour Approche A (vote SAM seul) et Approche C (frustum).
    """
    import open3d as o3d
    from sklearn.neighbors import KDTree
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd.colors=o3d.utility.Vector3dVector(np.clip(cols,0,1).astype(np.float64))
    pcd,_=pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    avg=np.mean(pcd.compute_nearest_neighbor_distance())
    pcd,_=pcd.remove_radius_outlier(nb_points=5,radius=avg*5)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg*5,max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    n=len(pcd.points); d=6 if n<500 else 8 if n<5000 else 9
    mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,depth=d,width=0,scale=POISSON_SCALE,linear_fit=False)
    dens=np.asarray(dens)
    mesh.remove_vertices_by_mask(dens<np.percentile(dens,10))
    obb_o=pcd.get_oriented_bounding_box(); obb_o.scale(1.15,obb_o.center)
    mesh=mesh.crop(obb_o)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    tc,cn,_=mesh.cluster_connected_triangles()
    tc=np.asarray(tc); cn=np.asarray(cn)
    if len(cn)>1:
        m=cn.argmax(); mesh.remove_triangles_by_mask(tc!=m)
        mesh.remove_unreferenced_vertices()
    mesh=mesh.filter_smooth_laplacian(number_of_iterations=2)
    mesh.compute_vertex_normals()
    if len(pts_c)>0:
        _,ir=KDTree(pts_c).query(np.asarray(mesh.vertices),k=1)
        mesh.vertex_colors=o3d.utility.Vector3dVector(cols_c[ir.flatten()])
    pcd_m=o3d.geometry.PointCloud(); pcd_m.points=mesh.vertices
    obb_m=pcd_m.get_oriented_bounding_box()
    dims=np.sort(np.array(obb_m.extent))[::-1]
    is_wt=mesh.is_watertight()
    vol=abs(mesh.get_volume()) if is_wt else float(dims[0]*dims[1]*dims[2]*0.5)
    return mesh,{"obb_dims":dims.tolist(),"obb_center":np.array(obb_m.center).tolist(),
                 "obb_rotation":np.array(obb_m.R).tolist(),"volume_cm3":vol*1e6,
                 "surface_cm2":mesh.get_surface_area()*1e4,
                 "n_triangles":len(mesh.triangles),"watertight":is_wt,
                 "poisson_depth":d}


# ───────────────────────────────────────────────────
# ETAPE 1 : DETECTION YOLOV8
# ───────────────────────────────────────────────────
def run_detection(images_dir, out_dir):
    """
    Detecte les dechets dans toutes les images avec YOLOv8.
    Sauvegarde les boites 2D, classes et confiances dans detections.json.
    """
    log("[1/3] Detection YOLOv8...", 1)
    from ultralytics import YOLO
    yolo=YOLO(YOLO_MODEL)
    all_files=sorted([str(f) for f in Path(images_dir).iterdir()
                      if f.suffix in IMG_EXT])
    n_total=len(all_files)
    all_files=subsample_images(all_files)
    if len(all_files) < n_total:
        log(f"{n_total} images -> sous-echantillonnees a {len(all_files)} (MAX_IMAGES={MAX_IMAGES})",2)
    else:
        log(f"{len(all_files)} images",2)
    all_det={}; n_det=0
    for img_path in all_files:
        img_name=Path(img_path).name
        img=cv2.imread(img_path)
        if img is None: continue
        H,W=img.shape[:2]
        try: results=yolo(img_path,conf=0.25,verbose=False)[0]
        except Exception as e:
            all_det[img_name]={"image_size":[W,H],"detections":[]}; continue
        dets=[]
        for box in results.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist()
            dets.append({"bbox_2d":[int(x1),int(y1),int(x2),int(y2)],
                         "class":yolo.names[int(box.cls)],
                         "conf":float(box.conf)})
        if dets: n_det+=1
        all_det[img_name]={"image_size":[W,H],"detections":dets}
    with open(f"{out_dir}/detections.json","w") as f:
        json.dump(all_det,f,indent=2)
    max_det=max((len(v["detections"]) for v in all_det.values()),default=0)
    log(f"OK {n_det}/{len(all_files)} images | max {max_det} dechets/image",2)
    return all_det, all_files


# ───────────────────────────────────────────────────
# ETAPE 2 : SEGMENTATION SAM
# ───────────────────────────────────────────────────
def run_sam(all_files, all_det, out_dir):
    """
    Segmente chaque dechet detecte avec SAM (Segment Anything Model).
    Produit un masque PNG binaire par dechet par image.
    Sauvegarde les chemins des masques dans detections_seg.json.
    """
    log("[2/3] Segmentation SAM...", 1)
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device : {DEVICE}",2)
    sam=sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(DEVICE)
    pred=SamPredictor(sam)
    masks_dir=f"{out_dir}/masks"; os.makedirs(masks_dir,exist_ok=True)
    all_det_sam={}; n_masked=0
    for img_path in all_files:
        img_name=Path(img_path).name; stem=Path(img_path).stem
        entry=all_det.get(img_name,{}); dets=entry.get("detections",[])
        W,H=entry.get("image_size",[0,0])
        if not dets:
            all_det_sam[img_name]={"image_size":[W,H],"detections":[]}; continue
        img_bgr=cv2.imread(img_path)
        if img_bgr is None:
            all_det_sam[img_name]={"image_size":[W,H],"detections":[]}; continue
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        try: pred.set_image(img_rgb)
        except Exception as e:
            all_det_sam[img_name]={"image_size":[W,H],"detections":[]}; continue
        dets_out=[]
        for det_idx,det in enumerate(dets):
            x1,y1,x2,y2=det["bbox_2d"]
            try:
                masks,scores,_=pred.predict(box=np.array([x1,y1,x2,y2]),
                                            multimask_output=True)
                best=masks[np.argmax(scores)].astype(np.uint8)
                mask_path=f"{masks_dir}/{stem}_mask_{det_idx}.png"
                cv2.imwrite(mask_path,best*255)
                dets_out.append({**det,"mask_path":mask_path})
            except Exception as e:
                dets_out.append(det)
        all_det_sam[img_name]={"image_size":[W,H],"detections":dets_out}
        n_masked+=1
    with open(f"{out_dir}/detections_seg.json","w") as f:
        json.dump(all_det_sam,f,indent=2)
    del sam,pred; clear_gpu()
    log(f"OK {n_masked} images segmentees",2)
    return all_det_sam


# ───────────────────────────────────────────────────
# ETAPE 3 : RECONSTRUCTION MAST3R
# ───────────────────────────────────────────────────
def run_mast3r(images_dir, all_files, out_dir):
    """
    Reconstruit le nuage de points 3D metrique avec MASt3R.

    Parametres anti-superposition :
    - swin-8 : chaque image comparee avec 8 voisines (vs 3 standard)
    - Loop closure 12 paires : premiere<->derniere + quart<->trois-quarts
    - 200 iterations : convergence plus profonde
    - Seuil confiance absolu 1.5 : evite le bug quantile MASt3R
    - Nettoyage outliers 3-sigma : retire les points aberrants
    """
    log("[3/3] Reconstruction MASt3R...", 1)
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy
    from PIL import ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device : {DEVICE}",2)
    valid=[]
    for p in all_files:
        try: Image.open(p).verify(); valid.append(p)
        except: pass
    model=AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        cache_dir=HF_CACHE).to(DEVICE).eval()
    images_dict=load_images(valid,size=512,verbose=False)
    n=len(images_dict)
    pairs=make_pairs(images_dict,scene_graph=SCENE_GRAPH,
                     prefilter=None,symmetrize=True)
    for i in range(LOOP_CLOSURE):
        # Loop closure debut<->fin uniquement.
        # Les paires (i, n//2+i) entre vues tres eloignees produisent des
        # appariements faibles qui injectent du bruit dans l'alignement global.
        pairs.append((images_dict[i%n],images_dict[(n-1-i)%n]))
        pairs.append((images_dict[(n-1-i)%n],images_dict[i%n]))
    pairs_u=list({(id(a),id(b)):(a,b) for a,b in pairs}.values())
    log(f"Paires : {len(pairs_u)}",2)
    with torch.no_grad():
        output=inference(pairs_u,model,DEVICE,batch_size=1,verbose=False)
    del model; clear_gpu(); time.sleep(3)
    scene_3d=global_aligner(output,device=DEVICE,
        mode=GlobalAlignerMode.PointCloudOptimizer)
    loss=scene_3d.compute_global_alignment(
        init="mst",niter=MAST3R_ITER,schedule="cosine",lr=0.01)
    log(f"Loss : {loss:.6f}"+(" WARNING superposition?" if loss>0.025 else " OK"),2)
    poses=to_numpy(scene_3d.get_im_poses())
    focals=to_numpy(scene_3d.get_focals())
    pts3d=scene_3d.get_pts3d(); conf=scene_3d.get_conf()
    masks=scene_3d.get_masks()   # <-- AJOUT : masque geometrique multi-vues fiable
    cam_data={}; all_pts=[]; all_cols=[]
    # Pour Approche S0 : on garde les pts3d natifs par image (H,W,3)
    # + masque de validite par image. Permet l association directe pixel<->pts3d
    # sans projection inverse (indexation native : la plus propre sur les bords).
    native_per_image={}
    for i,img_path in enumerate(valid):
        pts_np=to_numpy(pts3d[i]); conf_np=to_numpy(conf[i])
        H,W=pts_np.shape[:2]
        # FIX : combiner le masque natif MASt3R (get_masks) avec un seuil
        # de confiance RELATIF (quantile), au lieu d'un seuil absolu arbitraire.
        m_geom=to_numpy(masks[i]).reshape(-1).astype(bool)        # fiabilite geometrique
        c_flat=conf_np.reshape(-1)
        conf_thr_rel=np.quantile(c_flat, 0.50)                    # garde la moitie la + sure
        mask=m_geom & (c_flat>max(conf_thr_rel, CONF_THR))
        all_pts.append(pts_np.reshape(-1,3)[mask])
        img_bgr=cv2.imread(img_path)
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        cols=cv2.resize(img_rgb,(W,H)).reshape(-1,3)/255.0
        all_cols.append(cols[mask])
        focal=float(focals[i]) if focals.ndim==1 else float(focals[i][0])
        cam_data[Path(img_path).name]={"pose_4x4":poses[i].tolist(),
            "K":[[focal,0,W/2],[0,focal,H/2],[0,0,1]],
            "focal":focal,"image_size":[W,H]}
        # Pour S0 : sauver la grille H,W,3 + masque valide par pixel
        native_per_image[Path(img_path).name]={
            "pts3d_HW3": pts_np.astype(np.float32),
            "valid_HW":  mask.reshape(H,W),
            "H": H, "W": W,
        }
    pts_all=np.concatenate(all_pts); cols_all=np.clip(np.concatenate(all_cols),0,1)
    centroid=pts_all.mean(axis=0); dists=np.linalg.norm(pts_all-centroid,axis=1)
    inlier=dists<dists.mean()+3*dists.std()
    pts_all=pts_all[inlier]; cols_all=cols_all[inlier]
    log(f"Points : {len(pts_all):,} ({(~inlier).sum():,} outliers retires)",2)
    ply_path=f"{out_dir}/pointcloud_coarse.ply"
    cols_u8=(cols_all*255).astype(np.uint8)
    with open(ply_path,"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_all)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt,col in zip(pts_all,cols_u8):
            f.write(f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f} "
                    f"{col[0]} {col[1]} {col[2]}\n")
    with open(f"{out_dir}/camera_poses.json","w") as f:
        json.dump(cam_data,f,indent=2)
    # Sauvegarde des pts3d natifs par image pour Approche S0
    np.savez_compressed(f"{out_dir}/native_pts3d.npz",
        **{f"{k}__pts": v["pts3d_HW3"] for k,v in native_per_image.items()},
        **{f"{k}__valid": v["valid_HW"] for k,v in native_per_image.items()})
    log(f"OK {len(pts_all):,} pts | {len(cam_data)} cameras",2)
    return pts_all, cols_all, cam_data, native_per_image


# ════════════════════════════════════════════════════════════════════
# PRIMITIVE NATIVE -- Association 2D<->3D sans projection inverse
# ════════════════════════════════════════════════════════════════════
#
# Ces fonctions remplacent la projection inverse `pose @ p + lecture
# masque a (u,v)` qui introduit du bruit aux bords. A la place on
# exploite que MASt3R produit pts3d natifs (H,W,3) deja indexes par
# pixel image source. L association devient une indexation directe.
#
# Architecture :
#   get_native_seeds_for_det()  -> seeds propres par vue pour 1 dechet
#   vote_3d_consensus()         -> vote multi-vues dans l'espace 3D
#                                   (sans projection -- remplace
#                                    vote_sam_single dans A/B/D)
# ════════════════════════════════════════════════════════════════════

def get_native_seeds_for_det(det_idx, all_det_sam, camera_data,
                              native_per_image, masks_dir,
                              scene_name=None):
    """
    Pour la detection globale det_idx, recupere les seeds 3D propres
    issues de CHAQUE vue ou le dechet apparait, par indexation native
    (zero projection inverse).

    Retourne :
      seeds_per_view : dict {img_name: pts3d (N_i, 3)}
      classe, conf   : info YOLO majoritaire
    """
    seeds_per_view = {}
    classes_votes = {}
    confs_votes = []

    for img_name, cam in camera_data.items():
        if img_name not in native_per_image:
            continue
        if img_name not in all_det_sam:
            continue
        dets_img = all_det_sam[img_name].get("detections", [])
        if det_idx >= len(dets_img):
            continue

        det = dets_img[det_idx]
        # Recuperer le masque SAM
        stem = Path(img_name).stem
        mask_path = f"{masks_dir}/{stem}_mask_{det_idx}.png"
        if not os.path.exists(mask_path):
            # Fallback : champ mask_path dans le json
            mask_path = det.get("mask_path", "")
            if not mask_path or not os.path.exists(mask_path):
                continue
        sam_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if sam_mask is None:
            continue

        nat = native_per_image[img_name]
        pts3d_HW3 = nat["pts3d_HW3"]
        valid_HW = nat["valid_HW"]
        H, W = nat["H"], nat["W"]

        # Redimensionner le masque a la taille native pts3d MASt3R
        sam_resized = cv2.resize(sam_mask, (W, H),
                                  interpolation=cv2.INTER_NEAREST) > 127

        # Indexation DIRECTE -- zero projection
        select = valid_HW & sam_resized
        if select.sum() < 5:
            continue

        pts_view = pts3d_HW3[select]
        # Filtre des points non finis
        finite = np.isfinite(pts_view).all(axis=1)
        pts_view = pts_view[finite]
        if len(pts_view) < 5:
            continue

        seeds_per_view[img_name] = pts_view
        cls = det.get("classe", "?")
        classes_votes[cls] = classes_votes.get(cls, 0) + 1
        confs_votes.append(det.get("conf", 0.0))

    # Classe majoritaire + confiance moyenne
    if classes_votes:
        classe = max(classes_votes.items(), key=lambda x: x[1])[0]
        conf = float(np.mean(confs_votes))
    else:
        classe, conf = "?", 0.0

    return seeds_per_view, classe, conf


def vote_3d_consensus(seeds_per_view, points, eps=0.02, min_views=3):
    """
    Vote multi-vues pur dans l'espace 3D (sans projection).

    Pour chaque point p du nuage global, compte le nombre de vues
    DISTINCTES qui contiennent une seed dans un rayon eps autour de p.
    Garde les points vus par >= min_views vues.

    C'est l'equivalent fonctionnel du vote SAM, mais l'association
    se fait par proximite 3D au lieu de projection inverse. Donc
    les erreurs de pose/K/discretisation ne polluent pas le resultat.

    Args:
        seeds_per_view : dict {img_name: pts3d (N_i, 3)} -- sortie de
                         get_native_seeds_for_det
        points         : nuage global (N, 3)
        eps            : rayon de voisinage (m). Defaut 2cm pour
                         scenes smartphone metriques.
        min_views      : nb min de vues distinctes pour valider un pt

    Retourne :
        votes : np.ndarray (N,) -- nb vues distinctes par point
        keep_mask : np.ndarray (N,) bool -- votes >= min_views
    """
    from scipy.spatial import cKDTree

    if not seeds_per_view:
        return np.zeros(len(points), dtype=np.int32), \
               np.zeros(len(points), dtype=bool)

    N = len(points)
    votes = np.zeros(N, dtype=np.int32)
    tree = cKDTree(points)

    # Pour chaque vue, marquer les points du nuage global qui sont
    # proches d'au moins une seed de cette vue. On +1 le vote.
    for img_name, seeds in seeds_per_view.items():
        if len(seeds) == 0:
            continue
        # Pour chaque seed, trouver tous les points du nuage dans eps
        idx_lists = tree.query_ball_point(seeds, r=eps)
        # Aplatir et deduper PAR VUE (une vue = 1 vote par point max)
        seen_this_view = set()
        for lst in idx_lists:
            seen_this_view.update(lst)
        if seen_this_view:
            seen_arr = np.fromiter(seen_this_view, dtype=np.int64)
            votes[seen_arr] += 1

    keep_mask = votes >= min_views
    return votes, keep_mask


# ───────────────────────────────────────────────────
# VOTE SAM (commun Approches A et B)
# ───────────────────────────────────────────────────
def vote_sam_single(points, colors, camera_data, all_det_sam, masks_dir, det_idx):
    """
    Vote multi-vues SAM pour UN dechet (det_idx).

    Pour chaque camera :
    1. Projeter tous les points 3D dans l image 2D
    2. Charger le masque SAM du dechet det_idx
    3. Voter +1 pour chaque point qui tombe dans le masque

    Retourne le vecteur de votes (taille N = nb points coarse).
    """
    N=len(points); votes=np.zeros(N,dtype=np.int32); n_proc=0
    for img_name,cam in camera_data.items():
        if img_name not in all_det_sam: continue
        dets=all_det_sam[img_name].get("detections",[])
        if det_idx>=len(dets): continue
        stem=Path(img_name).stem
        local=f"{masks_dir}/{stem}_mask_{det_idx}.png"
        if not os.path.exists(local): continue
        mask_orig=cv2.imread(local,0)
        if mask_orig is None: continue
        W_m,H_m=cam["image_size"]
        mask_bin=(cv2.resize(mask_orig,(W_m,H_m),
            interpolation=cv2.INTER_NEAREST)>127).astype(np.uint8)
        if mask_bin.sum()==0: continue
        K=np.array(cam["K"]); pose=np.array(cam["pose_4x4"])
        fx=K[0,0]; fy=K[1,1]; cx_=K[0,2]; cy_=K[1,2]
        pts_h=np.hstack([points,np.ones((N,1))])
        pts_cam=(pose@pts_h.T).T[:,:3]
        front=pts_cam[:,2]>0.01; idx=np.where(front)[0]
        if len(idx)==0: continue
        pc=pts_cam[idx]
        u=(pc[:,0]*fx/pc[:,2]+cx_).astype(int)
        v=(pc[:,1]*fy/pc[:,2]+cy_).astype(int)
        in_view=(u>=0)&(u<W_m)&(v>=0)&(v<H_m)
        idx2=idx[in_view]; u2=u[in_view]; v2=v[in_view]
        np.add.at(votes,idx2[mask_bin[v2,u2]>0],1)
        n_proc+=1
    return votes, n_proc


def save_dechet_files(out_dir, prefix, det_idx, pts_d, cols_d,
                      classe, conf, volume_cm3, met,
                      points, colors, col, scores=None,
                      all_dechet_masks=None, scene_origin=None,
                      scene_R=None):
    """
    Sauvegarde tous les fichiers PLY + annotation JSON d un dechet.

    Fichiers produits :
    - dechet_X_N_points.ply  : nuage de points du dechet
    - dechet_X_N_obb.ply     : boite englobante + label texte 3D
    - dechet_X_N_mesh.ply    : maillage Poisson 3D
    - dechet_X_N_confidence.ply : scores confiance GSCE-C (si Approche B)
    """
    import open3d as o3d

    write_ply(f"{out_dir}/{prefix}_{det_idx}_points.ply",
              pts_d, np.tile(col,(len(pts_d),1)).astype(np.uint8))

    if scores is not None:
        write_ply_confidence(
            f"{out_dir}/{prefix}_{det_idx}_confidence.ply", pts_d, scores)

    if all_dechet_masks is not None:
        all_dechet_masks.append(make_dechet_mask(points, pts_d))

    # Calculer coordonnees dans le repere scene normalise
    if scene_origin is not None and scene_R is not None:
        center_monde = np.array(met["obb_center"])
        center_scene = transform_to_scene_frame(
            center_monde[np.newaxis,:], scene_origin, scene_R)[0]
        coords_scene = format_coordinates(center_scene)
    else:
        center_monde = np.array(met["obb_center"])
        coords_scene = {"x_m":round(float(center_monde[0]),4),
                        "y_m":round(float(center_monde[1]),4),
                        "z_m":round(float(center_monde[2]),4),
                        "description":"repere MASt3R (non normalise)"}

    # OBB + label lisible
    dims=np.array(met["obb_dims"]); center_o=np.array(met["obb_center"])
    R_o=np.array(met["obb_rotation"])
    obb_pts,obb_cols,_=build_obb_with_label(
        pts_d, classe, conf, volume_cm3,
        dims.tolist(), center_o.tolist(), R_o.tolist(),
        col, coords_scene)
    write_ply(f"{out_dir}/{prefix}_{det_idx}_obb.ply", obb_pts, obb_cols)

    # Maillage
    try:
        if scores is not None:
            mesh,_ = o3d.geometry.TriangleMesh(), None
            mesh,_ = poisson_weighted(pts_d,cols_d,scores,points,colors)
        else:
            mesh,_ = poisson_standard(pts_d,cols_d,points,colors)
        o3d.io.write_triangle_mesh(
            f"{out_dir}/{prefix}_{det_idx}_mesh.ply", mesh)
    except Exception as e:
        log(f"Maillage ERREUR : {e}",2)

    return coords_scene


# ───────────────────────────────────────────────────────────────────
# APPROCHE S0 : Native Pixel Mapping (baseline)
# ───────────────────────────────────────────────────────────────────
#
# Principe : ne PAS faire de projection inverse 3D->2D->masque.
# A la place, on exploite le fait que MASt3R produit pour chaque
# image un tableau pts3d de forme (H,W,3) ou chaque pixel (u,v)
# a sa coordonnee 3D directement associee.
#
# Pour chaque image :
#   1. Recuperer pts3d natifs (H,W,3) deja sauves par run_mast3r
#   2. Pour chaque detection SAM, redimensionner le masque a (H,W)
#   3. Selectionner les pts3d dont (valid_mast3r & sam_mask) = True
#   4. Concatener sur toutes les images du dechet
#   5. Cluster DBSCAN pour separer les multi-vues du meme dechet
#
# Avantage : zero erreur de projection (poses, K, discretisation).
# Limite : pas de vote multi-vues, donc moins de robustesse au bruit
# YOLO mais beaucoup plus propre sur les bords d objet.
#
# Reference : Internal research reference (anonymized).
# ───────────────────────────────────────────────────────────────────
def run_approche_S0(points, colors, camera_data, all_det_sam, all_det,
                    masks_dir, out_dir, scene_origin, scene_R,
                    native_per_image,
                    ground_plane=None, scene_density=0, scene_name=""):
    """
    Approche S0 : Native Pixel Mapping (baseline).

    Utilise les pts3d natifs (H,W,3) de MASt3R + masques SAM
    redimensionnes a (H,W). Pas de projection inverse.

    Cette approche sert de BASELINE pour mesurer l apport des
    approches A/B/C/D (vote multi-vues, expansion geometrique, etc).
    """
    log("[APPROCHE S0] Native Pixel Mapping (baseline)...", 1)

    if native_per_image is None:
        log("ERREUR : native_pts3d.npz absent -- relance avec --skip_mast3r=False", 2)
        return []

    from sklearn.cluster import DBSCAN

    # Compter le nombre max de detections (toutes images confondues)
    max_det = max(
        (len(v.get("detections", [])) for v in all_det_sam.values()),
        default=0
    )
    if max_det == 0:
        log("Aucune detection a traiter", 2)
        return []

    results = []
    all_masks_for_fond = []

    # Pour chaque detection (chaque "dechet potentiel" indexe globalement)
    for det_idx in range(max_det):

        pts_dechet = []
        cols_dechet = []
        n_vues_contribuant = 0

        for img_name, cam_info in camera_data.items():
            if img_name not in native_per_image:
                continue
            nat = native_per_image[img_name]
            pts3d_HW3 = nat["pts3d_HW3"]
            valid_HW = nat["valid_HW"]
            H, W = nat["H"], nat["W"]

            # Trouver le masque SAM pour det_idx dans cette image
            dets_img = all_det_sam.get(img_name, {}).get("detections", [])
            if det_idx >= len(dets_img):
                continue
            det_info = dets_img[det_idx]
            mask_path = det_info.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                continue

            sam_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if sam_mask is None:
                continue

            # Redimensionner le masque SAM a la taille de pts3d natif (H,W)
            sam_resized = cv2.resize(sam_mask, (W, H),
                                      interpolation=cv2.INTER_NEAREST) > 127

            # Selection directe : valide MASt3R ET dans le masque SAM
            select = valid_HW & sam_resized
            if select.sum() < 5:
                continue

            pts_dechet.append(pts3d_HW3[select])
            # Couleurs : reprendre depuis l image source
            img_bgr = cv2.imread(f"{Path(masks_dir).parent}/scenes/{scene_name}/{img_name}")
            if img_bgr is None:
                # Fallback : essayer un autre chemin
                img_bgr = cv2.imread(f"{SCENES_DIR}/{scene_name}/{img_name}")
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                cols_resized = cv2.resize(img_rgb, (W, H)) / 255.0
                cols_dechet.append(cols_resized[select])
            else:
                cols_dechet.append(np.full((select.sum(), 3), 0.5))

            n_vues_contribuant += 1

        if not pts_dechet or n_vues_contribuant < 2:
            log(f"Det S0-{det_idx} : <2 vues contributives, skip", 2)
            continue

        pts_all = np.concatenate(pts_dechet, axis=0)
        cols_all = np.concatenate(cols_dechet, axis=0)

        # Filtre des points non-finis
        finite = np.isfinite(pts_all).all(axis=1)
        pts_all = pts_all[finite]
        cols_all = cols_all[finite]

        if len(pts_all) < 30:
            log(f"Det S0-{det_idx} : <30 pts apres filtre, skip", 2)
            continue

        # DEBRUITAGE ROBUSTE (contribution principale)
        # Pipeline de nettoyage 3-etages, robuste sur toute scene :
        # DBSCAN -> plan-fond -> compactage
        pts_all, cols_all, clean_info = clean_seeds_robust(
            pts_all, cols_all, verbose=True)
        if len(pts_all) < 30:
            log(f"Det S0-{det_idx} : <30 pts apres clean_seeds, skip", 2)
            continue

        # Cluster pour garder le plus gros (cas multi-objets dans 1 detection)
        try:
            db = DBSCAN(eps=DBSCAN_EPS * 1.5,
                        min_samples=DBSCAN_MIN_SAMPLES).fit(pts_all)
            labels = db.labels_
            if (labels >= 0).sum() == 0:
                pts_d, cols_d = pts_all, cols_all
            else:
                vals, counts = np.unique(labels[labels >= 0], return_counts=True)
                best_label = vals[np.argmax(counts)]
                keep_c = labels == best_label
                pts_d = pts_all[keep_c]
                cols_d = cols_all[keep_c]
        except Exception:
            pts_d, cols_d = pts_all, cols_all

        if len(pts_d) < 30:
            continue

        classe, conf = get_classe_from_det(all_det, det_idx)
        col = COLORS_DECHETS[len(results) % len(COLORS_DECHETS)]

        try:
            mesh, met = poisson_standard(pts_d, cols_d, points, colors)

            # Filtre 3D geometrique (memes regles que A/B/C/D)
            center_scene = transform_to_scene_frame(
                np.array(met["obb_center"])[np.newaxis, :],
                scene_origin, scene_R)[0]
            _, _, view_ratio = count_views_visible(
                np.array(met["obb_center"]), 0.1, camera_data)
            candidate_info = {
                "pts":        pts_d,
                "obb_dims":   met["obb_dims"],
                "obb_center": met["obb_center"],
                "volume_cm3": met["volume_cm3"],
                "z_sol":      float(center_scene[2]),
            }
            is_valid, rules, n_pass = filter_fp_3d_geometric(
                candidate_info, ground_plane, scene_density,
                view_ratio, save_dir=out_dir)
            if not is_valid:
                log(f"Dechet S0-{det_idx} REJETE (FP3D) : {n_pass}/6", 2)
                continue

            coords = save_dechet_files(out_dir, "dechet_S0", det_idx,
                pts_d, cols_d, classe, conf, met["volume_cm3"], met,
                points, colors, col, None, all_masks_for_fond,
                scene_origin, scene_R)
            met.update({
                "id": det_idx, "classe": classe, "conf": float(conf),
                "n_pts": len(pts_d), "coords_scene": coords,
                "n_views_contrib": n_vues_contribuant,
                "fp3d_rules": rules, "fp3d_pass": n_pass,
            })
            met = add_svd_volume_to_met(met, pts_d)
            results.append(met)
            dims = met["obb_dims"]
            log(f"Dechet S0-{det_idx} [{classe}] : {len(pts_d):,} pts | "
                f"{n_vues_contribuant} vues | "
                f"OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
                f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
                f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
                f"ratio={met['volume_ratio']:.2f} | "
                f"FP3D={n_pass}/6 | z_sol={coords['z_m']:.3f}m", 2)
        except Exception as e:
            log(f"Dechet S0-{det_idx} ERREUR : {e}", 2)

    log(f"OK Approche S0 : {len(results)} dechet(s)", 1)
    return results


# ═══════════════════════════════════════════════════════════════════
# APPROCHE E -- Coastal-Waste-3D : Native Multi-view Consensus Segmentation
# Author, 2026
# ═══════════════════════════════════════════════════════════════════
#
# CONTRIBUTION PRINCIPALE. Methode unique remplacant l empilement
# A/B/C/D instable. Quatre etages, chacun justifie :
#
#   E1. Seeds natives multi-vues (indexation directe pts3d[i][mask_i],
#       zero projection inverse -> pas d erreur de pose/intrinseque).
#
#   E2. Confiance par consensus multi-vues SANS expansion : chaque seed
#       recoit un poids = nombre de vues qui la confirment dans son
#       voisinage 3D. Pas d expansion geodesique -> pas d explosion ni
#       de debordement sur le fond (resout l echec de GSCE-C).
#
#   E3. Separation objet/sol en REPERE NORMALISE : on retire les seeds
#       dont la hauteur (Z scene) est sous un seuil relatif local, et on
#       garde le composant connecte principal (DBSCAN). Robuste a toute
#       echelle (contrairement au seuil metrique absolu).
#
#   E4. Completion surfacique controlee : Poisson sur seeds ponderees par
#       la confiance E2, avec densite minimale -> comble l interieur creux
#       (canettes) SANS deborder, car l enveloppe est bornee par les seeds.
#
# Pourquoi c'est publiable : combinaison originale d association native
# MASt3R + consensus multi-vues 3D pondere + separation sol canonique,
# appliquee a la caracterisation de dechets a partir d images smartphone
# non calibrees. Pas d expansion incontrolee = robustesse demontree sur
# objets varies (cylindriques, plats, ecrases).
# ═══════════════════════════════════════════════════════════════════

def compute_multiview_confidence(seeds_per_view, eps=0.012):
    """
    Etage E2 : confiance par consensus multi-vues, SANS expansion.

    Concatene les seeds de toutes les vues, puis pour chaque point
    compte combien de VUES DISTINCTES ont une seed dans son voisinage
    de rayon eps. Plus un point est vu coherent par de nombreuses vues,
    plus il est fiable (= vraie surface objet, pas artefact d une vue).

    Args:
        seeds_per_view : dict {img_name: (Ni,3)}
        eps : rayon de voisinage (unites scene)

    Returns:
        pts_all (M,3), confidence (M,) in [0,1], view_count (M,) int
    """
    from scipy.spatial import cKDTree
    if not seeds_per_view:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)

    names = list(seeds_per_view.keys())
    n_views = len(names)
    blocks = [np.asarray(seeds_per_view[n]) for n in names]
    pts_all = np.concatenate(blocks, axis=0)
    view_id = np.concatenate([
        np.full(len(b), i, dtype=np.int32) for i, b in enumerate(blocks)
    ])

    tree = cKDTree(pts_all)
    view_count = np.zeros(len(pts_all), dtype=np.int32)
    # Pour chaque point, ensemble des vues distinctes dans son voisinage
    neighbors = tree.query_ball_point(pts_all, r=eps)
    for i, nb in enumerate(neighbors):
        if nb:
            view_count[i] = len(np.unique(view_id[nb]))
    # Normaliser en confiance [0,1]
    confidence = view_count / max(n_views, 1)
    confidence = np.clip(confidence, 0.0, 1.0)
    return pts_all, confidence, view_count


def run_approche_E(points, colors, camera_data, all_det_sam, all_det,
                   masks_dir, out_dir, scene_origin, scene_R,
                   native_per_image,
                   ground_plane=None, scene_density=0, scene_name=""):
    """
    Coastal-Waste-3D : Native Multi-view Consensus Segmentation.
    Methode unique, robuste, contribution principale.
    """
    log("[APPROCHE E] Coastal-Waste-3D : Native Multi-view Consensus...", 1)
    if native_per_image is None:
        log("ERREUR E : native_per_image requis", 2)
        return []

    from sklearn.cluster import DBSCAN

    # Parametres (overridables)
    CONF_MIN     = float(os.environ.get("NMC_CONF_MIN", "0.15"))   # vues min (fraction)
    CONS_EPS     = float(os.environ.get("NMC_EPS", "0.012"))       # rayon consensus
    HEIGHT_PCTL  = float(os.environ.get("NMC_HEIGHT_PCTL", "8"))   # percentile sol
    DBSCAN_EPS_F = float(os.environ.get("NMC_DBSCAN_EPSF", "2.5"))
    VOTE3D_MIN   = int(os.environ.get("NMC_VOTE3D_MINVIEWS", "3"))

    # ── Variantes de modules (interrupteurs d'ablation) ──
    # Voir docstring. Defaut = comportement valide (~75%).
    E1_VAR = os.environ.get("NMC_E1_VARIANT", "native")       # native|voteSAM
    E2_VAR = os.environ.get("NMC_E2_VARIANT", "consensus")    # consensus|vote3D|off
    E3_VAR = os.environ.get("NMC_E3_VARIANT", "soft")         # soft|hard|off
    E5_VAR = os.environ.get("NMC_E5_VARIANT", "poisson_std")  # poisson_std|poisson_weighted|obb_only
    E6_VAR = os.environ.get("NMC_E6_VARIANT", "full")         # full|noflat|off
    _defaults = [(E1_VAR, "native"), (E2_VAR, "consensus"),
                 (E3_VAR, "soft"), (E5_VAR, "poisson_std"), (E6_VAR, "full")]
    if any(v != d for v, d in _defaults):
        log(f"Variantes E : E1={E1_VAR} E2={E2_VAR} E3={E3_VAR} "
            f"E5={E5_VAR} E6={E6_VAR}", 1)

    # La variante frustum de E1 est scene-niveau (intersection de bbox +
    # clustering), pas par-detection : c'est exactement l'approche C.
    # On ne la simule pas dans E pour ne pas produire de donnees trompeuses.
    if E1_VAR == "frustum":
        log("E1=frustum non supporte dans E (scene-niveau). "
            "Lancer --approach C pour ce point de comparaison.", 1)
        return []

    max_det = max((len(v.get("detections", []))
                   for v in all_det_sam.values()), default=0)
    if max_det == 0:
        log("Aucune detection", 2)
        return []

    # Pre-calcul : hauteurs scene de reference (pour seuil sol relatif)
    all_heights = (points - scene_origin) @ scene_R[:, 2]
    ground_h = np.percentile(all_heights, HEIGHT_PCTL)
    span_h = np.percentile(all_heights, 95) - np.percentile(all_heights, 5)
    # FIX A : marge tres fine par defaut (soft). hard = marge large.
    if E3_VAR == "hard":
        GROUND_MARGIN_E = float(os.environ.get("NMC_GROUND_MARGIN", "0.02"))
    else:
        GROUND_MARGIN_E = float(os.environ.get("NMC_GROUND_MARGIN", "0.005"))
    height_thresh = ground_h + GROUND_MARGIN_E * max(span_h, 1e-6)

    results = []
    all_masks = []

    for det_idx in range(max_det):
        # ════════════ E1 + E2 : seeds + consensus ════════════
        classe, conf = get_classe_from_det(all_det_sam, det_idx)

        if E1_VAR == "voteSAM":
            # Variante E1 : vote SAM par projection inverse (ex-logique
            # voteSAM/A). Produit directement le nuage objet en votant sur
            # le nuage global -> pas de seeds par vue, E2 est implicite.
            votes, n_proc = vote_sam_single(
                points, colors, camera_data, all_det_sam, masks_dir, det_idx)
            keep = votes >= VOTE3D_MIN
            if keep.sum() < 30:
                continue
            pts_c = points[keep].copy()
            vmax = max(int(votes.max()), 1)
            conf_c = np.clip(votes[keep].astype(float) / vmax, 0.05, 1.0)
            n_views = int(n_proc)
        else:
            # E1 native (defaut) : seeds 2D->3D natives par vue
            seeds_per_view, classe, conf = get_native_seeds_for_det(
                det_idx, all_det_sam, camera_data, native_per_image,
                masks_dir, scene_name)
            n_views = len(seeds_per_view)
            if n_views < 2:
                continue

            # E2 : consensus multi-vues
            if E2_VAR == "off":
                # Aucune ponderation : concatener toutes les seeds
                all_s = [np.asarray(pv) for pv in seeds_per_view.values()
                         if len(pv) > 0]
                if not all_s:
                    continue
                pts_c = np.concatenate(all_s, axis=0)
                conf_c = np.ones(len(pts_c))
            elif E2_VAR == "vote3D":
                # Vote binaire : seuil sur nb de vues
                votes, keep_mask = vote_3d_consensus(
                    seeds_per_view, points, eps=CONS_EPS,
                    min_views=VOTE3D_MIN)
                if keep_mask.sum() < 30:
                    continue
                pts_c = points[keep_mask].copy()
                conf_c = np.ones(int(keep_mask.sum()))
            else:
                # consensus pondere (defaut)
                pts_all, confidence, view_count = \
                    compute_multiview_confidence(seeds_per_view, eps=CONS_EPS)
                if len(pts_all) < 30:
                    continue
                conf_mask = confidence >= CONF_MIN
                if conf_mask.sum() < 30:
                    conf_mask = view_count >= 2
                pts_c = pts_all[conf_mask]
                conf_c = confidence[conf_mask]

        if len(pts_c) < 30:
            log(f"Det E-{det_idx} : <30 pts apres E1/E2, skip", 2)
            continue

        # ════════════ E3 : separation objet/sol ════════════
        if E3_VAR != "off":
            heights = (pts_c - scene_origin) @ scene_R[:, 2]
            above = heights > height_thresh
            frac_below = 1.0 - (above.sum() / max(len(pts_c), 1))
            if E3_VAR == "hard":
                # Coupe brutale, sans garde anti-amputation
                if above.sum() >= 30:
                    pts_c = pts_c[above]
                    conf_c = conf_c[above]
            else:
                # soft (defaut) : garde anti-amputation des objets debout
                if above.sum() >= 30 and frac_below < 0.40:
                    pts_c = pts_c[above]
                    conf_c = conf_c[above]
                elif frac_below >= 0.40:
                    log(f"Det E-{det_idx} : coupe sol ignoree "
                        f"({frac_below*100:.0f}% sous seuil = objet bas/debout)", 2)

        if len(pts_c) < 30:
            log(f"Det E-{det_idx} : <30 pts apres E3, skip", 2)
            continue

        # Couleurs : reprises depuis le nuage global le plus proche
        from scipy.spatial import cKDTree
        gtree = cKDTree(points)
        _, gidx = gtree.query(pts_c, k=1)
        cols_c = colors[gidx.flatten()]

        # ════════════ E4 : nettoyage robuste ════════════
        # Composant principal DBSCAN + plan RANSAC + compactage MAD.
        # Respecte CLEAN_DBSCAN / CLEAN_PLANE / CLEAN_COMPACT (ablation).
        n_before_e4 = len(pts_c)
        pts_c, cols_c, _clean_info = clean_seeds_robust(
            pts_c, cols_c, verbose=False)
        if len(pts_c) != n_before_e4:
            # Re-alignement leger : conf devient la moyenne (la re-indexation
            # exacte point-a-point apres nettoyage serait couteuse et peu utile)
            conf_c = np.full(max(len(pts_c), 1),
                             float(np.mean(conf_c)) if n_before_e4 else 0.5)
            conf_c = conf_c[:len(pts_c)] if len(pts_c) else conf_c

        if len(pts_c) < 30:
            log(f"Det E-{det_idx} : <30 pts apres E4 (nettoyage), skip", 2)
            continue

        # ════════════ E5 : reconstruction surfacique ════════════
        col = COLORS_DECHETS[len(results) % len(COLORS_DECHETS)]
        try:
            if E5_VAR == "obb_only":
                # Pas de maillage : OBB SVD seul (rapide, borne sup. volume)
                svd = compute_obb_volume_svd(pts_c)
                met = {
                    "obb_center":   svd["centroid_m"],
                    "obb_dims":     svd["dims_m"],
                    "obb_rotation": svd["axes_R"],
                    "volume_cm3":   svd["volume_cm3"],
                }
                mesh = None
            elif E5_VAR == "poisson_weighted":
                # Poisson pondere par la confiance
                mesh, met = poisson_weighted(
                    pts_c, cols_c, conf_c, points, colors)
            else:
                # poisson_std (defaut)
                mesh, met = poisson_standard(pts_c, cols_c, points, colors)

            center_scene = transform_to_scene_frame(
                np.array(met["obb_center"])[np.newaxis, :],
                scene_origin, scene_R)[0]
            _, _, view_ratio = count_views_visible(
                np.array(met["obb_center"]), 0.1, camera_data)

            # ════════════ E6 : filtrage faux positifs 3D ════════════
            if E6_VAR == "off":
                is_valid, rules, n_pass = True, {}, 6
            else:
                # noflat : l exemption objets plats est desactivee en amont
                # via FP3D_FLAT_FOOTPRINT_MAX=0 (emis par env_overrides).
                candidate_info = {
                    "pts": pts_c, "obb_dims": met["obb_dims"],
                    "obb_center": met["obb_center"],
                    "volume_cm3": met["volume_cm3"],
                    "z_sol": float(center_scene[2]),
                }
                is_valid, rules, n_pass = filter_fp_3d_geometric(
                    candidate_info, ground_plane, scene_density,
                    view_ratio, save_dir=out_dir)
            if not is_valid:
                log(f"Dechet E-{det_idx} REJETE (FP3D) : {n_pass}/6 | "
                    f"{[k for k,v in rules.items() if not v]}", 2)
                continue

            coords = save_dechet_files(
                out_dir, "dechet_E", det_idx, pts_c, cols_c,
                classe, conf, met["volume_cm3"], met,
                points, colors, col, conf_c, all_masks,
                scene_origin, scene_R)
            met.update({
                "id": det_idx, "classe": classe, "conf": float(conf),
                "n_pts": len(pts_c), "coords_scene": coords,
                "n_views": n_views,
                "conf_mean": float(conf_c.mean()),
                "fp3d_rules": rules, "fp3d_pass": n_pass,
                "variant_E1": E1_VAR, "variant_E2": E2_VAR,
                "variant_E3": E3_VAR, "variant_E5": E5_VAR,
                "variant_E6": E6_VAR,
            })
            met = add_svd_volume_to_met(met, pts_c)

            # Garde-fous anti-faux-positifs (desactives si E6 off)
            if E6_VAR != "off":
                ratio = met.get("volume_ratio", 0)
                conf_mean_val = float(conf_c.mean())
                if ratio > 1.05:
                    log(f"Dechet E-{det_idx} REJETE : ratio={ratio:.2f} > 1.05 "
                        f"(artefact maillage, pas un objet creux)", 2)
                    continue
                if n_views < 5 and conf_mean_val < 0.35:
                    log(f"Dechet E-{det_idx} REJETE : {n_views} vues + "
                        f"conf={conf_mean_val:.2f} (detection peu fiable)", 2)
                    continue

            results.append(met)
            dims = met["obb_dims"]
            log(f"Dechet E-{det_idx} [{classe}] : {len(pts_c):,} pts | "
                f"{n_views} vues | conf_moy={conf_c.mean():.2f} | "
                f"OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
                f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
                f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
                f"ratio={met['volume_ratio']:.2f} | FP3D={n_pass}/6 | "
                f"z_sol={coords['z_m']:.3f}m", 2)
        except Exception as e:
            log(f"Dechet E-{det_idx} ERREUR : {e}", 2)

    log(f"OK Approche E (Coastal-Waste-3D) [{E1_VAR}/{E2_VAR}/{E3_VAR}/"
        f"{E5_VAR}/{E6_VAR}] : {len(results)} dechet(s)", 1)
    return results


# ───────────────────────────────────────────────────
# APPROCHE A : Vote 3D multi-vues sur seeds natives
# ───────────────────────────────────────────────────
def run_approche_A(points, colors, camera_data, all_det_sam,
                   all_det, masks_dir, out_dir,
                   scene_origin, scene_R,
                   ground_plane=None, scene_density=0,
                   scene_name="",
                   native_per_image=None):
    """
    Approche A : vote 3D multi-vues sur seeds natives.

    Principe : vote dans l espace 3D sur des seeds extraites par
    indexation native. (La projection inverse pose@K introduit du
    bruit aux bords ; l indexation native l evite.)

    Pipeline :
      1. Pour chaque vue, extraire seeds 3D propres (indexation native)
      2. Voter dans l espace 3D : un point est garde s il est proche
         de seeds provenant de >= K vues distinctes (cohérence multi-vues)
      3. DBSCAN pour separer eventuels multi-objets
      4. NMS 3D + filtre FP geometrique + OBB + Poisson

    Pas d expansion geometrique -- c est la difference clef avec B.
    """
    log("[APPROCHE A] Vote 3D multi-vues sur seeds natives...", 1)
    if native_per_image is None:
        log("ERREUR A : native_per_image requis (relance avec MASt3R complet)", 2)
        return []

    max_det=max((len(v.get("detections",[])) for v in all_det_sam.values()),default=0)
    results=[]; all_masks=[]; det_info={}

    # Phase 1 : pour chaque detection, extraire seeds natives + vote 3D
    for det_idx in range(max_det):
        seeds_per_view, classe, conf = get_native_seeds_for_det(
            det_idx, all_det_sam, camera_data, native_per_image,
            masks_dir, scene_name)
        if len(seeds_per_view) < 2:
            continue
        # Vote 3D pur (sans projection inverse)
        votes, keep_mask = vote_3d_consensus(
            seeds_per_view, points,
            eps=0.02,                  # 2cm -- echelle smartphone metrique
            min_views=max(3, int(0.2*len(seeds_per_view))))
        vote_max = int(votes.max())
        if vote_max == 0:
            continue
        cands = points[keep_mask]
        if len(cands) < 30:
            continue
        center = points[votes >= max(1, vote_max-2)].mean(axis=0)
        det_info[det_idx] = {"center":center, "vote_max":vote_max,
                             "classe":classe, "conf":conf,
                             "votes":votes, "keep_mask":keep_mask,
                             "n_views":len(seeds_per_view)}
        log(f"Det {det_idx} [{classe} {conf:.0%}] : "
            f"{len(seeds_per_view)} vues | vote_max={vote_max} | "
            f"{int(keep_mask.sum())} pts retenus",2)

    # Phase 2 : NMS 3D
    kept_indices = nms_3d(det_info, points, {})
    log(f"Apres NMS 3D : {len(kept_indices)}/{len(det_info)} dechets gardes", 2)

    # Phase 3 : traiter les dechets gardes
    for det_idx in kept_indices:
        info = det_info[det_idx]
        keep_mask = info["keep_mask"]
        classe = info["classe"]; conf = info["conf"]
        cands = points[keep_mask]; cols_c = colors[keep_mask]
        pts_d, cols_d = get_best_cluster(cands, cols_c, info["center"])

        # DEBRUITAGE ROBUSTE avant reconstruction (contribution principale)
        # Critique pour A : le vote 3D peut inclure des points de fond
        # proches (eps=2cm). clean_seeds elimine le plan-fond + points isoles.
        log(f"clean_seeds A-{det_idx}...", 2)
        pts_d, cols_d, clean_info = clean_seeds_robust(
            pts_d, cols_d, verbose=True)
        if len(pts_d) < 30:
            log(f"Dechet A-{det_idx} : <30 pts apres clean, skip", 2)
            continue

        col = COLORS_DECHETS[len(results) % len(COLORS_DECHETS)]

        try:
            mesh, met = poisson_standard(pts_d, cols_d, points, colors)

            # Filtre 3D geometrique (inchange)
            center_scene = transform_to_scene_frame(
                np.array(met["obb_center"])[np.newaxis,:],
                scene_origin, scene_R)[0]
            _,_,view_ratio = count_views_visible(
                np.array(met["obb_center"]), 0.1, camera_data)
            candidate_info = {
                "pts":        pts_d,
                "obb_dims":   met["obb_dims"],
                "obb_center": met["obb_center"],
                "volume_cm3": met["volume_cm3"],
                "z_sol":      float(center_scene[2]),
            }
            is_valid, rules, n_pass = filter_fp_3d_geometric(
                candidate_info, ground_plane, scene_density,
                view_ratio, save_dir=out_dir)
            if not is_valid:
                log(f"Dechet A-{det_idx} REJETE (FP3D) : "
                    f"{n_pass}/6 regles | {[k for k,v in rules.items() if not v]}", 2)
                source_imgs = []
                for img_name, img_data in all_det.items():
                    dets_img = img_data.get("detections", [])
                    if det_idx < len(dets_img):
                        img_path = f"{Path(masks_dir).parent.parent.parent}/{scene_name}/{img_name}"
                        source_imgs.append((img_path, dets_img[det_idx]["bbox_2d"]))
                save_false_positive(out_dir, scene_name, det_idx,
                                     candidate_info, source_imgs, rules)
                continue
            coords = save_dechet_files(out_dir,"dechet_A",det_idx,
                pts_d,cols_d,classe,conf,met["volume_cm3"],met,
                points,colors,col,None,all_masks,scene_origin,scene_R)
            met.update({"id":det_idx,"classe":classe,"conf":float(conf),
                        "n_pts":len(pts_d),"coords_scene":coords,
                        "n_views":info["n_views"],
                        "fp3d_rules": rules, "fp3d_pass": n_pass,
                        "clean_info": clean_info})
            met = add_svd_volume_to_met(met, pts_d)
            results.append(met)
            dims = met["obb_dims"]
            log(f"Dechet A-{det_idx} [{classe}] : {len(pts_d):,} pts | "
                f"{info['n_views']} vues | "
                f"OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
                f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
                f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
                f"ratio={met['volume_ratio']:.2f} | "
                f"FP3D={n_pass}/6 | z_sol={coords['z_m']:.3f}m",2)
        except Exception as e:
            log(f"Dechet A-{det_idx} ERREUR : {e}",2)

    if all_masks: reconstruct_fond(points,colors,all_masks,out_dir)
    log(f"OK Approche A (native) : {len(results)} dechet(s)",1)
    return results


# ───────────────────────────────────────────────────
# APPROCHE B : SAM + GSCE-C + Poisson pondere
# ───────────────────────────────────────────────────
def run_approche_B(points, colors, camera_data, all_det_sam,
                   all_det, masks_dir, out_dir,
                   scene_origin, scene_R,
                   ground_plane=None, scene_density=0,
                   scene_name="",
                   native_per_image=None):
    """
    Approche B : seeds natives + GSCE-C + Poisson pondere.

    Principe : extraction native des seeds (zero bruit aux bords),
    consolidation 3D multi-vues, puis expansion GSCE-C.

    Pipeline :
      1. Seeds natives par vue (indexation directe pts3d[i] dans masque SAM)
      2. Consolidation 3D : agregation + filtre cohérence multi-vues
      3. GSCE-C : expansion geometrique avec scores confiance (NORMAL,
         PROFONDEUR, DISTANCE, DENSITE) -- contribution principale
      4. Poisson pondere par scores GSCE-C -> mesh + OBB + volume
      5. Filtre FP geometrique 6 regles
    """
    log("[APPROCHE B] Seeds natives + GSCE-C + Poisson pondere...", 1)
    if native_per_image is None:
        log("ERREUR B : native_per_image requis (relance avec MASt3R complet)", 2)
        return []

    max_det=max((len(v.get("detections",[])) for v in all_det_sam.values()),default=0)
    results=[]; all_masks=[]; det_info={}

    # Phase 1 : extraire seeds natives par detection
    for det_idx in range(max_det):
        seeds_per_view, classe, conf = get_native_seeds_for_det(
            det_idx, all_det_sam, camera_data, native_per_image,
            masks_dir, scene_name)
        if len(seeds_per_view) < 2:
            continue
        # Concatenation seeds toutes vues + filtre coherence multi-vues 3D
        all_seeds = np.concatenate(list(seeds_per_view.values()), axis=0)
        if len(all_seeds) < 30:
            continue
        # Consolidation : un point seed n'est garde que si plusieurs vues
        # voient le meme voisinage (vote 3D applique aux SEEDS elles-memes,
        # pas au nuage entier -- on rejette les seeds isolees d'une seule vue)
        from scipy.spatial import cKDTree
        seed_tree = cKDTree(all_seeds)
        # Pour chaque seed, compter le nombre de vues distinctes qui ont
        # contribue dans son voisinage de 1.5cm
        view_ids = np.concatenate([
            np.full(len(v), i, dtype=np.int32)
            for i,(_,v) in enumerate(seeds_per_view.items())
        ])
        coherence = np.zeros(len(all_seeds), dtype=np.int32)
        idx_lists = seed_tree.query_ball_point(all_seeds, r=0.015)
        for k, lst in enumerate(idx_lists):
            if lst:
                coherence[k] = len(np.unique(view_ids[lst]))
        min_coh = max(2, int(0.15*len(seeds_per_view)))
        keep_seeds = coherence >= min_coh
        pts_seeds = all_seeds[keep_seeds]
        if len(pts_seeds) < 30:
            log(f"Det B-{det_idx} : seeds insuffisantes apres coherence", 2)
            continue
        # Filtre statistique additionnel (SOR-like) puis DEBRUITAGE ROBUSTE
        # CRITIQUE pour B : si on laisse passer du bruit ici, GSCE-C va
        # l amplifier exponentiellement par expansion geometrique.
        pts_seeds, _ = filter_seed_outliers(pts_seeds, np.full_like(pts_seeds, 0.5))
        if len(pts_seeds) < 30:
            continue
        # Nettoyage robuste : DBSCAN + plan-fond + compactage
        pts_seeds, _, clean_info_b = clean_seeds_robust(
            pts_seeds, None, verbose=True)
        if len(pts_seeds) < 30:
            log(f"Det B-{det_idx} : <30 seeds apres clean, skip", 2)
            continue
        center = pts_seeds.mean(axis=0)
        det_info[det_idx] = {"center":center,
                              "classe":classe, "conf":conf,
                              "pts_seeds":pts_seeds,
                              "n_views":len(seeds_per_view),
                              "vote_max":len(seeds_per_view),
                              "clean_info":clean_info_b}
        log(f"Det {det_idx} [{classe} {conf:.0%}] : "
            f"{len(seeds_per_view)} vues | {len(pts_seeds)} seeds propres",2)

    # Phase 2 : NMS 3D
    kept_indices = nms_3d(det_info, points, {})
    log(f"Apres NMS 3D : {len(kept_indices)}/{len(det_info)} dechets gardes",2)

    # Phase 3 : GSCE-C + Poisson (inchanges -- la contribution scientifique)
    for det_idx in kept_indices:
        info = det_info[det_idx]
        classe = info["classe"]; conf = info["conf"]
        pts_seeds = info["pts_seeds"]

        log(f"GSCE-C det {det_idx} [{classe}] ({len(pts_seeds):,} seeds natives)...", 2)
        pts_exp, cols_exp, scores, n_iter = gsce_c(pts_seeds, points, colors)

        # POST-CLEAN : meme apres GSCE-C, on re-nettoie pour eviter
        # que les points expanses dans le fond ne pourrissent le Poisson.
        # Cette double-passe (pre + post) est la clef du resultat propre.
        log(f"clean_seeds B-{det_idx} (post-GSCE)...", 2)
        n_before = len(pts_exp)
        # GARDE-FOU : si GSCE-C a retourne trop peu de points (cas nuage
        # coarse trop reduit), on skip ce dechet proprement plutot que
        # de crasher Open3D ("No normals in PointCloud").
        if len(pts_exp) < 30:
            log(f"Dechet B-{det_idx} : GSCE-C a retourne {len(pts_exp)} pts "
                f"(<30), skip", 2)
            continue
        pts_exp_clean, cols_exp_clean, _ = clean_seeds_robust(
            pts_exp, cols_exp, verbose=False)
        if len(pts_exp_clean) >= 30:
            # Reajuster les scores GSCE-C en consequence (interpolation)
            from sklearn.neighbors import NearestNeighbors as _NN
            nn = _NN(n_neighbors=1).fit(pts_exp)
            _, idx_keep = nn.kneighbors(pts_exp_clean)
            scores = scores[idx_keep.flatten()]
            pts_exp = pts_exp_clean
            cols_exp = cols_exp_clean
            log(f"  Post-clean : {n_before} -> {len(pts_exp)} pts", 2)

        col = COLORS_DECHETS[len(results) % len(COLORS_DECHETS)]

        try:
            mesh, met = poisson_weighted(pts_exp, cols_exp, scores, points, colors)

            center_scene = transform_to_scene_frame(
                np.array(met["obb_center"])[np.newaxis,:],
                scene_origin, scene_R)[0]
            _,_,view_ratio = count_views_visible(
                np.array(met["obb_center"]), 0.1, camera_data)
            candidate_info = {
                "pts":        pts_exp,
                "obb_dims":   met["obb_dims"],
                "obb_center": met["obb_center"],
                "volume_cm3": met["volume_cm3"],
                "z_sol":      float(center_scene[2]),
            }
            is_valid, rules, n_pass = filter_fp_3d_geometric(
                candidate_info, ground_plane, scene_density,
                view_ratio, save_dir=out_dir)
            if not is_valid:
                log(f"Dechet B-{det_idx} REJETE (FP3D) : "
                    f"{n_pass}/6 regles | {[k for k,v in rules.items() if not v]}", 2)
                source_imgs = []
                for img_name, img_data in all_det.items():
                    dets_img = img_data.get("detections", [])
                    if det_idx < len(dets_img):
                        img_path = f"{Path(masks_dir).parent.parent.parent}/{scene_name}/{img_name}"
                        source_imgs.append((img_path, dets_img[det_idx]["bbox_2d"]))
                save_false_positive(out_dir, scene_name, det_idx,
                                     candidate_info, source_imgs, rules)
                continue
            coords = save_dechet_files(out_dir,"dechet_B",det_idx,
                pts_exp,cols_exp,classe,conf,met["volume_cm3"],met,
                points,colors,col,scores,all_masks,scene_origin,scene_R)
            met.update({"id":det_idx,"classe":classe,"conf":float(conf),
                        "n_pts_seeds":len(pts_seeds),"n_pts_gsce":len(pts_exp),
                        "gsce_iterations":n_iter,"coords_scene":coords,
                        "n_views":info["n_views"],
                        "fp3d_rules": rules, "fp3d_pass": n_pass,
                        "clean_info": info.get("clean_info", {})})
            met = add_svd_volume_to_met(met, pts_exp)
            results.append(met)
            dims = met["obb_dims"]
            log(f"Dechet B-{det_idx} [{classe}] : "
                f"{len(pts_seeds):,}->{len(pts_exp):,} pts | "
                f"{info['n_views']} vues | "
                f"OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
                f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
                f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
                f"ratio={met['volume_ratio']:.2f} | "
                f"score={scores.mean():.3f} | FP3D={n_pass}/6 | "
                f"z_sol={coords['z_m']:.3f}m",2)
        except Exception as e:
            log(f"Dechet B-{det_idx} ERREUR : {e}",2)

    if all_masks: reconstruct_fond(points,colors,all_masks,out_dir)
    log(f"OK Approche B (native) : {len(results)} dechet(s)",1)
    return results


# ───────────────────────────────────────────────────
# APPROCHE C : Frustum Intersection 3D
# ───────────────────────────────────────────────────
def diagnose_c_quality(volume_cm3, n_pts, n_pts_expected=10000,
                        expected_vol=None):
    """
    Diagnostic du resultat Approche C.

    Compare le volume detecte au volume attendu pour decider :
    - 'bruit'  : volume >> attendu -> raffiner par GSCE-C
    - 'maigre' : volume << attendu -> completer par vote SAM
    - 'ok'     : volume coherent  -> garder tel quel

    Args:
        volume_cm3 : volume du candidat C en cm3
        n_pts : nombre de points du candidat
        n_pts_expected : nb pts attendu (proxy)
        expected_vol : volume attendu (defaut EXPECTED_VOLUME_CM3)
    """
    if expected_vol is None:
        expected_vol = EXPECTED_VOLUME_CM3

    ratio_vol = volume_cm3 / max(expected_vol, 1.0)

    if ratio_vol > D_VOLUME_BRUIT_RATIO:
        return "bruit", ratio_vol
    elif ratio_vol < D_VOLUME_MAIGRE_RATIO:
        return "maigre", ratio_vol
    else:
        return "ok", ratio_vol


def run_approche_D(points, colors, camera_data, all_det_sam, all_det,
                    masks_dir, out_dir, scene_origin, scene_R,
                    ground_plane=None, scene_density=0,
                    scene_name="", results_C=None,
                    native_per_image=None):
    """
    Approche D : raffinement adaptatif (seeds natives).

    La strategie "maigre" s appuie sur les seeds natives et le vote 3D
    consensus, ce qui evite de re-introduire du bruit aux bords.

    Part du resultat de l Approche C et le raffine selon diagnostic :
    - Trop de bruit  -> GSCE-C sur les points C comme graines
    - Trop maigre    -> Seeds natives + vote 3D autour du centre C
    - OK             -> Garder C tel quel (mais avec metriques D)

    Cette approche corrige les faiblesses de C (sans SAM = bruit OU
    pauvre selon les scenes) en lui appliquant le raffinement
    contextualise -- toujours sans projection inverse.

    Args:
        results_C : resultats de run_approche_C (liste de dicts)
    """
    log("[APPROCHE D] Raffinement adaptatif (seeds natives)...", 1)

    if results_C is None:
        log("ERREUR : results_C requis pour Approche D", 2)
        return []

    if not results_C:
        log("Approche C n a rien trouve -- D ne peut pas raffiner", 2)
        return []

    if native_per_image is None:
        log("WARN D : native_per_image absent -- strategie 'maigre' degradee", 2)

    results = []
    all_masks = []
    # FIX bug doublons D : un meme det_idx SAM ne doit pas etre
    # reassigne a plusieurs detections C. On garde un set d indices
    # SAM deja utilises et on les exclut dans les iterations suivantes.
    used_sam_indices = set()

    for det_c in results_C:
        det_idx = det_c.get("id", 0)
        classe = det_c.get("classe", "dechet")
        conf = det_c.get("conf", 0.5)
        volume_c = det_c.get("volume_cm3", 0)

        # Charger les points C depuis le PLY
        pts_c_path = f"{out_dir}/dechet_C_{det_idx}_points.ply"
        if not os.path.exists(pts_c_path):
            log(f"Det D-{det_idx} : pts C manquants {pts_c_path}", 2)
            continue

        pts_c = []
        reading = False
        with open(pts_c_path, "r", errors="ignore") as f:
            for line in f:
                if line.strip() == "end_header":
                    reading = True; continue
                if not reading: continue
                p = line.split()
                if len(p) >= 3:
                    pts_c.append([float(p[0]), float(p[1]), float(p[2])])
        pts_c = np.array(pts_c, dtype=np.float32)

        if len(pts_c) < 30:
            log(f"Det D-{det_idx} : trop peu de pts C ({len(pts_c)})", 2)
            continue

        # Diagnostic
        diagnostic, ratio_vol = diagnose_c_quality(
            volume_c, len(pts_c))
        log(f"Det D-{det_idx} [{classe}] : V_C={volume_c:.0f}cm3 "
            f"(ratio {ratio_vol:.2f}x attendu) -> diagnostic='{diagnostic}'", 2)

        if D_MODE == "B":
            strategy = "bruit"
        elif D_MODE == "A":
            strategy = "maigre"
        else:
            strategy = diagnostic

        col = COLORS_DECHETS[len(results) % len(COLORS_DECHETS)]

        # ────────────────────────────────────────────
        # Strategie 1 : "bruit" -> GSCE-C sur pts_c
        # ────────────────────────────────────────────
        if strategy == "bruit":
            log(f"Strategie : GSCE-C sur points C (raffinement geom)", 2)
            cols_c = np.tile([0.5, 0.5, 0.5], (len(pts_c), 1))
            pts_seeds, cols_seeds = filter_seed_outliers(pts_c, cols_c)
            if len(pts_seeds) < 30:
                log(f"Det D-{det_idx} : trop peu apres filtre", 2)
                continue
            # DEBRUITAGE ROBUSTE avant GSCE-C (critique pour D)
            pts_seeds, cols_seeds, _ = clean_seeds_robust(
                pts_seeds, cols_seeds, verbose=False)
            if len(pts_seeds) < 30:
                log(f"Det D-{det_idx} : <30 pts apres clean, skip", 2)
                continue
            log(f"GSCE-C det D-{det_idx} ({len(pts_seeds):,} graines)...", 2)
            pts_d, cols_d, scores, n_iter = gsce_c(
                pts_seeds, points, colors, verbose=True)
            # Post-GSCE clean (analogue B)
            pts_d_clean, cols_d_clean, _ = clean_seeds_robust(
                pts_d, cols_d, verbose=False)
            if len(pts_d_clean) >= 30:
                pts_d, cols_d = pts_d_clean, cols_d_clean
            try:
                mesh, met = poisson_weighted(
                    pts_d, cols_d, scores[:len(pts_d)] if len(scores)>=len(pts_d) else np.ones(len(pts_d)),
                    points, colors)
            except Exception as e:
                log(f"Poisson D ERREUR : {e}", 2); continue

        # ────────────────────────────────────────────
        # Strategie 2 : "maigre" -> Seeds natives + vote 3D consensus
        # ────────────────────────────────────────────
        elif strategy == "maigre":
            log(f"Strategie : Seeds natives + vote 3D autour centre C", 2)
            center_c = pts_c.mean(axis=0)

            if native_per_image is None:
                # Fallback degrade (sans seeds natives)
                log(f"Det D-{det_idx} : pas de native_per_image, skip", 2)
                continue

            # Trouver le det_idx SAM le plus proche du centre C
            # en se basant sur le CENTROID des seeds natives par detection.
            # FIX : exclure les indices SAM deja utilises par une iteration
            # precedente de D (evite que 2 C trouvent la meme SAM).
            max_det = max((len(v.get("detections", []))
                           for v in all_det_sam.values()), default=0)
            best_sam_idx = -1
            best_dist = np.inf
            best_seeds = None
            for try_idx in range(max_det):
                if try_idx in used_sam_indices:
                    continue
                seeds_pv, _, _ = get_native_seeds_for_det(
                    try_idx, all_det_sam, camera_data,
                    native_per_image, masks_dir, scene_name)
                if len(seeds_pv) < 2:
                    continue
                all_s = np.concatenate(list(seeds_pv.values()), axis=0)
                if len(all_s) == 0:
                    continue
                c_s = all_s.mean(axis=0)
                d = np.linalg.norm(c_s - center_c)
                if d < best_dist:
                    best_dist = d
                    best_sam_idx = try_idx
                    best_seeds = seeds_pv
            if best_sam_idx < 0 or best_seeds is None:
                log(f"Det D-{det_idx} : pas de seeds SAM disponibles "
                    f"(deja utilisees : {sorted(used_sam_indices)})", 2)
                continue
            used_sam_indices.add(best_sam_idx)
            log(f"Det SAM proche : idx={best_sam_idx} dist={best_dist:.3f}m", 2)

            # Vote 3D consensus permissif (seuil bas pour le mode 'maigre')
            votes, keep_mask = vote_3d_consensus(
                best_seeds, points, eps=0.02,
                min_views=max(2, int(0.10*len(best_seeds))))
            if int(keep_mask.sum()) < 30:
                log(f"Det D-{det_idx} : vote 3D trop faible", 2); continue
            cands = points[keep_mask]
            cols_cands = colors[keep_mask]
            pts_d, cols_d = get_best_cluster(cands, cols_cands, center_c)
            # DEBRUITAGE ROBUSTE
            pts_d, cols_d, _ = clean_seeds_robust(pts_d, cols_d, verbose=False)
            if len(pts_d) < 30:
                log(f"Det D-{det_idx} : <30 pts apres clean, skip", 2)
                continue
            try:
                mesh, met = poisson_standard(
                    pts_d, cols_d, points, colors)
            except Exception as e:
                log(f"Poisson D ERREUR : {e}", 2); continue

        # ────────────────────────────────────────────
        # Strategie 3 : "ok" -> Reutiliser C avec Poisson
        # ────────────────────────────────────────────
        else:
            log(f"Strategie : C est OK, reutilisation directe", 2)
            cols_c_full = np.tile([0.5, 0.5, 0.5], (len(pts_c), 1))
            pts_d = pts_c
            cols_d = cols_c_full
            # DEBRUITAGE ROBUSTE meme en mode "ok" (C n a pas SAM, donc bruite)
            pts_d, cols_d, _ = clean_seeds_robust(pts_d, cols_d, verbose=False)
            if len(pts_d) < 30:
                log(f"Det D-{det_idx} : <30 pts apres clean ok, skip", 2)
                continue
            try:
                mesh, met = poisson_standard(
                    pts_d, cols_d, points, colors)
            except Exception as e:
                log(f"Poisson D ERREUR : {e}", 2); continue

        # Filtre 3D applique au resultat
        center_scene = transform_to_scene_frame(
            np.array(met["obb_center"])[np.newaxis, :],
            scene_origin, scene_R)[0]
        _, _, view_ratio = count_views_visible(
            np.array(met["obb_center"]), 0.1, camera_data)
        candidate_info = {
            "pts":        pts_d,
            "obb_dims":   met["obb_dims"],
            "obb_center": met["obb_center"],
            "volume_cm3": met["volume_cm3"],
            "z_sol":      float(center_scene[2]),
        }
        is_valid, rules, n_pass = filter_fp_3d_geometric(
            candidate_info, ground_plane, scene_density,
            view_ratio, save_dir=out_dir)

        if not is_valid:
            log(f"Dechet D-{det_idx} REJETE FP3D : {n_pass}/6 | "
                f"{[k for k,v in rules.items() if not v]}", 2)
            continue

        coords = save_dechet_files(
            out_dir, "dechet_D", det_idx, pts_d, cols_d,
            classe, conf, met["volume_cm3"], met,
            points, colors, col, None, all_masks,
            scene_origin, scene_R)
        met.update({
            "id": det_idx, "classe": classe, "conf": float(conf),
            "n_pts": len(pts_d), "coords_scene": coords,
            "d_strategy": strategy, "d_ratio_vol": float(ratio_vol),
            "fp3d_rules": rules, "fp3d_pass": n_pass
        })
        met = add_svd_volume_to_met(met, pts_d)
        results.append(met)
        dims = met["obb_dims"]
        log(f"Dechet D-{det_idx} [{classe}] strategy={strategy} : "
            f"{len(pts_d):,} pts | OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
            f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
            f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
            f"ratio={met['volume_ratio']:.2f} | "
            f"FP3D={n_pass}/6 | z_sol={coords['z_m']:.3f}m", 2)

    log(f"OK Approche D (native) : {len(results)} dechet(s)", 1)
    return results


def run_approche_C(points, colors, camera_data, all_det, out_dir,
                   scene_origin, scene_R,
                   ground_plane=None, scene_density=0,
                   scene_name=""):
    """
    Approche C : Frustum Intersection 3D (sans SAM).

    Projette les bounding boxes YOLOv8 en frustums 3D.
    Vote pondere par la confiance de detection.
    Plus rapide que A et B, ne necessite pas SAM.

    NMS 3D integre via DBSCAN (les clusters = dechets distincts).
    """
    log("[APPROCHE C] Frustum Intersection 3D...", 1)
    N=len(points); votes=np.zeros(N,dtype=np.float32); n_cam=0; all_masks=[]
    for img_name,cam in camera_data.items():
        if img_name not in all_det: continue
        dets=[d for d in all_det[img_name].get("detections",[])
              if d.get("conf",1.0)>=CONF_THRESH_C]
        if not dets: continue
        W_m,H_m=cam["image_size"]; W_o,H_o=all_det[img_name]["image_size"]
        sx=W_m/W_o if W_o!=W_m else 1.0; sy=H_m/H_o if H_o!=H_m else 1.0
        K=np.array(cam["K"]); pose=np.array(cam["pose_4x4"])
        fx=K[0,0]; fy=K[1,1]; cx_=K[0,2]; cy_=K[1,2]
        pts_h=np.hstack([points,np.ones((N,1))])
        pts_cam=(pose@pts_h.T).T[:,:3]
        front=pts_cam[:,2]>0.01; idx=np.where(front)[0]
        if len(idx)==0: continue
        pc=pts_cam[idx]
        u=(pc[:,0]*fx/pc[:,2]+cx_).astype(int)
        v=(pc[:,1]*fy/pc[:,2]+cy_).astype(int)
        in_view=(u>=0)&(u<W_m)&(v>=0)&(v<H_m)
        idx2=idx[in_view]; u2=u[in_view]; v2=v[in_view]
        for det in dets:
            x1,y1,x2,y2=det["bbox_2d"]
            x1m,y1m=int(x1*sx),int(y1*sy); x2m,y2m=int(x2*sx),int(y2*sy)
            if x1m>=x2m or y1m>=y2m: continue
            in_b=(u2>=x1m)&(u2<=x2m)&(v2>=y1m)&(v2<=y2m)
            np.add.at(votes,idx2[in_b],det.get("conf",1.0))
        n_cam+=1
    thresh=n_cam*VOTE_RATIO_C
    n_cands=int((votes>=thresh).sum())
    log(f"Cameras={n_cam} | seuil={thresh:.1f} | candidats={n_cands:,}",2)
    if thresh<=0 or n_cands<30:
        log("Pas assez de candidats -- essaie --vote_ratio_c 0.1",2); return []
    cands=points[votes>=thresh]; cols_c=colors[votes>=thresh]
    from sklearn.cluster import DBSCAN
    db=DBSCAN(eps=DBSCAN_EPS*1.5,min_samples=DBSCAN_MIN_SAMPLES).fit(cands)
    labels=db.labels_; unique=set(labels)-{-1}
    log(f"Clusters detectes : {len(unique)}",2)
    results=[]
    for i,lbl in enumerate(unique):
        mask=labels==lbl; pts_d=cands[mask]; col_d=cols_c[mask]
        if len(pts_d)<30: continue
        col=COLORS_DECHETS[i%len(COLORS_DECHETS)]
        classe,conf=get_classe_from_det(all_det,i)
        try:
            mesh,met=poisson_standard(pts_d,col_d,points,colors)

            # ── FILTRE 3D GEOMETRIQUE ──
            center_scene = transform_to_scene_frame(
                np.array(met["obb_center"])[np.newaxis,:],
                scene_origin, scene_R)[0]
            _,_,view_ratio = count_views_visible(
                np.array(met["obb_center"]), 0.1, camera_data)
            candidate_info = {
                "pts":        pts_d,
                "obb_dims":   met["obb_dims"],
                "obb_center": met["obb_center"],
                "volume_cm3": met["volume_cm3"],
                "z_sol":      float(center_scene[2]),
            }
            is_valid, rules, n_pass = filter_fp_3d_geometric(
                candidate_info, ground_plane, scene_density,
                view_ratio, save_dir=out_dir)
            if not is_valid:
                log(f"Dechet C-{i} REJETE (FP3D) : "
                    f"{n_pass}/6 | {[k for k,v in rules.items() if not v]}", 2)
                continue
            coords=save_dechet_files(out_dir,"dechet_C",i,
                pts_d,col_d,classe,conf,met["volume_cm3"],met,
                points,colors,col,None,all_masks,scene_origin,scene_R)
            met.update({"id":i,"classe":classe,"conf":float(conf),
                        "n_pts":len(pts_d),"coords_scene":coords,
                        "fp3d_rules": rules, "fp3d_pass": n_pass})
            met = add_svd_volume_to_met(met, pts_d)
            results.append(met)
            dims=met["obb_dims"]
            log(f"Dechet C-{i} [{classe}] : {len(pts_d):,} pts | "
                f"OBB {dims[0]:.3f}x{dims[1]:.3f}x{dims[2]:.3f}m | "
                f"V_SVD={met['volume_svd_cm3']:.0f}cm3 | "
                f"V_Poisson={met['volume_poisson_cm3']:.0f}cm3 | "
                f"ratio={met['volume_ratio']:.2f} | "
                f"FP3D={n_pass}/6 | z_sol={coords['z_m']:.3f}m",2)
        except Exception as e:
            log(f"Dechet C-{i} ERREUR : {e}",2)
    if all_masks: reconstruct_fond(points,colors,all_masks,out_dir)
    log(f"OK Approche C : {len(results)} dechet(s)",1)
    return results


# ───────────────────────────────────────────────────
# ZIP POUR MAC M4
# ───────────────────────────────────────────────────
def zip_scene_for_mac(scene_name, out_dir, zip_dir):
    """
    Zippe les fichiers intermediaires pour pipeline_mac.py (si utilise).
    Inclut : PLY coarse, poses cameras, detections, masques SAM.
    """
    os.makedirs(zip_dir,exist_ok=True)
    zip_path=f"{zip_dir}/{scene_name}_mac.zip"
    masks_dir=f"{out_dir}/masks"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for fname in ["pointcloud_coarse.ply","camera_poses.json",
                      "detections.json","detections_seg.json"]:
            fpath=f"{out_dir}/{fname}"
            if os.path.exists(fpath): zf.write(fpath,fname)
        if os.path.exists(masks_dir):
            masks=[f for f in os.listdir(masks_dir) if f.endswith(".png")]
            for mask in sorted(masks):
                zf.write(f"{masks_dir}/{mask}",f"masks/{mask}")
    size_mb=os.path.getsize(zip_path)/(1024**2)
    log(f"ZIP : {zip_path} ({size_mb:.1f} MB)",2)


def scene_already_done(out_dir):
    """
    Verifie si une scene a deja ete completement traitee.
    Critere : les 4 fichiers MASt3R existent + rapport.json.
    """
    required=["pointcloud_coarse.ply","camera_poses.json",
              "detections.json","detections_seg.json","rapport.json"]
    return all(os.path.exists(f"{out_dir}/{f}") for f in required)


# ───────────────────────────────────────────────────
# PIPELINE PAR SCENE
# ───────────────────────────────────────────────────
def process_scene(scene_name, images_dir, approach,
                  skip_done=False,
                  skip_mast3r=False,
                  skip_detection=False):
    """
    Traite une scene complete : YOLOv8 + SAM + MASt3R + Approches.

    Arguments :
    - scene_name  : nom de la scene (ex: 'scene00')
    - images_dir  : dossier contenant les images JPG
    - approach    : 'A', 'B', 'C', ou 'all'
    - skip_done   : si True, saute les scenes deja traitees

    Sorties dans OUT_DIR/scene_name/ :
    - pointcloud_coarse.ply
    - camera_poses.json
    - detections.json + detections_seg.json
    - masks/*.png
    - fond_mesh.ply
    - dechet_A_N_*.ply (si approche A ou all)
    - dechet_B_N_*.ply (si approche B ou all)
    - dechet_C_N_*.ply (si approche C ou all)
    - rapport.json (toutes les metriques)
    - annotation.json (format lisible pour le robot)
    - scene_name_mac.zip (fichiers intermediaires)
    """
    out_dir=f"{OUT_DIR}/{scene_name}"; os.makedirs(out_dir,exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SCENE : {scene_name} | APPROCHE : {approach}")
    if skip_detection: print(f"  [SKIP] YOLOv8 + SAM + MASt3R -- tout depuis disque")
    elif skip_mast3r:  print(f"  [SKIP] MASt3R -- relance YOLOv8 + SAM")
    print(f"{'='*60}")

    img_files=[f for f in Path(images_dir).iterdir() if f.suffix in IMG_EXT]
    if not img_files:
        log(f"ERREUR : aucune image dans {images_dir}"); return None

    if skip_done and scene_already_done(out_dir):
        log("Scene deja traitee -- skip"); return None

    t0=time.time()
    try:
        # ── Detection YOLOv8 + SAM ────────────────────────────────
        if skip_detection:
            # Charger depuis disque
            for fname in ["detections.json","detections_seg.json"]:
                if not os.path.exists(f"{out_dir}/{fname}"):
                    log(f"ERREUR : {fname} manquant -- lance sans --skip_detection")
                    return None
            with open(f"{out_dir}/detections.json") as f:
                all_det=json.load(f)
            with open(f"{out_dir}/detections_seg.json") as f:
                all_det_sam=json.load(f)
            all_files=sorted([str(f) for f in Path(images_dir).iterdir()
                               if f.suffix in IMG_EXT])
            all_files=subsample_images(all_files)
            log(f"Detection chargee depuis disque | {len(all_det)} images "
                f"| {len(all_files)} all_files (MAX_IMAGES={MAX_IMAGES})",2)
        else:
            # Relancer YOLOv8 + SAM
            all_det, all_files = run_detection(images_dir, out_dir)

            # ── NMS 2D intra-image ──
            # Si 2 detections dans la meme image ont IoU > 80%,
            # garde celle avec la meilleure confiance YOLO
            all_det, nms_intra_stats = nms_2d_intra_image(all_det)
            log(f"NMS intra : -{nms_intra_stats['n_rejected']} doublons intra", 1)

            # ── FILTRE 2D MULTI-VUES ──
            # Elimine les detections isolees (artefacts texturaux locaux)
            all_det_filtered, fp2d_stats = filter_fp_2d_multiview(all_det)
            log(f"Filtre 2D : -{fp2d_stats['n_rejected']} FP rejetes", 1)
            with open(f"{out_dir}/detections_filtered.json", "w") as f:
                json.dump(all_det_filtered, f, indent=2)
            all_det_sam = run_sam(all_files, all_det_filtered, out_dir)
            all_det = all_det_filtered

        # ── Reconstruction MASt3R ─────────────────────────────────
        if skip_mast3r or skip_detection:
            # Charger depuis disque
            for fname in ["pointcloud_coarse.ply","camera_poses.json"]:
                if not os.path.exists(f"{out_dir}/{fname}"):
                    log(f"ERREUR : {fname} manquant -- lance sans --skip_mast3r")
                    return None
            log("Chargement pointcloud_coarse.ply...",1)
            pts=[]; cols=[]; reading=False
            with open(f"{out_dir}/pointcloud_coarse.ply","r",errors="ignore") as f:
                for line in f:
                    if line.strip()=="end_header": reading=True; continue
                    if not reading: continue
                    p=line.split()
                    if len(p)>=6:
                        pts.append([float(p[0]),float(p[1]),float(p[2])])
                        cols.append([int(p[3])/255,int(p[4])/255,int(p[5])/255])
            points=np.array(pts,dtype=np.float32)
            colors=np.array(cols,dtype=np.float32)
            with open(f"{out_dir}/camera_poses.json") as f:
                cam_data=json.load(f)
            # Charger les pts3d natifs si dispo (necessaire pour S0)
            native_per_image=None
            npz_path=f"{out_dir}/native_pts3d.npz"
            if os.path.exists(npz_path):
                z=np.load(npz_path)
                native_per_image={}
                names=sorted({k.split("__")[0] for k in z.files})
                for n in names:
                    if f"{n}__pts" in z.files and f"{n}__valid" in z.files:
                        native_per_image[n]={
                            "pts3d_HW3": z[f"{n}__pts"],
                            "valid_HW":  z[f"{n}__valid"],
                            "H": z[f"{n}__pts"].shape[0],
                            "W": z[f"{n}__pts"].shape[1],
                        }
                log(f"OK native_pts3d.npz charge ({len(native_per_image)} vues)",2)
            log(f"OK {len(points):,} points | {len(cam_data)} cameras charges",2)
        else:
            # Relancer MASt3R
            points,colors,cam_data,native_per_image=run_mast3r(images_dir,all_files,out_dir)

        # Nuage fond sous-echantillonne pour visualisation
        fp=points[::FOND_SUBSAMPLE]; fc=(colors[::FOND_SUBSAMPLE]*255).astype(np.uint8)
        write_ply(f"{out_dir}/scene_complete.ply",fp,fc)

        # ── PRE-CALCULS POUR FILTRE 3D ──
        # Plan sol pour regle 1 (coplanarite)
        ground_plane = estimate_ground_plane(points)
        if ground_plane is not None:
            log(f"Plan sol estime : normale=({ground_plane[0][0]:.3f}, "
                f"{ground_plane[0][1]:.3f}, {ground_plane[0][2]:.3f})", 1)
        else:
            log("WARNING : plan sol non estime -- regle 1 desactivee", 1)
        # Densite scene pour regle 4
        scene_density = compute_scene_density(points)
        log(f"Densite scene : {scene_density:.0f} pts/m3", 1)

        # Repere normalise
        scene_origin,scene_R,_=compute_scene_frame(points)
        log(f"Repere scene : origine=({scene_origin[0]:.3f},"
            f"{scene_origin[1]:.3f},{scene_origin[2]:.3f})",1)

        # ── PRIORITE 1 : suppression du sol au niveau scene ──
        # Nuage SANS sol pour A/B/D. Utilise le REPERE NORMALISE
        # (scene_origin, scene_R) ou Z = hauteur reelle, avec seuil
        # RELATIF + garde-fou (ne detruit jamais > 90% du nuage).
        points_ng, colors_ng, _ = remove_ground_from_scene(
            points, colors, ground_plane,
            scene_origin=scene_origin, scene_R=scene_R, verbose=True)

        masks_dir=f"{out_dir}/masks"
        rapport={"scene":scene_name,"approach":approach,
                 "n_points_total":int(len(points)),
                 "dechets_S0":[],"dechets_A":[],"dechets_B":[],
                 "dechets_C":[],"dechets_D":[],"dechets_E":[]}

        # Approches
        # E (Coastal-Waste-3D) = methode principale, utilise le nuage COMPLET
        # (fait sa propre separation sol en E3 en repere normalise).
        if approach in ["E","all"]:
            rapport["dechets_E"]=run_approche_E(
                points,colors,cam_data,all_det_sam,all_det,
                masks_dir,out_dir,scene_origin,scene_R,
                native_per_image,
                ground_plane, scene_density, scene_name)

        # S0 utilise le nuage complet (pts3d natifs par image).
        if approach in ["S0","all"]:
            rapport["dechets_S0"]=run_approche_S0(
                points,colors,cam_data,all_det_sam,all_det,
                masks_dir,out_dir,scene_origin,scene_R,
                native_per_image,
                ground_plane, scene_density, scene_name)

        # A/B/C/D travaillent sur le nuage SANS sol (points_ng) pour
        # eviter l import de la galette de fond.
        if approach in ["A","all"]:
            rapport["dechets_A"]=run_approche_A(
                points_ng,colors_ng,cam_data,all_det_sam,all_det,
                masks_dir,out_dir,scene_origin,scene_R,
                ground_plane, scene_density, scene_name,
                native_per_image=native_per_image)

        if approach in ["B","all"]:
            rapport["dechets_B"]=run_approche_B(
                points_ng,colors_ng,cam_data,all_det_sam,all_det,
                masks_dir,out_dir,scene_origin,scene_R,
                ground_plane, scene_density, scene_name,
                native_per_image=native_per_image)

        if approach in ["C","all","D"]:
            rapport["dechets_C"]=run_approche_C(
                points_ng,colors_ng,cam_data,all_det,
                out_dir,scene_origin,scene_R,
                ground_plane, scene_density, scene_name)

        if approach in ["D","all"]:
            rapport["dechets_D"]=run_approche_D(
                points_ng,colors_ng,cam_data,all_det_sam,all_det,
                masks_dir,out_dir,scene_origin,scene_R,
                ground_plane, scene_density, scene_name,
                results_C=rapport["dechets_C"],
                native_per_image=native_per_image)

        rapport["temps_sec"]=round(time.time()-t0,1)

        # Sauvegarder rapport.json
        with open(f"{out_dir}/rapport.json","w") as f:
            json.dump(rapport,f,indent=2,default=str)

        # Annotation lisible pour le robot
        # E (Coastal-Waste-3D) en priorite car methode principale.
        dechets_info=[]
        for approche,key in [("E","dechets_E"),("D","dechets_D"),
                              ("B","dechets_B"),("A","dechets_A"),
                              ("C","dechets_C"),("S0","dechets_S0")]:
            for d in rapport.get(key,[]):
                if not any(x["id"]==d["id"] for x in dechets_info):
                    dechets_info.append({
                        "id":          d["id"],
                        "classe":      d.get("classe","?"),
                        "confiance":   f"{d.get('conf',0):.0%}",
                        "approche":    approche,
                        "position":    d.get("coords_scene",{}),
                        "dimensions":  {
                            "longueur_m": round(d.get("obb_dims",[0,0,0])[0],4),
                            "largeur_m":  round(d.get("obb_dims",[0,0,0])[1],4),
                            "hauteur_m":  round(d.get("obb_dims",[0,0,0])[2],4),
                        },
                        "volume_cm3":  round(d.get("volume_cm3",0),1),
                        "surface_cm2": round(d.get("surface_cm2",0),1),
                        "score_gsce":  round(d.get("score_moyen",0),3),
                    })
        write_annotation(f"{out_dir}/annotation.json",scene_name,dechets_info)

        # ZIP intermediaire
        zip_scene_for_mac(scene_name,out_dir,f"{OUT_DIR}/zips")

        # Resume
        elapsed=round(time.time()-t0,1)
        log(f"Scene {scene_name} OK en {elapsed}s",1)
        print(f"\n  Dechets detectes :")
        for d in dechets_info:
            p=d["position"]
            print(f"    [{d['approche']}] {d['classe']} {d['confiance']} | "
                  f"x={p.get('x_m',0):.3f}m y={p.get('y_m',0):.3f}m "
                  f"z={p.get('z_m',0):.3f}m | V={d['volume_cm3']:.1f}cm3")

        return rapport

    except Exception as e:
        log(f"ERREUR : {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        clear_gpu()


# ───────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser(
        description="Coastal-Waste-3D -- detection et segmentation 3D de dechets cotiers")
    parser.add_argument("--scene",      default=None,
        help="Traiter une seule scene (ex: scene00)")
    parser.add_argument("--approach",   default="all",
        choices=["S0","A","B","C","D","E","all"],
        help="Approche(s) a lancer (defaut: all). E=Coastal-Waste-3D (principale)")
    parser.add_argument("--skip_done",      action="store_true",
        help="Sauter les scenes deja completement traitees")
    parser.add_argument("--skip_mast3r",    action="store_true",
        help="Ne pas relancer MASt3R -- charger pointcloud_coarse.ply "
             "et camera_poses.json depuis le disque")
    parser.add_argument("--skip_detection", action="store_true",
        help="Ne pas relancer YOLOv8 + SAM -- charger detections.json, "
             "detections_seg.json et masks/ depuis le disque. "
             "Implique aussi --skip_mast3r.")
    args=parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PIPELINE LIGHTNING AI v2 -- COMPLET")
    print(f"  Author")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Approche : {args.approach}")
    print(f"{'='*60}")

    install_deps(); setup_mast3r(); download_sam()
    os.makedirs(OUT_DIR,exist_ok=True)
    os.makedirs(f"{OUT_DIR}/zips",exist_ok=True)

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            name=torch.cuda.get_device_name(0)
            mem=torch.cuda.get_device_properties(0).total_memory//(1024**3)
            log(f"GPU : {name} ({mem} GB)")
    except: pass

    # Trouver les scenes
    scenes_root=Path(SCENES_DIR)
    if args.scene:
        scene_dir=scenes_root/args.scene
        if not scene_dir.exists():
            log(f"ERREUR : {args.scene} introuvable"); return
        scenes=[scene_dir]
    else:        scenes=sorted([d for d in scenes_root.iterdir()
                       if d.is_dir() and any(f.suffix in IMG_EXT
                                             for f in d.iterdir())])

    log(f"Scenes : {len(scenes)}")
    for s in scenes:
        n=len([f for f in s.iterdir() if f.suffix in IMG_EXT])
        done=" [DEJA FAIT]" if scene_already_done(f"{OUT_DIR}/{s.name}") else ""
        log(f"  -> {s.name} ({n} images){done}")

    rapport_global={}; t_total=time.time()

    for i,scene_dir in enumerate(scenes):
        scene_name=scene_dir.name
        print(f"\n[{i+1}/{len(scenes)}] {scene_name}...")
        r=process_scene(scene_name, str(scene_dir), args.approach,
                        args.skip_done,
                        args.skip_mast3r,
                        args.skip_detection)
        rapport_global[scene_name]="OK" if r else "ERREUR"

    # Rapport final
    elapsed=round(time.time()-t_total,1)
    print(f"\n{'='*60}")
    print(f"  RAPPORT FINAL -- {elapsed}s | {args.approach}")
    print(f"{'='*60}")
    for sn,status in rapport_global.items():
        print(f"  [{'OK' if status=='OK' else '!!'}] {sn}")

    # ZIP global
    zip_global=f"{OUT_DIR}/toutes_scenes_resultats.zip"
    zips_dir=f"{OUT_DIR}/zips"
    if os.path.exists(zips_dir):
        with zipfile.ZipFile(zip_global,"w",zipfile.ZIP_DEFLATED) as zf:
            for zp in sorted(Path(zips_dir).glob("*.zip")):
                zf.write(str(zp),zp.name)
        size_mb=os.path.getsize(zip_global)/(1024**2)
        print(f"\n  ZIP global : {zip_global} ({size_mb:.1f} MB)")
    print(f"{'='*60}\n")


if __name__=="__main__":
    main()
