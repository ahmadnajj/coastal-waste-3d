"""
Runner d'ablation Coastal-Waste-3D.

Exécute les expériences d'ablation en pilotant le pipeline via les
variables d'environnement (env_overrides de la config), puis agrège
les métriques. Inclut le calcul statistique (moyenne, écart-type, IC bootstrap,
test de Wilcoxon).

Conçu pour tourner sur un cluster SLURM ou en local.

Usage :
    python -m coastal_waste_3d.runner --group all --scenes scene00 scene01 scene02
    python -m coastal_waste_3d.runner --group B --scenes scene00   # juste le groupe consensus
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import PipelineConfig, Method, E1Variant
from .ablation import generate_all_experiments, generate_group


# ─────────────────────────────────────────────────────────────────────
#  Résolution de l'approche réelle du pipeline
# ─────────────────────────────────────────────────────────────────────

def approach_for(cfg: PipelineConfig) -> str:
    """
    Détermine quel `--approach` du pipeline exécuter pour une config.

    Cas particulier : la variante frustum de E1 est scène-niveau et
    correspond à l'approche C native (pas à E avec un interrupteur).
    On la route donc vers --approach C.
    """
    if cfg.method == Method.E and cfg.e1.variant == E1Variant.FRUSTUM:
        return "C"
    return cfg.method.value


def dechets_key_for(cfg: PipelineConfig) -> str:
    """Clé du rapport.json à lire selon l'approche réellement exécutée."""
    return f"dechets_{approach_for(cfg)}"


# ─────────────────────────────────────────────────────────────────────
#  Exécution d'une expérience
# ─────────────────────────────────────────────────────────────────────

def run_experiment(cfg: PipelineConfig, scene: str,
                   pipeline_path: Path, base_out: Path,
                   ablation_root: Path) -> dict:
    """
    Exécute une expérience (config × scène) en pilotant le pipeline.

    Le pipeline est lancé en sous-processus avec les variables
    d'environnement dérivées de la config. Les sorties sont collectées
    dans ablation_root/<scene>/<cfg.name>/.

    Returns:
        dict avec statut, durée, chemin du rapport.
    """
    run_dir = ablation_root / scene / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder la config de l'expérience (traçabilité)
    cfg.to_json(run_dir / "config.json")

    # Environnement = base + overrides de la config
    env = os.environ.copy()
    env.update(cfg.env_overrides())
    env["SKIP_INSTALL_DEPS"] = "1"
    env["SKIP_NUMPY_FIX"] = "1"
    # La méthode (S0 ou E) devient l'approche du pipeline.
    # Cas frustum (G1) : routé vers l'approche C native.
    approach = approach_for(cfg)

    # Purge du cache scène : garantit que la collecte ci-dessous ne
    # ramasse que les sorties du run courant (aucun .ply périmé laissé
    # par une exécution précédente avec une autre config/approche).
    for old in (base_out / scene).glob("dechet_*_points.ply"):
        try:
            old.unlink()
        except OSError:
            pass

    cmd = [
        sys.executable, str(pipeline_path),
        "--scene", scene,
        "--approach", approach,
        "--skip_mast3r", "--skip_detection",
    ]

    t0 = time.time()
    log_path = run_dir / "run.log"
    with open(log_path, "w") as lf:
        ret = subprocess.run(cmd, env=env, stdout=lf,
                             stderr=subprocess.STDOUT,
                             cwd=str(pipeline_path.parent.parent))
    elapsed = time.time() - t0

    # Récupérer le rapport produit par le pipeline (dans le cache scène)
    rapport_src = base_out / scene / "rapport.json"
    rapport_dst = run_dir / "rapport.json"
    if rapport_src.exists():
        import shutil
        shutil.copy2(rapport_src, rapport_dst)
        # Copier aussi les PLY pour évaluation GT ultérieure
        for ply in (base_out / scene).glob("dechet_*_points.ply"):
            shutil.copy2(ply, run_dir / ply.name)

    return {
        "config": cfg.name, "scene": scene,
        "status": ret.returncode, "elapsed_sec": elapsed,
        "rapport": str(rapport_dst) if rapport_dst.exists() else None,
    }


# ─────────────────────────────────────────────────────────────────────
#  Agrégation statistique
# ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """
    Intervalle de confiance par bootstrap (adapté aux petits échantillons).

    Rééchantillonne avec remise n_boot fois, retourne les percentiles
    alpha/2 et 1-alpha/2 de la distribution des moyennes.
    """
    if len(values) < 2:
        v = float(values[0]) if len(values) == 1 else 0.0
        return v, v
    rng = np.random.default_rng(42)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def wilcoxon_vs_reference(ref_values: np.ndarray,
                          test_values: np.ndarray) -> Optional[float]:
    """
    Test de Wilcoxon apparié (non paramétrique).

    Compare la config de référence à une ablation. Retourne la p-value,
    ou None si scipy absent ou échantillon trop petit.

    Note : avec n=5 scènes, la puissance est faible. À interpréter avec
    prudence (cf. limites du plan d'ablation).
    """
    try:
        from scipy.stats import wilcoxon
        if len(ref_values) < 3 or len(test_values) < 3:
            return None
        diff = ref_values - test_values
        if np.allclose(diff, 0):
            return 1.0
        _, p = wilcoxon(ref_values, test_values)
        return float(p)
    except Exception:
        return None


def aggregate_results(ablation_root: Path, scenes: List[str],
                      configs: List[PipelineConfig]) -> dict:
    """
    Agrège les métriques de toutes les expériences sur toutes les scènes.

    Lit les rapport.json, extrait les métriques par déchet, calcule les
    statistiques (moyenne, écart-type, IC) par configuration.

    Returns:
        dict {config_name: {metric: {mean, std, ci_lo, ci_hi, n}}}
    """
    # Métriques extraites de chaque déchet du rapport
    metric_keys = [
        "volume_svd_cm3", "volume_poisson_cm3", "volume_ratio",
        "metrics_sphericity", "metrics_planarity", "metrics_bbox_fill",
        "metrics_hull_ratio", "metrics_n_pts", "metrics_density_pts_cm3",
        "metrics_linearity", "metrics_aspect_ratio", "metrics_flatness",
    ]

    per_config: dict = {}
    for cfg in configs:
        # Collecter les valeurs sur toutes les scènes
        collected = {k: [] for k in metric_keys}
        n_objects = 0
        for scene in scenes:
            rpath = ablation_root / scene / cfg.name / "rapport.json"
            if not rpath.exists():
                continue
            with open(rpath) as f:
                rap = json.load(f)
            # Déchets de la méthode réellement exécutée (E, S0, ou C pour frustum)
            key = dechets_key_for(cfg)
            for d in rap.get(key, []):
                n_objects += 1
                for mk in metric_keys:
                    if mk in d and d[mk] is not None:
                        collected[mk].append(float(d[mk]))
        # Statistiques
        stats = {}
        for mk, vals in collected.items():
            arr = np.array(vals)
            if len(arr) == 0:
                stats[mk] = {"mean": None, "std": None, "n": 0}
                continue
            ci_lo, ci_hi = bootstrap_ci(arr)
            stats[mk] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "n": len(arr),
            }
        per_config[cfg.name] = {
            "n_objects": n_objects,
            "method": cfg.method.value,
            "metrics": stats,
        }
    return per_config


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Runner d'ablation Coastal-Waste-3D")
    ap.add_argument("--group", default="all",
                    help="Groupe d'ablation (A-G) ou 'all'")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Scènes à traiter (caches MASt3R requis)")
    ap.add_argument("--pipeline",
                    default="pipeline.py",
                    help="Chemin du pipeline principal")
    ap.add_argument("--base_out",
                    default=os.environ.get("COASTAL_OUTPUTS",
                                           "../outputs"))
    ap.add_argument("--ablation_out",
                    default=os.environ.get("COASTAL_ABLATION",
                                           "../outputs_ablation"))
    ap.add_argument("--dry_run", action="store_true",
                    help="Affiche les expériences sans les exécuter")
    args = ap.parse_args()

    configs = (generate_all_experiments() if args.group == "all"
               else generate_group(args.group))

    print(f"═══ Ablation : {len(configs)} configs × "
          f"{len(args.scenes)} scènes = "
          f"{len(configs)*len(args.scenes)} runs ═══")

    if args.dry_run:
        for c in configs:
            print(f"  {c.name}")
        return

    pipeline_path = Path(args.pipeline).resolve()
    base_out = Path(args.base_out)
    ablation_root = Path(args.ablation_out)
    ablation_root.mkdir(parents=True, exist_ok=True)

    results = []
    for i, cfg in enumerate(configs):
        for scene in args.scenes:
            print(f"[{i+1}/{len(configs)}] {cfg.name} × {scene}")
            r = run_experiment(cfg, scene, pipeline_path,
                               base_out, ablation_root)
            results.append(r)
            print(f"    status={r['status']} | {r['elapsed_sec']:.0f}s")

    # Agrégation
    agg = aggregate_results(ablation_root, args.scenes, configs)
    agg_path = ablation_root / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n═══ Métriques agrégées : {agg_path} ═══")

    # Résumé console
    print(f"\n{'Config':<24} {'n_obj':>6} {'V_SVD':>10} {'ratio':>8} {'sphér.':>8}")
    for name, data in agg.items():
        m = data["metrics"]
        v = m.get("volume_svd_cm3", {}).get("mean")
        r = m.get("volume_ratio", {}).get("mean")
        s = m.get("metrics_sphericity", {}).get("mean")
        vs = f"{v:.0f}" if v else "-"
        rs = f"{r:.2f}" if r else "-"
        ss = f"{s:.3f}" if s else "-"
        print(f"{name:<24} {data['n_objects']:>6} {vs:>10} {rs:>8} {ss:>8}")


if __name__ == "__main__":
    main()
