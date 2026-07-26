"""
Mini-test ciblé — Isoler l'effet du consensus multi-vues (E2).

MOTIVATION : dans l'ablation, le Groupe B (E2 off/vote3D/consensus)
montre des résultats proches. Hypothèse : le nettoyage E4 (DBSCAN) en
AVAL masque l'effet de E2 — même si E2=off laisse du bruit, DBSCAN le
retire ensuite.

TEST : croiser E2 (3 variantes) × E4-DBSCAN (on/off) = 6 configs.
  - Si E2 a un effet visible SANS DBSCAN mais pas AVEC -> hypothèse confirmée :
    "le consensus retire le bruit en amont, redondant avec DBSCAN sur le
    volume mais important pour la précision".
  - Si E2 n'a d'effet dans aucun cas -> le consensus n'apporte rien de
    mesurable sur ces métriques (à investiguer via IoU/GT).

Usage (noeud de calcul) :
    python -m coastal_waste_3d.minitest_e2 --scenes scene07 scene10 \
        --pipeline ./pipeline.py \
        --base_out ../outputs \
        --ablation_out ../outputs_ablation
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import List

from .config import PipelineConfig, Method, E2Variant
from .ablation import reference_config
from .runner import run_experiment, aggregate_results, dechets_key_for


def make_minitest_configs() -> List[PipelineConfig]:
    """
    Génère les 6 configs : E2 {off, vote3D, consensus} × E4-DBSCAN {on, off}.

    Tout le reste reste au défaut (E1=native, E3=soft, E5=poisson_std,
    E6=full). Seuls E2 et le flag DBSCAN varient — isolation propre.
    """
    base = reference_config()
    configs = []
    e2_variants = [
        ("off", E2Variant.OFF),
        ("vote3D", E2Variant.VOTE_3D),
        ("consensus", E2Variant.CONSENSUS),
    ]
    for dbscan_on in (True, False):
        for tag, variant in e2_variants:
            cfg = copy.deepcopy(base)
            db_tag = "dbON" if dbscan_on else "dbOFF"
            cfg.name = f"MT_E2{tag}_{db_tag}"
            cfg.e2.variant = variant
            cfg.e4.dbscan = dbscan_on
            # On laisse plane/compact au défaut (True) pour ne changer
            # qu'une variable structurante : le composant DBSCAN.
            configs.append(cfg)
    return configs


def main():
    ap = argparse.ArgumentParser(description="Mini-test effet du consensus E2")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Scènes (privilégier les multi-objets : scene07 scene10)")
    ap.add_argument("--pipeline",
                    default="pipeline.py")
    ap.add_argument("--base_out",
                    default=os.environ.get("COASTAL_OUTPUTS",
                                           "../outputs"))
    ap.add_argument("--ablation_out",
                    default=os.environ.get("COASTAL_ABLATION",
                                           "../outputs_ablation"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    configs = make_minitest_configs()
    print(f"═══ Mini-test E2 : {len(configs)} configs × "
          f"{len(args.scenes)} scènes = "
          f"{len(configs)*len(args.scenes)} runs ═══\n")
    for c in configs:
        print(f"  {c.name:24s} | E2={c.e2.variant.value:9s} "
              f"DBSCAN={c.e4.dbscan}")

    if args.dry_run:
        return

    pipeline_path = Path(args.pipeline).resolve()
    base_out = Path(args.base_out)
    ablation_root = Path(args.ablation_out)
    ablation_root.mkdir(parents=True, exist_ok=True)

    print()
    for i, cfg in enumerate(configs):
        for scene in args.scenes:
            print(f"[{i+1}/{len(configs)}] {cfg.name} × {scene}")
            r = run_experiment(cfg, scene, pipeline_path,
                               base_out, ablation_root)
            print(f"    status={r['status']} | {r['elapsed_sec']:.0f}s")

    # Agrégation des 6 configs
    agg = aggregate_results(ablation_root, args.scenes, configs)

    # Tableau croisé lisible
    print(f"\n{'Config':<24} {'n_obj':>6} {'V_SVD':>9} {'ratio':>7} "
          f"{'sphér':>7} {'planar':>7}")
    print("─" * 70)
    for name, data in agg.items():
        m = data["metrics"]
        g = lambda k: m.get(k, {}).get("mean")
        f = lambda x, d=0: (f"{x:.{d}f}" if x is not None else "-")
        print(f"{name:<24} {data['n_objects']:>6} "
              f"{f(g('volume_svd_cm3')):>9} {f(g('volume_ratio'),2):>7} "
              f"{f(g('metrics_sphericity'),3):>7} "
              f"{f(g('metrics_planarity'),3):>7}")

    out = ablation_root / "minitest_e2_results.json"
    with open(out, "w") as fp:
        json.dump({"scenes": args.scenes, "configs": agg}, fp, indent=2)
    print(f"\n═══ Sauvegardé : {out} ═══")

    # Aide à la lecture
    print("\n── Comment lire ──")
    print("Compare les paires E2off/E2consensus dans chaque régime DBSCAN :")
    print("  • dbOFF : si consensus << off en V_SVD/ratio -> le consensus")
    print("            retire bien le bruit en amont (effet réel).")
    print("  • dbON  : si les deux se ressemblent -> DBSCAN masque l'effet.")
    print("Si effet net en dbOFF mais pas en dbON => hypothèse confirmée.")


if __name__ == "__main__":
    main()
