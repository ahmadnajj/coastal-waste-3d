"""
Générateur des configurations d'ablation.

Produit automatiquement les 25 expériences du plan d'ablation,
organisées par groupe (A à G). Chaque expérience est une PipelineConfig
dérivée de la config de référence par modification d'un seul module.

Usage :
    from coastal_waste_3d.ablation import generate_all_experiments
    configs = generate_all_experiments()
    for cfg in configs:
        print(cfg.name, cfg.to_dict())
"""
from __future__ import annotations

import copy
from typing import List

from .config import (
    PipelineConfig, Method,
    E1Variant, E2Variant, E3Variant, E5Variant, E6Variant,
)


def reference_config() -> PipelineConfig:
    """Configuration de référence (tous modules au défaut)."""
    return PipelineConfig(name="ref_E_full", method=Method.E)


def _derive(base: PipelineConfig, name: str, **mutations) -> PipelineConfig:
    """
    Dérive une nouvelle config en modifiant des champs imbriqués.

    mutations : clés de la forme "e2.variant" → valeur.
    """
    cfg = copy.deepcopy(base)
    cfg.name = name
    for path, value in mutations.items():
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return cfg


# ─────────────────────────────────────────────────────────────────────
#  Groupes d'expériences (A à G)
# ─────────────────────────────────────────────────────────────────────

def group_A_baseline(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe A — baseline S0 vs méthode complète E."""
    return [
        _derive(base, "A0_S0_baseline", method=Method.S0),
        _derive(base, "A1_E_full"),  # référence
    ]


def group_B_consensus(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe B — effet du consensus multi-vues (E2)."""
    return [
        _derive(base, "B0_E2_off", **{"e2.variant": E2Variant.OFF}),
        _derive(base, "B1_E2_vote3D", **{"e2.variant": E2Variant.VOTE_3D}),
        _derive(base, "B2_E2_consensus", **{"e2.variant": E2Variant.CONSENSUS}),
    ]


def group_C_ground(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe C — effet de la séparation sol (E3)."""
    return [
        _derive(base, "C0_E3_off", **{"e3.variant": E3Variant.OFF}),
        _derive(base, "C1_E3_hard", **{"e3.variant": E3Variant.HARD}),
        _derive(base, "C2_E3_soft", **{"e3.variant": E3Variant.SOFT}),
    ]


def group_D_cleaning(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe D — nettoyage robuste (E4), factoriel 2^3."""
    exps = []
    combos = [
        ("D0_none", False, False, False),
        ("D1_dbscan", True, False, False),
        ("D2_plane", False, True, False),
        ("D3_compact", False, False, True),
        ("D4_dbscan_plane", True, True, False),
        ("D5_dbscan_compact", True, False, True),
        ("D6_plane_compact", False, True, True),
        ("D7_full", True, True, True),
    ]
    for name, db, pl, cp in combos:
        exps.append(_derive(
            base, name,
            **{"e4.dbscan": db, "e4.plane": pl, "e4.compact": cp}))
    return exps


def group_E_reconstruction(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe E — reconstruction surfacique (E5)."""
    return [
        _derive(base, "E0_obb_only", **{"e5.variant": E5Variant.OBB_ONLY}),
        _derive(base, "E1_poisson_std", **{"e5.variant": E5Variant.POISSON_STD}),
        _derive(base, "E2_poisson_weighted",
                **{"e5.variant": E5Variant.POISSON_WEIGHTED}),
    ]


def group_F_fpfilter(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe F — filtrage faux positifs 3D (E6)."""
    return [
        _derive(base, "F0_E6_off", **{"e6.variant": E6Variant.OFF}),
        _derive(base, "F1_E6_noflat", **{"e6.variant": E6Variant.NOFLAT}),
        _derive(base, "F2_E6_full", **{"e6.variant": E6Variant.FULL}),
    ]


def group_G_seeds(base: PipelineConfig) -> List[PipelineConfig]:
    """Groupe G — variantes de génération de seeds (E1)."""
    return [
        _derive(base, "G0_voteSAM", **{"e1.variant": E1Variant.VOTE_SAM}),
        _derive(base, "G1_frustum", **{"e1.variant": E1Variant.FRUSTUM}),
        _derive(base, "G2_native", **{"e1.variant": E1Variant.NATIVE}),
    ]


def generate_all_experiments() -> List[PipelineConfig]:
    """
    Génère les 25 expériences d'ablation (groupes A à G).

    Returns:
        Liste de PipelineConfig nommées, prêtes à exécuter.
    """
    base = reference_config()
    all_exps: List[PipelineConfig] = []
    all_exps += group_A_baseline(base)
    all_exps += group_B_consensus(base)
    all_exps += group_C_ground(base)
    all_exps += group_D_cleaning(base)
    all_exps += group_E_reconstruction(base)
    all_exps += group_F_fpfilter(base)
    all_exps += group_G_seeds(base)
    return all_exps


def generate_group(group: str) -> List[PipelineConfig]:
    """Génère un seul groupe par lettre (A-G)."""
    base = reference_config()
    mapping = {
        "A": group_A_baseline, "B": group_B_consensus,
        "C": group_C_ground, "D": group_D_cleaning,
        "E": group_E_reconstruction, "F": group_F_fpfilter,
        "G": group_G_seeds,
    }
    if group.upper() not in mapping:
        raise ValueError(f"Groupe inconnu : {group}. Choix : {list(mapping)}")
    return mapping[group.upper()](base)


if __name__ == "__main__":
    exps = generate_all_experiments()
    print(f"{len(exps)} expériences générées :\n")
    for e in exps:
        print(f"  {e.name:28s} | method={e.method.value} | "
              f"E1={e.e1.variant.value} E2={e.e2.variant.value} "
              f"E3={e.e3.variant.value} "
              f"E4=[{int(e.e4.dbscan)}{int(e.e4.plane)}{int(e.e4.compact)}] "
              f"E5={e.e5.variant.value} E6={e.e6.variant.value}")
