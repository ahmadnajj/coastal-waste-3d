"""
Configuration typée du pipeline Coastal-Waste-3D.

Toute la paramétrisation du pipeline passe par ces dataclasses. Une
configuration d'ablation est une instance de PipelineConfig. Le chargement
depuis YAML et la sérialisation sont gérés ici.

Conception : dataclasses (typage statique, valeurs par défaut, validation
légère) plutôt qu'un framework de configuration lourd.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import json


# ─────────────────────────────────────────────────────────────────────
#  Enumérations des variantes (absorbent A/B/C/D comme options)
# ─────────────────────────────────────────────────────────────────────

class Method(str, Enum):
    """Méthode de segmentation 3D."""
    S0 = "S0"      # baseline native pixel mapping
    E = "E"        # Coastal-Waste-3D (méthode principale)


class E1Variant(str, Enum):
    """Génération de seeds (module E1)."""
    NATIVE = "native"      # pts3d natifs indexés par masque SAM (défaut)
    FRUSTUM = "frustum"    # intersection de frustums YOLO (exécutée via --approach C)
    VOTE_SAM = "voteSAM"   # projection inverse classique (référence)


class E2Variant(str, Enum):
    """Consensus multi-vues (module E2)."""
    CONSENSUS = "consensus"  # confiance pondérée par fraction de vues (défaut)
    VOTE_3D = "vote3D"       # vote binaire avec seuil sur le nombre de vues
    OFF = "off"              # pas de pondération


class E3Variant(str, Enum):
    """Séparation objet/sol (module E3)."""
    SOFT = "soft"   # coupe fine + anti-amputation objets debout (défaut)
    HARD = "hard"   # coupe brutale
    OFF = "off"     # aucune coupe (s'appuie sur ground removal global)


class E5Variant(str, Enum):
    """Reconstruction surfacique (module E5)."""
    POISSON_STD = "poisson_std"            # Poisson non pondéré (défaut)
    POISSON_WEIGHTED = "poisson_weighted"  # Poisson pondéré par la confiance
    OBB_ONLY = "obb_only"                  # volume OBB seul, pas de maillage


class E6Variant(str, Enum):
    """Filtrage faux positifs 3D (module E6)."""
    FULL = "full"      # 6 règles + exemption objets plats + garde-fous (défaut)
    NOFLAT = "noflat"  # sans exemption objets plats
    OFF = "off"        # aucun filtrage


# ─────────────────────────────────────────────────────────────────────
#  Sous-configurations par étage / module
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    """Chemins d'entrée/sortie (informatifs : les chemins effectifs sont
    fournis par les variables d'environnement du script de soumission)."""
    mast3r_dir: str = "./data/mast3r"
    sam_ckpt: str = "./data/models/sam_vit_h.pth"
    yolo_model: str = "./data/models/yolo_taco/best.pt"
    scenes_dir: str = "./data/scenes"
    outputs_dir: str = "../outputs"
    ground_truth_dir: str = "./data/ground_truth"
    hf_home: str = "./data/hf_cache"


@dataclass
class PerceptionConfig:
    """Étage 1 : perception 2D + reconstruction 3D (commun, non ablaté)."""
    max_images: int = 34              # sous-échantillonnage pour MASt3R
    yolo_conf_threshold: float = 0.25
    nms_2d_iou: float = 0.5
    fp2d_iou_threshold: float = 0.5
    fp2d_min_views_ratio: float = 0.10
    mast3r_scene_graph: str = "swin-5"
    mast3r_iter: int = 200
    ground_removal: bool = True
    ground_margin: float = 0.01       # fraction du span vertical


@dataclass
class E1Config:
    """Module E1 — génération de seeds."""
    variant: E1Variant = E1Variant.NATIVE
    min_views: int = 2


@dataclass
class E2Config:
    """Module E2 — consensus multi-vues."""
    variant: E2Variant = E2Variant.CONSENSUS
    conf_min: float = 0.15            # fraction de vues minimale
    consensus_eps: float = 0.012      # rayon de consensus (m)
    vote_3d_eps: float = 0.02         # rayon pour variante vote3D
    vote_3d_min_views: int = 3


@dataclass
class E3Config:
    """Module E3 — séparation objet/sol."""
    variant: E3Variant = E3Variant.SOFT
    height_pctl: float = 8.0          # percentile sol
    ground_margin: float = 0.005      # marge fine (fraction span)
    anti_amputation_frac: float = 0.40  # seuil objet debout


@dataclass
class E4Config:
    """Module E4 — nettoyage robuste (3 sous-interrupteurs)."""
    dbscan: bool = True
    plane: bool = True
    compact: bool = True
    dbscan_eps_factor: float = 2.0
    min_inlier_ratio: float = 0.30
    plane_inlier_thresh: float = 0.01
    plane_min_pts: int = 800          # seuil min pour activer RANSAC plan
    plane_max_ratio: float = 0.70     # au-delà = objet plat, on n'élimine pas
    compact_z_thresh: float = 2.5


@dataclass
class E5Config:
    """Module E5 — reconstruction surfacique."""
    variant: E5Variant = E5Variant.POISSON_STD
    poisson_depth: int = 9
    gsce_max_vol_factor: float = 3.0  # arrêt expansion (si variante avec GSCE)


@dataclass
class E6Config:
    """Module E6 — filtrage faux positifs 3D."""
    variant: E6Variant = E6Variant.FULL
    min_rules_pass: int = 5
    flat_footprint_max: float = 0.35  # seuil objet plat (m)
    guard_ratio_max: float = 1.05     # rejet si ratio Poisson/SVD dépasse
    guard_min_views: int = 5          # garde-fou vues
    guard_min_conf: float = 0.35      # garde-fou confiance


@dataclass
class ReproConfig:
    """Reproductibilité."""
    seed: int = 42
    deterministic: bool = True


@dataclass
class PipelineConfig:
    """
    Configuration complète d'un run du pipeline.

    Une expérience d'ablation = une instance de cette classe avec des
    variantes/flags spécifiques. Le nom identifie l'expérience.
    """
    name: str = "default_E"
    method: Method = Method.E
    paths: PathsConfig = field(default_factory=PathsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    e1: E1Config = field(default_factory=E1Config)
    e2: E2Config = field(default_factory=E2Config)
    e3: E3Config = field(default_factory=E3Config)
    e4: E4Config = field(default_factory=E4Config)
    e5: E5Config = field(default_factory=E5Config)
    e6: E6Config = field(default_factory=E6Config)
    repro: ReproConfig = field(default_factory=ReproConfig)

    # ── Sérialisation ──

    def to_dict(self) -> dict:
        """Convertit en dict sérialisable (enums → str)."""
        d = asdict(self)
        return _enums_to_str(d)

    def to_yaml(self, path: str | Path) -> None:
        """Sauvegarde en YAML (nécessite pyyaml)."""
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False,
                           default_flow_style=False)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruit depuis un dict (str → enums)."""
        return cls(
            name=d.get("name", "loaded"),
            method=Method(d.get("method", "E")),
            paths=PathsConfig(**d.get("paths", {})),
            perception=PerceptionConfig(**d.get("perception", {})),
            e1=_build_e1(d.get("e1", {})),
            e2=_build_e2(d.get("e2", {})),
            e3=_build_e3(d.get("e3", {})),
            e4=E4Config(**d.get("e4", {})),
            e5=_build_e5(d.get("e5", {})),
            e6=_build_e6(d.get("e6", {})),
            repro=ReproConfig(**d.get("repro", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def env_overrides(self) -> dict[str, str]:
        """
        Traduit la configuration typée en variables d'environnement :
        c'est l'interface de pilotage du pipeline (pipeline.py),
        qui lit toute sa configuration dans l'environnement.
        """
        env = {}
        # Perception
        env["COASTAL_MAX_IMAGES"] = str(self.perception.max_images)
        env["COASTAL_GROUND_REMOVE"] = "1" if self.perception.ground_removal else "0"
        env["COASTAL_GROUND_MARGIN"] = str(self.perception.ground_margin)
        # ── Variantes de modules E (interrupteurs d'ablation) ──
        env["NMC_E1_VARIANT"] = self.e1.variant.value   # native|voteSAM|frustum
        env["NMC_E2_VARIANT"] = self.e2.variant.value   # consensus|vote3D|off
        env["NMC_E3_VARIANT"] = self.e3.variant.value   # soft|hard|off
        env["NMC_E5_VARIANT"] = self.e5.variant.value   # poisson_std|weighted|obb_only
        env["NMC_E6_VARIANT"] = self.e6.variant.value   # full|noflat|off
        env["NMC_VOTE3D_MINVIEWS"] = str(self.e2.vote_3d_min_views)
        # E2 (paramètres)
        env["NMC_CONF_MIN"] = str(self.e2.conf_min)
        env["NMC_EPS"] = str(self.e2.consensus_eps)
        # E3 (paramètres)
        env["NMC_GROUND_MARGIN"] = str(self.e3.ground_margin)
        env["NMC_HEIGHT_PCTL"] = str(self.e3.height_pctl)
        # E4 (sous-interrupteurs)
        env["CLEAN_DBSCAN"] = "1" if self.e4.dbscan else "0"
        env["CLEAN_PLANE"] = "1" if self.e4.plane else "0"
        env["CLEAN_COMPACT"] = "1" if self.e4.compact else "0"
        # E5 (paramètre GSCE si variante l'utilise)
        env["GSCE_MAX_VOL_FACTOR"] = str(self.e5.gsce_max_vol_factor)
        # E6 : noflat = désactiver l'exemption objets plats (seuil 0)
        if self.e6.variant == E6Variant.NOFLAT:
            env["FP3D_FLAT_FOOTPRINT_MAX"] = "0.0"
        else:
            env["FP3D_FLAT_FOOTPRINT_MAX"] = str(self.e6.flat_footprint_max)
        # Repro
        env["COASTAL_SEED"] = str(self.repro.seed)
        return env


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _enums_to_str(obj):
    """Convertit récursivement les enums en str dans un dict/list."""
    if isinstance(obj, dict):
        return {k: _enums_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enums_to_str(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _build_e1(d: dict) -> E1Config:
    d = dict(d)
    if "variant" in d:
        d["variant"] = E1Variant(d["variant"])
    return E1Config(**d)


def _build_e2(d: dict) -> E2Config:
    d = dict(d)
    if "variant" in d:
        d["variant"] = E2Variant(d["variant"])
    return E2Config(**d)


def _build_e3(d: dict) -> E3Config:
    d = dict(d)
    if "variant" in d:
        d["variant"] = E3Variant(d["variant"])
    return E3Config(**d)


def _build_e5(d: dict) -> E5Config:
    d = dict(d)
    if "variant" in d:
        d["variant"] = E5Variant(d["variant"])
    return E5Config(**d)


def _build_e6(d: dict) -> E6Config:
    d = dict(d)
    if "variant" in d:
        d["variant"] = E6Variant(d["variant"])
    return E6Config(**d)
