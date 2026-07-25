"""
coastal_waste_3d -- framework de configuration, d'exécution et d'ablation
du pipeline Coastal-Waste-3D (segmentation 3D de déchets côtiers par
consensus multi-vues).

Modules :
  config      configuration typée + interface env_overrides()
  ablation    génération des configurations expérimentales (A-G)
  runner      exécution des expériences et agrégation statistique
  minitest_e2 test ciblé consensus (E2) x nettoyage (DBSCAN)
"""
