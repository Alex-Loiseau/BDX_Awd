# BDX_Awd - IsaacLab Version

Version migr\u00e9e du projet BDX_Awd utilisant **IsaacLab** (Isaac Sim 5.1.0) au lieu d'IsaacGym.

## Vue d'ensemble

Ce dossier contient la version IsaacLab du projet d'apprentissage locomotion pour robots bipèdes. Il remplace l'ancienne version IsaacGym tout en conservant les mêmes fonctionnalités et algorithmes d'apprentissage.

### Avantages de la migration

- ✅ **Performances améliorées** : IsaacLab est optimisé pour Isaac Sim moderne
- ✅ **Support actif** : IsaacGym n'est plus maintenu, IsaacLab est activement développé
- ✅ **Meilleures visualisations** : Isaac Sim offre un rendu de meilleure qualité
- ✅ **APIs modernes** : Code plus propre et plus maintenable
- ✅ **Compatibilité future** : Prêt pour les futures mises à jour d'Isaac Sim

## Structure du Projet

```
awd_isaaclab/
├── configs/                 # Configurations des environnements et robots
│   ├── robots/
│   │   ├── mini_bdx_cfg.py # Configuration Mini BDX
│   │   └── go_bdx_cfg.py   # Configuration Go BDX
│   └── __init__.py
│
├── envs/                    # Environnements d'apprentissage
│   ├── duckling_base_env.py      # Classe de base (remplace Duckling)
│   ├── duckling_command_env.py   # Tâche de suivi de commandes
│   └── __init__.py
│
├── scripts/                 # Scripts exécutables
│   ├── run_isaaclab.py     # Point d'entrée principal
│   ├── convert_assets.py   # Conversion URDF → USD
│   └── train.py            # Script d'entraînement (TODO)
│
├── utils/                   # Utilitaires
│   ├── motion_lib.py       # Bibliothèque de motions (TODO)
│   └── __init__.py
│
└── README.md               # Ce fichier
```

## Installation

Voir [INSTALL.md](../INSTALL.md) pour les instructions complètes d'installation.

### Installation Rapide

```bash
# 1. Installer IsaacLab
cd /home/alexandre/Developpements
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install

# 2. Installer dépendances
cd /home/alexandre/Developpements/BDX_Awd
pip install -r requirements_isaaclab.txt

# 3. Convertir assets URDF → USD
python awd_isaaclab/scripts/convert_assets.py --all

# 4. Tester
python awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test
```

## Utilisation

### Entraînement

```bash
# Entraînement avec visualisation
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 4096 \
    --experiment mini_bdx_walk

# Entraînement headless (recommandé)
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000
```

### Exécution d'une Politique

```bash
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --play \
    --checkpoint runs/mini_bdx_walk/checkpoint.pth
```

### Options Disponibles

- `--task` : Tâche à exécuter (`DucklingCommand`, etc.)
- `--robot` : Type de robot (`mini_bdx`, `go_bdx`)
- `--num_envs` : Nombre d'environnements parallèles (défaut: 4096)
- `--train` : Mode entraînement
- `--play` : Mode exécution (requiert `--checkpoint`)
- `--headless` : Mode sans visualisation (plus rapide)
- `--experiment` : Nom de l'expérience pour logging
- `--max_iterations` : Nombre d'itérations d'entraînement

## Tâches Disponibles

### DucklingCommand

Entraîne le robot à suivre des commandes de vitesse (vx, vy, vyaw).

**Observations :**
- Orientation du corps (quaternion)
- Vitesses linéaires et angulaires
- Positions et vitesses des joints
- Actions précédentes
- Commandes de vitesse

**Actions :**
- Couples des joints (ou cibles PD)

**Récompenses :**
- Suivi de vitesse linéaire (xy)
- Suivi de vitesse angulaire (yaw)
- Pénalité de couple
- Pénalité de variation d'action

### Tâches à Migrer (TODO)

- `DucklingAMP` : Apprentissage avec Adversarial Motion Priors
- `DucklingHeading` : Suivi de direction
- `DucklingPerturb` : Robustesse aux perturbations
- `DucklingViewMotion` : Visualisation de motions

## Configuration

Les configurations sont définies en Python (pas YAML comme dans IsaacGym).

### Exemple : Modifier les Récompenses

```python
# Dans awd_isaaclab/configs/robots/mini_bdx_cfg.py

MINI_BDX_PARAMS = {
    # ...
    "reward_scales": {
        "lin_vel_xy": 0.5,      # Augmenter pour favoriser vitesse
        "ang_vel_z": 0.25,
        "torque": -0.000025,    # Diminuer pour moins pénaliser
        "action_rate": -1.0,
    },
}
```

### Exemple : Modifier les Plages de Commandes

```python
MINI_BDX_PARAMS = {
    # ...
    "command_ranges": {
        "linear_x": [-0.2, 0.2],  # Réduire vitesse max
        "linear_y": [-0.15, 0.15],
        "yaw": [-0.5, 0.5],
    },
}
```

## Différences avec IsaacGym

### APIs Principales

| IsaacGym | IsaacLab | Notes |
|----------|----------|-------|
| `BaseTask` | `DirectRLEnv` | Classe de base |
| `gymtorch.wrap_tensor()` | Accès direct | Plus besoin de wrapper |
| `gym.refresh_*()` | Automatique | Refresh automatique |
| Quaternions `(x,y,z,w)` | `(w,x,y,z)` | ⚠️ Ordre différent |
| YAML config | Python `@configclass` | Configuration en Python |

### Avantages IsaacLab

1. **Moins de code boilerplate** : Pas besoin de gérer manuellement les environnements
2. **Tenseurs automatiques** : Plus besoin de `wrap_tensor()` et `refresh()`
3. **Meilleure intégration** : Support natif de Gymnasium
4. **Performances** : Optimisations internes pour GPU

Voir [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) pour plus de détails.

## Développement

### Ajouter une Nouvelle Tâche

1. Créer `awd_isaaclab/envs/ma_tache_env.py`
2. Hériter de `DucklingBaseEnv`
3. Implémenter `_get_observations()` et `_get_rewards()`
4. Créer la configuration dans `configs/`
5. Tester !

### Ajouter un Nouveau Robot

1. Convertir URDF → USD
2. Créer `configs/robots/mon_robot_cfg.py`
3. Définir `MON_ROBOT_CFG` et `MON_ROBOT_PARAMS`
4. Ajouter le choix dans `run_isaaclab.py`

## Problèmes Connus

### Import Errors

Si vous voyez des erreurs d'import IsaacLab :

```bash
# Vérifier que IsaacLab est bien installé
source /home/alexandre/Developpements/env_isaaclab/bin/activate
python -c "import omni.isaac.lab; print(omni.isaac.lab.__file__)"
```

### USD Conversion Failures

Si la conversion URDF → USD échoue :

1. Vérifier que tous les meshes sont présents
2. Vérifier les chemins dans le URDF
3. Utiliser directement le URDF (moins performant mais fonctionne)

### GPU Out of Memory

Réduire `--num_envs` :

```bash
--num_envs 2048  # Au lieu de 4096
--num_envs 1024  # Ou encore moins
```

## Ressources

- **Documentation IsaacLab** : https://isaac-sim.github.io/IsaacLab/
- **GitHub IsaacLab** : https://github.com/isaac-sim/IsaacLab
- **Exemples IsaacLab** : https://github.com/isaac-sim/IsaacLab/tree/main/source/extensions/omni.isaac.lab_tasks
- **Forum NVIDIA** : https://forums.developer.nvidia.com/c/omniverse/simulation/69

## Contribuer

Pour contribuer au projet :

1. Tester vos modifications
2. Documenter les changements
3. Vérifier que les tests passent
4. Créer une pull request

## Licence

Voir [LICENSE](../LICENSE) du projet principal.

## Contact

Pour questions ou support, consulter :
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md)
- [INSTALL.md](../INSTALL.md)
- Documentation IsaacLab officielle

---

**Version** : 1.0.0 (Migration initiale)
**Date** : 2025-11-21
**IsaacLab** : Compatible Isaac Sim 5.1.0
