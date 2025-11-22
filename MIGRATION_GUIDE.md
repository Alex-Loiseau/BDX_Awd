# Guide de Migration IsaacGym → IsaacLab pour BDX_Awd

Ce guide documente la migration complète du projet BDX_Awd de IsaacGym vers IsaacLab (Isaac Sim 5.1.0).

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Architecture](#architecture)
5. [Correspondances API](#correspondances-api)
6. [Étapes de Migration](#étapes-de-migration)
7. [Tests et Validation](#tests-et-validation)

---

## Vue d'ensemble

### Différences Principales

| Aspect | IsaacGym | IsaacLab |
|--------|----------|----------|
| **Base Class** | `BaseTask` personnalisée | `DirectRLEnv` |
| **Configuration** | YAML | Python `@configclass` |
| **Tenseurs** | `gymtorch.wrap_tensor()` | Accès direct via `.data` |
| **Refresh** | Manuel (`gym.refresh_*`) | Automatique |
| **Quaternions** | `(x, y, z, w)` | `(w, x, y, z)` ⚠️ |
| **Vectorisation** | Manuelle (boucles) | Automatique |
| **Assets** | URDF direct | USD (conversion requise) |

---

## Prérequis

### Logiciels

- **Isaac Sim 5.1.0** : Installé dans `/home/alexandre/Developpements/env_isaaclab`
- **Python 3.10+** : Version compatible avec Isaac Sim
- **PyTorch** : Version compatible (généralement 2.0+)
- **CUDA** : Compatible avec votre GPU

### Vérifications

```bash
# Vérifier Isaac Sim
ls /home/alexandre/Developpements/env_isaaclab

# Activer l'environnement
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Vérifier Python
python --version
```

---

## Installation

### 1. Cloner IsaacLab

```bash
cd /home/alexandre/Developpements
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 2. Installer IsaacLab

```bash
# Avec Isaac Sim déjà installé
./isaaclab.sh --install

# Vérifier l'installation
./isaaclab.sh -p -m pip list | grep isaac
```

### 3. Installer les dépendances du projet

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Activer l'environnement IsaacLab
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Installer les dépendances (sans placo pour l'instant)
pip install torch numpy termcolor pyyaml scipy tensorboard pybullet six meshcat flask
pip install rl-games==1.1.4

# Pour placo (après avoir installé Eigen3)
sudo apt-get install libeigen3-dev
pip install placo==0.6.2
```

### 4. Convertir les URDF en USD

IsaacLab préfère le format USD pour de meilleures performances :

```bash
# Convertir mini_bdx
python -m isaaclab.utils.assets.urdf_converter \
    --input awd/data/assets/mini_bdx/urdf/bdx.urdf \
    --output awd/data/assets/mini_bdx/bdx.usd

# Convertir go_bdx
python -m isaaclab.utils.assets.urdf_converter \
    --input awd/data/assets/go_bdx/go_bdx.urdf \
    --output awd/data/assets/go_bdx/go_bdx.usd
```

---

## Architecture

### Structure du Projet (Après Migration)

```
BDX_Awd/
├── awd/                          # Code original IsaacGym (conservé)
│   ├── run.py
│   ├── env/tasks/
│   └── data/
│
├── awd_isaaclab/                 # Nouveau code IsaacLab
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── duckling_base_env.py       # Classe de base
│   │   ├── duckling_command_env.py    # Tâche de commande
│   │   ├── duckling_amp_env.py        # Tâche AMP
│   │   ├── duckling_heading_env.py    # Tâche de direction
│   │   └── duckling_perturb_env.py    # Tâche avec perturbations
│   │
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── duckling_base_cfg.py       # Config de base
│   │   ├── duckling_command_cfg.py    # Config commande
│   │   └── robots/
│   │       ├── mini_bdx_cfg.py
│   │       └── go_bdx_cfg.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── motion_lib.py              # Bibliothèque de motions (adapté)
│   │   ├── amp_utils.py               # Utilitaires AMP
│   │   └── rewards.py                 # Fonctions de récompense
│   │
│   └── scripts/
│       ├── run_isaaclab.py            # Point d'entrée principal
│       ├── train_command.py           # Entraînement commande
│       ├── train_amp.py               # Entraînement AMP
│       └── play.py                    # Exécution de politiques
│
├── MIGRATION_GUIDE.md           # Ce fichier
├── INSTALL.md                   # Instructions d'installation
└── requirements_isaaclab.txt    # Dépendances IsaacLab
```

---

## Correspondances API

### Acquisition de Tenseurs

| IsaacGym | IsaacLab |
|----------|----------|
| `gym.acquire_actor_root_state_tensor(sim)` | `robot.data.root_state_w` |
| `gym.acquire_dof_state_tensor(sim)` | `robot.data.joint_pos`, `robot.data.joint_vel` |
| `gym.acquire_rigid_body_state_tensor(sim)` | `robot.data.body_pos_w`, `robot.data.body_quat_w` |
| `gym.acquire_net_contact_force_tensor(sim)` | `robot.data.net_contact_force` |
| `gym.acquire_force_sensor_tensor(sim)` | `robot.data.force_sensor_data` |
| `gymtorch.wrap_tensor(tensor)` | **Supprimé** (accès direct) |

### Refresh de Tenseurs

| IsaacGym | IsaacLab |
|----------|----------|
| `gym.refresh_dof_state_tensor(sim)` | **Automatique** |
| `gym.refresh_actor_root_state_tensor(sim)` | **Automatique** |
| `gym.refresh_rigid_body_state_tensor(sim)` | **Automatique** |
| `gym.refresh_net_contact_force_tensor(sim)` | **Automatique** |

⚠️ **Important** : Les tenseurs sont automatiquement mis à jour après `sim.step()` dans IsaacLab.

### Application d'Actions

| IsaacGym | IsaacLab |
|----------|----------|
| `gym.set_dof_actuation_force_tensor(sim, forces)` | `robot.set_joint_effort_target(forces)` |
| `gym.set_dof_position_target_tensor(sim, positions)` | `robot.set_joint_position_target(positions)` |
| `gym.set_dof_velocity_target_tensor(sim, velocities)` | `robot.set_joint_velocity_target(velocities)` |

### Chargement d'Assets

**IsaacGym :**
```python
from isaacgym import gymapi

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
```

**IsaacLab :**
```python
from isaaclab.assets import ArticulationCfg
from isaaclab.assets.urdf import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg

robot_cfg = ArticulationCfg(
    prim_path="/World/envs/.*/Robot",
    spawn=UsdFileCfg(
        usd_path="path/to/robot.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,  # Position par défaut
        }
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=1000.0,
            damping=100.0,
        ),
    },
)
```

### État du Robot

**IsaacGym :**
```python
# Après refresh
root_states = self._root_states  # (num_envs, 13)
root_pos = root_states[:, :3]
root_quat = root_states[:, 3:7]   # (x, y, z, w)
root_lin_vel = root_states[:, 7:10]
root_ang_vel = root_states[:, 10:13]

dof_pos = self._dof_pos  # (num_envs, num_dof)
dof_vel = self._dof_vel  # (num_envs, num_dof)
```

**IsaacLab :**
```python
# Automatique, pas de refresh
robot = self._scene.articulations["robot"]

root_pos = robot.data.root_pos_w           # (num_envs, 3)
root_quat = robot.data.root_quat_w         # (num_envs, 4) ⚠️ (w, x, y, z)
root_lin_vel = robot.data.root_lin_vel_w   # (num_envs, 3)
root_ang_vel = robot.data.root_ang_vel_w   # (num_envs, 3)

dof_pos = robot.data.joint_pos             # (num_envs, num_joints)
dof_vel = robot.data.joint_vel             # (num_envs, num_joints)
```

⚠️ **ATTENTION : Format des quaternions différent !**

```python
# Conversion IsaacGym → IsaacLab
quat_lab = torch.cat([quat_gym[..., 3:4], quat_gym[..., :3]], dim=-1)

# Conversion IsaacLab → IsaacGym
quat_gym = torch.cat([quat_lab[..., 1:4], quat_lab[..., 0:1]], dim=-1)
```

### Création d'Environnements

**IsaacGym :**
```python
class Duckling(BaseTask):
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, duckling_asset)
```

**IsaacLab :**
```python
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg

class DucklingEnv(DirectRLEnv):
    cfg: DucklingEnvCfg

    def _setup_scene(self):
        # Ajouter le robot (créé automatiquement pour tous les envs)
        self._scene.add(self.cfg.robot, "robot")

        # Ajouter le sol
        from isaaclab.terrains import TerrainImporterCfg
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
        )
        self._scene.add(terrain_cfg, "terrain")
```

---

## Étapes de Migration

### Étape 1 : Créer la Configuration de Base

Créer `awd_isaaclab/configs/duckling_base_cfg.py` :

```python
from dataclasses import MISSING
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils.configclass import configclass

@configclass
class DucklingBaseCfg(DirectRLEnvCfg):
    """Configuration de base pour les environnements Duckling."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0,
        substeps=2,
        physics_engine="physx",
        physx=PhysxCfg(
            solver_type=1,  # TGS
            num_threads=4,
            num_position_iterations=4,
            num_velocity_iterations=0,
            contact_offset=0.02,
            rest_offset=0.0,
            bounce_threshold_velocity=0.2,
            max_depenetration_velocity=10.0,
            default_buffer_size_multiplier=5.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
        replicate_physics=True,
    )

    # Episode
    episode_length_s: float = 500 * (1/60.0)  # 500 steps

    # Observations/Actions
    num_observations: int = MISSING  # À définir dans les sous-classes
    num_actions: int = MISSING       # À définir dans les sous-classes

    # Control
    decimation: int = 2  # Équivalent de controlFrequencyInv
    action_scale: float = 1.0

    # Termination
    enable_early_termination: bool = True
    termination_height: float = 0.1

    # Randomization
    randomize_com: bool = False
    com_range: list = None
    randomize_torques: bool = False
    torque_multiplier_range: list = None

    # Debug
    debug_vis: bool = False
```

### Étape 2 : Créer la Classe d'Environnement de Base

Créer `awd_isaaclab/envs/duckling_base_env.py` (voir fichier généré).

### Étape 3 : Créer les Configurations Robot

Créer `awd_isaaclab/configs/robots/mini_bdx_cfg.py` et `go_bdx_cfg.py`.

### Étape 4 : Migrer les Tâches Spécifiques

- `duckling_command_env.py`
- `duckling_amp_env.py`
- `duckling_heading_env.py`
- etc.

### Étape 5 : Adapter le Motion Loader

Migrer `awd/utils/motion_lib.py` vers `awd_isaaclab/utils/motion_lib.py`.

### Étape 6 : Créer le Script d'Entraînement

Créer `awd_isaaclab/scripts/run_isaaclab.py`.

### Étape 7 : Tests et Validation

Comparer les résultats avec la version IsaacGym.

---

## Tests et Validation

### Test de Base

```bash
cd /home/alexandre/Developpements/BDX_Awd
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --num_envs 16 \
    --headless
```

### Checklist de Validation

- [ ] Les observations ont les mêmes dimensions
- [ ] Les récompenses sont similaires
- [ ] Les quaternions sont correctement convertis
- [ ] Les capteurs de force fonctionnent
- [ ] Le reset fonctionne correctement
- [ ] Les motions AMP se chargent
- [ ] L'entraînement converge
- [ ] Les performances sont comparables

### Debugging

Pour déboguer les différences :

```python
# Dans l'environnement
def debug_observations(self):
    print(f"Root pos: {self._robot.data.root_pos_w[0]}")
    print(f"Root quat: {self._robot.data.root_quat_w[0]}")
    print(f"Joint pos: {self._robot.data.joint_pos[0]}")
    print(f"Contact forces: {self._robot.data.net_contact_force[0]}")
```

---

## Ressources

- **IsaacLab Documentation** : https://isaac-sim.github.io/IsaacLab/
- **IsaacLab GitHub** : https://github.com/isaac-sim/IsaacLab
- **Isaac Sim Documentation** : https://docs.omniverse.nvidia.com/isaacsim/latest/
- **Forum NVIDIA** : https://forums.developer.nvidia.com/c/omniverse/simulation/69

---

## Support

Pour toute question sur la migration :

1. Consulter ce guide
2. Vérifier les exemples dans `/path/to/IsaacLab/source/examples/`
3. Consulter la documentation officielle
4. Ouvrir une issue sur le repo du projet

---

## Changelog

- **2025-11-21** : Création du guide de migration initial
- Version IsaacGym : Legacy (pré-2023)
- Version IsaacLab : Compatible Isaac Sim 5.1.0
