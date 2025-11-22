# Guide de Démarrage Rapide - Migration IsaacLab

Ce guide vous aide à démarrer rapidement avec la version IsaacLab de BDX_Awd.

## 🚀 Installation en 5 Minutes

```bash
# 1. Cloner et installer IsaacLab
cd /home/alexandre/Developpements
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 2. Installer les dépendances du projet
cd /home/alexandre/Developpements/BDX_Awd
source /home/alexandre/Developpements/env_isaaclab/bin/activate
pip install -r requirements_isaaclab.txt

# 3. Convertir les assets URDF → USD
python awd_isaaclab/scripts/convert_assets.py --all

# 4. Test rapide (16 environnements, 5 secondes)
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

Si tout fonctionne, vous devriez voir la simulation se lancer avec 16 robots !

## 📚 Structure du Projet

Le projet est maintenant organisé en deux parties :

```
BDX_Awd/
├── awd/                    # Version IsaacGym ORIGINALE (conservée)
│   ├── run.py             # Ancien point d'entrée
│   └── env/tasks/         # Anciennes tâches
│
└── awd_isaaclab/          # Version IsaacLab NOUVELLE
    ├── scripts/
    │   └── run_isaaclab.py    # ← NOUVEAU point d'entrée
    ├── envs/
    │   ├── duckling_base_env.py
    │   └── duckling_command_env.py
    └── configs/
        └── robots/
```

**Important** : Le code IsaacGym original est conservé dans `awd/` pour référence, mais vous devez utiliser `awd_isaaclab/` pour IsaacLab.

## 🎯 Commandes Principales

### Test (court, pour vérifier que tout fonctionne)

```bash
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

### Entraînement (petite échelle, avec visualisation)

```bash
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 512 \
    --max_iterations 1000 \
    --experiment test_run
```

### Entraînement (grande échelle, headless)

```bash
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --experiment mini_bdx_walk_v1
```

### Exécution d'une politique entraînée

```bash
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --play \
    --checkpoint runs/mini_bdx_walk_v1/checkpoint.pth
```

## 🤖 Robots Disponibles

### Mini BDX

- **Fichier** : `configs/robots/mini_bdx_cfg.py`
- **Hauteur initiale** : 0.18 m
- **Plages de commandes** : ±0.13 m/s (x), ±0.1 m/s (y), ±0.5 rad/s (yaw)
- **Période de démarche** : 0.432 s

```bash
--robot mini_bdx
```

### Go BDX

- **Fichier** : `configs/robots/go_bdx_cfg.py`
- **Hauteur initiale** : 0.0 m (au sol)
- **Plages de commandes** : ±0.3 m/s (x/y), ±0.2 rad/s (yaw)
- **Période de démarche** : 0.6 s

```bash
--robot go_bdx
```

## 📊 Monitoring de l'Entraînement

Les logs sont sauvegardés dans `runs/<experiment_name>/`.

### TensorBoard

```bash
tensorboard --logdir runs/
```

Ouvrir http://localhost:6006 dans votre navigateur.

### Checkpoints

Les checkpoints sont sauvegardés automatiquement :

```
runs/
└── mini_bdx_walk_v1/
    ├── checkpoint.pth
    ├── config.yaml
    └── events.out.tfevents.*
```

## 🔧 Personnalisation

### Modifier les Récompenses

Éditer `awd_isaaclab/configs/robots/mini_bdx_cfg.py` :

```python
MINI_BDX_PARAMS = {
    "reward_scales": {
        "lin_vel_xy": 1.0,        # ← Augmenter pour favoriser vitesse
        "ang_vel_z": 0.25,
        "torque": -0.00001,       # ← Diminuer pénalité
        "action_rate": -0.5,
    },
}
```

### Modifier les Plages de Commandes

```python
MINI_BDX_PARAMS = {
    "command_ranges": {
        "linear_x": [-0.2, 0.3],  # ← Vitesse max différente avant/arrière
        "linear_y": [-0.15, 0.15],
        "yaw": [-0.5, 0.5],
    },
}
```

### Ajouter des Observations

Modifier `awd_isaaclab/envs/duckling_command_env.py` dans `_get_observations()` :

```python
def _get_observations(self) -> Dict[str, torch.Tensor]:
    # ... observations existantes ...

    # Ajouter par exemple les forces de contact
    contact_forces = self._robot.data.net_contact_force[:, self._feet_ids]

    obs = torch.cat([
        # ... observations existantes ...
        contact_forces.flatten(1),  # ← Nouvelle observation
    ], dim=-1)

    return {"policy": obs}
```

## ⚠️ Différences Importantes avec IsaacGym

### 1. Format des Quaternions

```python
# IsaacGym : (x, y, z, w)
quat_gym = [0, -0.08, 0, 1]

# IsaacLab : (w, x, y, z)
quat_lab = [1, 0, -0.08, 0]
```

### 2. Plus de Wrapper de Tenseurs

```python
# IsaacGym (ANCIEN)
dof_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_tensor)
gym.refresh_dof_state_tensor(sim)  # ← Obligatoire !

# IsaacLab (NOUVEAU)
dof_pos = robot.data.joint_pos  # ← Direct, déjà à jour !
dof_vel = robot.data.joint_vel
```

### 3. Configuration en Python

```python
# IsaacGym (ANCIEN) - YAML
# duckling_command.yaml
env:
  numEnvs: 4096
  learn:
    linearVelocityXYRewardScale: 0.5

# IsaacLab (NOUVEAU) - Python
@configclass
class DucklingCommandCfg(DirectRLEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096
    )
    lin_vel_xy_reward_scale: float = 0.5
```

## 🐛 Dépannage Rapide

### Problème : `ModuleNotFoundError: No module named 'omni'`

**Solution** : IsaacLab n'est pas installé ou l'environnement n'est pas activé.

```bash
source /home/alexandre/Developpements/env_isaaclab/bin/activate
cd /home/alexandre/Developpements/IsaacLab
./isaaclab.sh --install
```

### Problème : Conversion USD échoue

**Solution** : Vérifier que le URDF et les meshes existent.

```bash
# Vérifier URDF
ls awd/data/assets/mini_bdx/urdf/bdx.urdf

# Vérifier meshes
ls awd/data/assets/mini_bdx/meshes/

# Utiliser URDF directement (moins performant)
# Modifier mini_bdx_cfg.py : usd_path="...bdx.urdf"
```

### Problème : GPU out of memory

**Solution** : Réduire le nombre d'environnements.

```bash
--num_envs 1024  # Au lieu de 4096
```

### Problème : Simulation trop lente

**Solution** : Mode headless + réduire fréquence de rendu.

```bash
--headless  # Pas de visualisation
```

## 📖 Documentation Complète

- **[INSTALL.md](INSTALL.md)** : Installation détaillée étape par étape
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** : Guide complet de migration IsaacGym → IsaacLab
- **[awd_isaaclab/README.md](awd_isaaclab/README.md)** : Documentation du code IsaacLab

## 🎓 Prochaines Étapes

1. ✅ Installation et test basique (vous êtes ici)
2. 📝 Lire [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) pour comprendre les différences
3. 🏋️ Lancer un entraînement complet
4. 🎮 Tester la politique entraînée
5. 🔧 Personnaliser les récompenses et observations
6. 🚀 Migrer les autres tâches (AMP, Heading, etc.)

## 💡 Conseils

1. **Commencez petit** : Testez d'abord avec peu d'environnements (16-512)
2. **Mode headless** : Utilisez `--headless` pour l'entraînement final
3. **Sauvegardez régulièrement** : Les checkpoints sont sauvegardés automatiquement
4. **Monitoring** : Utilisez TensorBoard pour suivre l'entraînement
5. **GPU** : Surveillez l'utilisation GPU avec `nvidia-smi`

## 📞 Support

Problèmes ? Consultez dans l'ordre :

1. Ce fichier (QUICKSTART.md)
2. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Différences API
3. [INSTALL.md](INSTALL.md) - Installation détaillée
4. [IsaacLab Docs](https://isaac-sim.github.io/IsaacLab/) - Documentation officielle
5. [Forum NVIDIA](https://forums.developer.nvidia.com/c/omniverse/simulation/69) - Support communautaire

---

**Bonne chance ! 🎉**

Si le test rapide fonctionne, vous êtes prêt à commencer l'entraînement !
