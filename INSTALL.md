# Installation Guide - BDX_Awd with IsaacLab

Ce guide vous aide à installer et configurer BDX_Awd avec IsaacLab pour Isaac Sim 5.1.0.

## Prérequis

### Matériel

- GPU NVIDIA avec support CUDA (recommandé: RTX 3000 series ou supérieur)
- 16 GB RAM minimum (32 GB recommandé)
- 50 GB d'espace disque

### Logiciels

- **Ubuntu 20.04/22.04** (ou compatible)
- **NVIDIA Driver** : Version 525+ recommandée
- **Isaac Sim 5.1.0** : Installé dans `/home/alexandre/Developpements/env_isaaclab`

## Étape 1 : Vérifier Isaac Sim

```bash
# Vérifier que Isaac Sim est installé
ls /home/alexandre/Developpements/env_isaaclab

# Activer l'environnement
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Vérifier Python
python --version  # Devrait être 3.10+
```

## Étape 2 : Installer IsaacLab

```bash
# Aller dans le dossier de développements
cd /home/alexandre/Developpements

# Cloner IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Installer IsaacLab
./isaaclab.sh --install

# Vérifier l'installation
./isaaclab.sh -p -m pip list | grep isaac
```

Si l'installation réussit, vous devriez voir plusieurs packages `isaac-*`.

## Étape 3 : Installer les Dépendances du Projet

```bash
# Retourner au projet BDX_Awd
cd /home/alexandre/Developpements/BDX_Awd

# Activer l'environnement IsaacLab
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Installer les dépendances Python
pip install -r requirements_isaaclab.txt
```

## Étape 4 : Installer Eigen3 (pour placo)

Le package `placo` requiert Eigen3 pour la génération de démarches :

```bash
# Installer Eigen3
sudo apt-get update
sudo apt-get install libeigen3-dev

# Installer placo
pip install placo==0.6.2
```

Si vous n'avez pas besoin de `placo`, vous pouvez ignorer cette étape.

## Étape 5 : Convertir les URDF en USD

IsaacLab utilise le format USD (Universal Scene Description) pour de meilleures performances.

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Convertir mini_bdx
python -m omni.isaac.lab.utils.assets.urdf_converter \\
    --input awd/data/assets/mini_bdx/urdf/bdx.urdf \\
    --output awd/data/assets/mini_bdx/bdx.usd \\
    --make-instanceable

# Convertir go_bdx (si vous avez ce fichier)
python -m omni.isaac.lab.utils.assets.urdf_converter \\
    --input awd/data/assets/go_bdx/go_bdx.urdf \\
    --output awd/data/assets/go_bdx/go_bdx.usd \\
    --make-instanceable
```

**Note** : Si vous rencontrez des erreurs de conversion, vérifiez que :
1. Les chemins vers les meshes dans le URDF sont corrects
2. Tous les fichiers de mesh (.stl, .obj, .dae) sont présents
3. Le URDF est valide (pas d'erreurs de syntaxe XML)

## Étape 6 : Tester l'Installation

### Test Rapide (16 environnements)

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test simple sans entraînement
python awd_isaaclab/scripts/run_isaaclab.py \\
    --task DucklingCommand \\
    --robot mini_bdx \\
    --test \\
    --num_envs 16
```

Si cela fonctionne, vous devriez voir :
- La simulation se lancer
- 16 robots apparaître
- Des informations sur les observations/actions
- La simulation s'exécuter pendant quelques secondes

### Test d'Entraînement (court)

```bash
# Entraînement court (headless)
python awd_isaaclab/scripts/run_isaaclab.py \\
    --task DucklingCommand \\
    --robot mini_bdx \\
    --train \\
    --headless \\
    --num_envs 512 \\
    --max_iterations 100
```

## Étape 7 : Entraînement Complet

Une fois les tests réussis, vous pouvez lancer un entraînement complet :

```bash
# Avec visualisation (plus lent)
python awd_isaaclab/scripts/run_isaaclab.py \\
    --task DucklingCommand \\
    --robot mini_bdx \\
    --train \\
    --num_envs 4096 \\
    --experiment mini_bdx_command \\
    --max_iterations 10000

# Headless (recommandé pour entraînement)
python awd_isaaclab/scripts/run_isaaclab.py \\
    --task DucklingCommand \\
    --robot mini_bdx \\
    --train \\
    --headless \\
    --num_envs 4096 \\
    --experiment mini_bdx_command \\
    --max_iterations 10000
```

Les checkpoints seront sauvegardés dans `runs/<experiment_name>/`.

## Étape 8 : Exécution d'une Politique Entraînée

```bash
python awd_isaaclab/scripts/run_isaaclab.py \\
    --task DucklingCommand \\
    --robot mini_bdx \\
    --play \\
    --checkpoint runs/mini_bdx_command/checkpoint.pth \\
    --num_envs 1
```

## Dépannage

### Problème : IsaacLab ne s'installe pas

**Solution** : Vérifiez que Isaac Sim 5.1.0 est correctement installé et que l'environnement Python est activé.

```bash
# Vérifier Isaac Sim
ls /home/alexandre/Developpements/env_isaaclab/isaac-sim*

# Réinstaller IsaacLab
cd /home/alexandre/Developpements/IsaacLab
./isaaclab.sh --install --force
```

### Problème : Conversion URDF → USD échoue

**Solution 1** : Vérifier les chemins dans le URDF

```bash
# Ouvrir le URDF et vérifier les chemins
cat awd/data/assets/mini_bdx/urdf/bdx.urdf | grep -i mesh
```

**Solution 2** : Utiliser directement le URDF (moins performant)

Modifier `awd_isaaclab/configs/robots/mini_bdx_cfg.py` :

```python
spawn=ArticulationCfg.SpawnCfg(
    usd_path="awd/data/assets/mini_bdx/urdf/bdx.urdf",  # Utiliser URDF directement
    # ...
)
```

### Problème : placo ne s'installe pas

**Solution** : Installer Eigen3 manuellement

```bash
# Télécharger et compiler Eigen3
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake ..
sudo make install

# Réessayer placo
pip install placo==0.6.2
```

### Problème : GPU out of memory

**Solution** : Réduire le nombre d'environnements

```bash
# Utiliser moins d'environnements
--num_envs 2048  # Au lieu de 4096
--num_envs 1024  # Ou encore moins
```

### Problème : Simulation trop lente

**Solution 1** : Mode headless

```bash
--headless  # Pas de rendu visuel
```

**Solution 2** : Réduire la fréquence de rendu

Modifier dans la config :

```python
sim: SimulationCfg(
    render_interval=10,  # Render every 10 steps
    # ...
)
```

## Structure des Fichiers Après Installation

```
BDX_Awd/
├── awd/                     # Code IsaacGym original (conservé)
├── awd_isaaclab/            # Nouveau code IsaacLab
│   ├── configs/
│   ├── envs/
│   ├── scripts/
│   └── utils/
├── awd/data/assets/
│   ├── mini_bdx/
│   │   ├── urdf/bdx.urdf
│   │   └── bdx.usd          # ← Nouveau (converti)
│   └── go_bdx/
│       ├── go_bdx.urdf
│       └── go_bdx.usd       # ← Nouveau (converti)
├── runs/                    # Checkpoints d'entraînement
├── MIGRATION_GUIDE.md
├── INSTALL.md              # Ce fichier
└── requirements_isaaclab.txt
```

## Prochaines Étapes

1. Consulter [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) pour comprendre les différences API
2. Commencer l'entraînement avec vos propres configurations
3. Adapter les récompenses et paramètres selon vos besoins

## Support

Pour des questions :
1. Consulter la [documentation IsaacLab](https://isaac-sim.github.io/IsaacLab/)
2. Vérifier les [exemples IsaacLab](https://github.com/isaac-sim/IsaacLab/tree/main/source/extensions/omni.isaac.lab_tasks)
3. Forum NVIDIA : https://forums.developer.nvidia.com/c/omniverse/simulation/69

---

**Bonne chance avec votre migration ! 🚀**
