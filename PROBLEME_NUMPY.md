# Problème NumPy - Isaac Sim 5.1.0

## 🔴 Problème Identifié

Votre environnement `/home/alexandre/Developpements/env_isaaclab` contient une version de NumPy incompatible avec Isaac Sim 5.1.0.

**Erreur** : `AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`

**Cause** : Conflit entre :
- NumPy interne d'Isaac Sim (version ancienne, compatible)
- NumPy installé dans `env_isaaclab` (version récente, incompatible)

## ✅ Solution

**NE PAS utiliser `env_isaaclab`**. Isaac Sim a son propre Python avec toutes les dépendances.

### Option 1 : Utiliser Isaac Sim Python Directement (RECOMMANDÉ)

```bash
# Lancer directement avec le Python d'Isaac Sim
/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

### Option 2 : Wrapper Simplifié

Créer un wrapper qui utilise Isaac Sim Python :

```bash
#!/bin/bash
# run_isaac_direct.sh

ISAAC_SIM="/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
SCRIPT="$1"
shift
ARGS="$@"

"$ISAAC_SIM/python.sh" "$SCRIPT" $ARGS
```

Utilisation :
```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test
```

### Option 3 : Réinstaller env_isaaclab Proprement

Si vous voulez vraiment utiliser `env_isaaclab`, il faut le recréer avec les bonnes versions :

```bash
# ATTENTION : Ceci supprime l'environnement actuel
rm -rf /home/alexandre/Developpements/env_isaaclab

# Créer un nouvel environnement vide
python3.11 -m venv /home/alexandre/Developpements/env_isaaclab

# Activer
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# N'installer QUE les packages nécessaires, SANS numpy/scipy
pip install torch gymnasium rl-games tensorboard
# NE PAS installer numpy, scipy - ils viendront d'Isaac Sim
```

## ⚠️ Ce qu'il NE FAUT PAS Faire

❌ **Ne pas installer numpy dans env_isaaclab**
❌ **Ne pas installer scipy dans env_isaaclab**
❌ **Ne pas mélanger les environnements Python**

## 🎯 Solution Immédiate

Pour tester tout de suite :

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test direct avec Isaac Sim Python
/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
    awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

## 📚 Explication Technique

Isaac Sim embarque ses propres versions de :
- NumPy (version spécifique compatible)
- SciPy
- PyTorch (avec CUDA)
- Toutes les dépendances Omniverse

Quand vous utilisez un environnement virtuel externe qui a ses propres versions, il y a des conflits d'imports.

**La bonne pratique** : Toujours utiliser le Python d'Isaac Sim (`python.sh`) qui a tout préconfiguré.

---

**Prochaine étape** : Tester avec le wrapper direct !
