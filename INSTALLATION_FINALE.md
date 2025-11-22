# Installation Finale - BDX_Awd avec IsaacLab

**Date** : 2025-11-21
**Status** : Configuration nécessaire

---

## 🎯 Situation Actuelle

✅ **Isaac Sim installé** : `/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64`
✅ **IsaacLab cloné** : `/home/alexandre/Developpements/IsaacLab`
✅ **Code migré** : `awd_isaaclab/` complet
✅ **Imports corrigés** : Compatible IsaacLab 0.48.4

❌ **IsaacLab pas configuré** : Lien vers Isaac Sim manquant

---

## 🚀 Installation en 2 Étapes

### Étape 1 : Configurer IsaacLab (5 minutes)

Lancez le script de configuration automatique :

```bash
cd /home/alexandre/Developpements/BDX_Awd
./setup_isaaclab.sh
```

Ce script va :
1. ✅ Créer un lien symbolique `_isaac_sim` pointant vers Isaac Sim
2. ✅ Installer IsaacLab avec pip dans l'environnement Isaac Sim
3. ✅ Vérifier que tout est prêt

### Étape 2 : Tester (30 secondes)

```bash
./run_with_isaac_configured.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

**Résultat attendu** : La simulation démarre avec 4 robots Mini BDX

---

## 📋 Commandes Complètes

### Test Rapide (4 robots)

```bash
cd /home/alexandre/Developpements/BDX_Awd

./run_with_isaac_configured.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

### Entraînement Court (256 robots, 500 itérations)

```bash
./run_with_isaac_configured.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 256 \
    --max_iterations 500 \
    --experiment test_mini_bdx
```

### Entraînement Complet (4096 robots, headless)

```bash
./run_with_isaac_configured.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --experiment mini_bdx_walk_v1
```

---

## 🔧 Scripts Créés

| Script | Description |
|--------|-------------|
| `setup_isaaclab.sh` | Configure IsaacLab avec Isaac Sim (à lancer une fois) |
| `run_with_isaac_configured.sh` | Lance les scripts avec IsaacLab |
| `run_with_isaaclab.sh` | Ancien wrapper (peut avoir problèmes) |
| `test_direct.py` | Test de diagnostic |

---

## ✅ Checklist

- [ ] Lancer `./setup_isaaclab.sh`
- [ ] Vérifier qu'il se termine sans erreur
- [ ] Vérifier que le lien `_isaac_sim` est créé dans IsaacLab
- [ ] Tester avec 4 environnements
- [ ] Si ça fonctionne → Entraînement !

---

## 🐛 Si Problèmes

### setup_isaaclab.sh échoue

**Vérifier** qu'Isaac Sim est bien à `/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64` :

```bash
ls /isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh
```

Si le chemin est différent, éditez `setup_isaaclab.sh` ligne 11 :
```bash
ISAAC_SIM_PATH="/votre/chemin/vers/isaac-sim"
```

### Test échoue avec "python: command not found"

Le lien symbolique n'a pas été créé. Lancez :

```bash
cd /home/alexandre/Developpements/IsaacLab
ln -s /isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64 _isaac_sim
```

### Autre erreur

Consultez les logs complets et vérifiez :
1. Isaac Sim est bien installé
2. Python d'Isaac Sim fonctionne : `/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh --version`
3. IsaacLab est bien cloné

---

## 📚 Documentation

- **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - Guide de lancement détaillé
- **[START_HERE.md](START_HERE.md)** - Guide général
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Détails techniques

---

## 🎯 Après Installation

Une fois `setup_isaaclab.sh` terminé, vous pourrez :

1. ✅ Tester l'environnement
2. ✅ Lancer des entraînements
3. ✅ Visualiser les résultats
4. ✅ Ajuster les hyperparamètres

---

**Commencez par lancer `./setup_isaaclab.sh` ! 🚀**
