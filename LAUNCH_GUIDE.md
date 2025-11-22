# Guide de Lancement - BDX_Awd avec IsaacLab

**Mise à jour** : 2025-11-21 23:00
**IsaacLab** : 0.48.4
**IsaacSim** : 5.1.0

---

## ⚡ IMPORTANT : Changement de Namespace

IsaacLab **0.48.4** utilise le namespace `isaaclab` au lieu de `omni.isaac.lab`.

✅ **Tous les fichiers ont été mis à jour** pour supporter les deux versions.

---

## 🚀 Méthodes de Lancement

### Méthode 1 : Wrapper Simplifié (RECOMMANDÉ)

J'ai créé un wrapper qui contourne les problèmes de terminal :

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test rapide
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test

# Entraînement
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 512 \
    --max_iterations 1000
```

### Méthode 2 : Directement via isaaclab.sh

Si votre terminal supporte les fonctionnalités avancées :

```bash
cd /home/alexandre/Developpements/IsaacLab

# Avec TERM=xterm
TERM=xterm ./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

### Méthode 3 : Via un Terminal Graphique

Ouvrez un terminal graphique (gnome-terminal, xterm, etc.) et lancez :

```bash
cd /home/alexandre/Developpements/IsaacLab
./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand --robot mini_bdx --test
```

---

## 🔧 Problèmes Connus et Solutions

### Problème : `'ansi+tabs': unknown terminal type`

**Cause** : Votre terminal ne supporte pas les fonctionnalités ANSI avancées.

**Solutions** :
1. **Utiliser le wrapper** : `./run_with_isaaclab.sh` (déjà créé)
2. **Exporter TERM** : `export TERM=xterm` avant de lancer
3. **Terminal graphique** : Lancer depuis gnome-terminal ou xterm

### Problème : `No module named 'omni'`

**Cause** : Vous essayez d'exécuter directement avec Python.

**Solution** : Toujours utiliser `isaaclab.sh` ou le wrapper `run_with_isaaclab.sh`

### Problème : `ModuleNotFoundError: No module named 'isaaclab.envs'`

**Cause** : IsaacLab nécessite l'environnement complet d'Isaac Sim.

**Solution** : Utiliser `isaaclab.sh` qui configure tout automatiquement.

---

## ✅ Vérification de l'Installation

Test rapide pour vérifier que tout fonctionne :

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test 1: Vérifier les imports
python test_direct.py

# Test 2: Lancer avec le wrapper
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

**Résultat attendu** : La simulation devrait se lancer avec 4 robots.

---

## 📊 Options de Configuration

### Nombre d'Environnements

```bash
--num_envs 16    # Pour test rapide
--num_envs 512   # Pour entraînement léger
--num_envs 4096  # Pour entraînement complet (nécessite bon GPU)
```

### Mode Headless

```bash
--headless  # Pas de visualisation, plus rapide
```

### Expérience

```bash
--experiment mon_experience  # Nom pour les logs
--max_iterations 10000      # Nombre d'itérations
```

---

## 📝 Exemples Complets

### Test Ultra-Rapide (30 secondes)

```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

### Entraînement Court (5-10 minutes)

```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 256 \
    --max_iterations 500 \
    --experiment test_rapide
```

### Entraînement Complet (plusieurs heures)

```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --experiment mini_bdx_walk_v1
```

---

## 🐛 Debugging

### Activer le mode debug

```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --debug
```

### Voir les logs détaillés

Les logs sont affichés dans la console. Pour les sauvegarder :

```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 512 2>&1 | tee training.log
```

### Vérifier l'utilisation GPU

Dans un terminal séparé :

```bash
watch -n 1 nvidia-smi
```

---

## 📚 Fichiers Créés

- **`run_with_isaaclab.sh`** : Wrapper simplifié pour lancer les scripts
- **`test_direct.py`** : Test des imports IsaacLab
- **`LAUNCH_GUIDE.md`** : Ce guide

---

## 🎯 Prochaines Étapes

1. ✅ Lancer un test rapide (4 envs)
2. ✅ Vérifier que la simulation démarre
3. ✅ Lancer un entraînement court (256 envs, 500 iter)
4. ✅ Analyser les résultats
5. ✅ Ajuster les paramètres
6. ✅ Entraînement complet

---

## 💡 Conseils

1. **Toujours commencer par un test** avec peu d'environnements
2. **Utiliser `--headless`** pour entraînement final
3. **Surveiller le GPU** avec `nvidia-smi`
4. **Sauvegarder les logs** avec `tee`
5. **Tester différents hyperparamètres** avant entraînement long

---

**Vous êtes prêt ! Lancez votre premier test :)**

```bash
cd /home/alexandre/Developpements/BDX_Awd
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test
```
