# Démarrage Rapide - BDX_Awd avec IsaacLab

**Date**: 2025-11-21
**Statut**: Prêt à tester

---

## 🎯 Objectif

Lancer votre simulation BDX_Awd migrée vers IsaacLab avec Isaac Sim 5.1.0.

---

## ⚡ Solution Rapide (RECOMMANDÉE)

### Étape 1 : Test Immédiat

Lancez directement avec le Python d'Isaac Sim (évite les conflits NumPy) :

```bash
cd /home/alexandre/Developpements/BDX_Awd

./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

**Résultat attendu** : Fenêtre Isaac Sim avec 4 robots Mini BDX qui bougent aléatoirement.

### Étape 2 : Vérifier que ça fonctionne

Si vous voyez les robots sans erreurs NumPy → ✅ **C'est bon !**

Vous pouvez passer aux tests d'entraînement.

---

## 🚀 Commandes de Test

### Test avec 4 robots (validation rapide)

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 4
```

### Test avec Go BDX

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot go_bdx \
    --test \
    --num_envs 4
```

---

## 🎓 Commandes d'Entraînement

### Entraînement court (256 robots, 500 itérations)

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 256 \
    --max_iterations 500 \
    --experiment test_mini_bdx
```

**Durée estimée** : 10-15 minutes
**Résultats** : Sauvegardés dans `logs/rl_games/DucklingCommand/`

### Entraînement complet (4096 robots, headless)

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --experiment mini_bdx_walk_v1
```

**Durée estimée** : Plusieurs heures
**Nécessite** : GPU puissant (RTX 3090+ recommandé)

---

## 📊 Visualiser un Modèle Entraîné

Après l'entraînement, relancez en mode lecture :

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --play \
    --checkpoint logs/rl_games/DucklingCommand/mini_bdx_walk_v1/nn/model.pth \
    --num_envs 16
```

---

## ❓ Questions Fréquentes

### Pourquoi utiliser `run_isaac_direct.sh` ?

Isaac Sim 5.1.0 embarque sa propre version de NumPy. Si vous utilisez un environnement virtuel externe (comme `env_isaaclab`), il y a des conflits de versions.

**Solution** : Toujours utiliser le Python d'Isaac Sim via ce wrapper.

### Puis-je utiliser mon environnement virtuel `env_isaaclab` ?

**Non recommandé**. Isaac Sim a déjà tous les packages nécessaires (NumPy, PyTorch, etc.).

Si vous voulez vraiment utiliser un venv, consultez [PROBLEME_NUMPY.md](PROBLEME_NUMPY.md) pour les options avancées.

### Comment changer le nombre de robots ?

Utilisez `--num_envs <nombre>` :
- Test rapide : `--num_envs 4`
- Test moyen : `--num_envs 64`
- Entraînement : `--num_envs 256` ou plus

### La simulation est lente

Options pour accélérer :
1. Réduire `--num_envs`
2. Utiliser `--headless` (pas de visualisation)
3. Vérifier que le GPU est utilisé : `nvidia-smi`

### Erreur "Task not found: DucklingCommand"

Vérifiez que vous êtes dans le bon répertoire :
```bash
cd /home/alexandre/Developpements/BDX_Awd
```

### Erreur "python: command not found"

Le wrapper `run_isaac_direct.sh` cherche Isaac Sim à :
```
/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
```

Si votre installation est ailleurs, éditez la ligne 5 du script.

---

## 🔧 Scripts Disponibles

| Script | Usage |
|--------|-------|
| `run_isaac_direct.sh` | **RECOMMANDÉ** - Lance avec Isaac Sim Python directement |
| `run_with_isaac_configured.sh` | Alternative avec IsaacLab wrapper (peut avoir problèmes) |
| `setup_isaaclab.sh` | Configure IsaacLab (optionnel pour approche directe) |

---

## 📚 Documentation Complète

- **[PROBLEME_NUMPY.md](PROBLEME_NUMPY.md)** - Explication détaillée du conflit NumPy
- **[INSTALLATION_FINALE.md](INSTALLATION_FINALE.md)** - Guide d'installation complet
- **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - Options de lancement détaillées
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Détails techniques de la migration

---

## 🐛 En Cas de Problème

### 1. Vérifier Isaac Sim

```bash
ls /isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh
```

Si le fichier n'existe pas, vérifiez votre installation d'Isaac Sim.

### 2. Vérifier les fichiers migrés

```bash
ls -la awd_isaaclab/scripts/run_isaaclab.py
ls -la awd_isaaclab/envs/
ls -la awd_isaaclab/configs/
```

Tous ces fichiers doivent exister.

### 3. Logs complets

Si vous avez une erreur, regardez les dernières lignes :
```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand --robot mini_bdx --test 2>&1 | tail -50
```

### 4. Test minimal Python

Vérifiez que le Python d'Isaac Sim fonctionne :
```bash
/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh --version
```

---

## ✅ Checklist de Démarrage

- [ ] Isaac Sim installé à `/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64`
- [ ] Vous êtes dans `/home/alexandre/Developpements/BDX_Awd`
- [ ] Lancer le test avec 4 robots : `./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test --num_envs 4`
- [ ] Voir les robots bouger dans Isaac Sim
- [ ] Si succès → Lancer un entraînement court
- [ ] Analyser les résultats dans `logs/`

---

## 🎯 Prochaines Étapes

Une fois le test validé :

1. **Entraînement court** pour valider le pipeline complet
2. **Ajuster les hyperparamètres** dans `awd_isaaclab/configs/robots/`
3. **Migration des autres tâches** (AMP, Heading, Perturb) si nécessaire
4. **Optimisation** (vitesse, nombre d'environnements, etc.)

---

**Commencez par le test rapide ! 🚀**

```bash
cd /home/alexandre/Developpements/BDX_Awd
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test --num_envs 4
```
