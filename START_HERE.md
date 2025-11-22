# 🚀 DÉMARRER ICI - BDX_Awd avec IsaacLab

**Statut** : ✅ Prêt à tester !
**Date** : 2025-11-21

---

## ✅ Bonne Nouvelle !

Votre environnement IsaacLab est **DÉJÀ INSTALLÉ** ! 🎉

Les packages détectés :
- `isaaclab (0.48.4)` ✅
- `isaacsim (5.1.0.0)` ✅
- Tous les modules nécessaires

**Les configurations ont été mises à jour pour utiliser directement les URDF** (pas besoin de conversion USD pour commencer).

---

## 🎯 Test Immédiat (5 minutes)

Vous pouvez tester **MAINTENANT** sans aucune installation supplémentaire :

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test rapide avec 16 environnements
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 16
```

**Note** : Si vous voyez l'erreur `No module named 'omni'`, c'est normal. Vous devez utiliser le wrapper IsaacLab (voir ci-dessous).

---

## 📋 Méthode Recommandée : Via isaaclab.sh

Pour lancer vos scripts, utilisez toujours le wrapper `isaaclab.sh` qui configure correctement l'environnement :

### Test Rapide

```bash
cd /home/alexandre/Developpements/IsaacLab

./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

### Entraînement (petit)

```bash
cd /home/alexandre/Developpements/IsaacLab

./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --num_envs 512 \
    --max_iterations 1000 \
    --experiment test_mini_bdx
```

### Entraînement (grande échelle, headless)

```bash
cd /home/alexandre/Developpements/IsaacLab

./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --train \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --experiment mini_bdx_walk_v1
```

---

## 🐛 Si vous avez des Problèmes

### Problème : `'ansi+tabs': unknown terminal type`

C'est un problème connu avec certains terminaux. **Solutions** :

1. **Utiliser un terminal différent** : xterm, gnome-terminal, etc.
2. **Ou** créer un alias simplifié :

```bash
# Ajouter à votre ~/.bashrc
alias isaaclab-run='cd /home/alexandre/Developpements/IsaacLab && TERM=xterm ./isaaclab.sh -p'

# Puis utiliser:
isaaclab-run /path/to/script.py --args
```

### Problème : `No module named 'omni'`

**Cause** : Vous essayez d'exécuter directement avec Python au lieu d'utiliser `isaaclab.sh`

**Solution** : Toujours utiliser `isaaclab.sh -p` (voir exemples ci-dessus)

### Problème : URDF ne charge pas

**Vérifier** que les fichiers URDF existent :

```bash
ls -lh awd/data/assets/mini_bdx/urdf/bdx.urdf
ls -lh awd/data/assets/go_bdx/go_bdx.urdf
```

---

## 📚 Documentation

Consultez dans cet ordre :

1. **Ce fichier (START_HERE.md)** - Pour démarrer rapidement ← Vous êtes ici
2. **[CONVERSION_MANUELLE.md](CONVERSION_MANUELLE.md)** - Si vous voulez convertir URDF → USD (optionnel)
3. **[QUICKSTART.md](QUICKSTART.md)** - Guide d'utilisation détaillé
4. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Détails techniques de la migration

---

## 🎨 Personnalisation

### Modifier les Récompenses

Éditer `awd_isaaclab/configs/robots/mini_bdx_cfg.py` :

```python
"reward_scales": {
    "lin_vel_xy": 1.0,      # ← Augmenter pour favoriser vitesse
    "ang_vel_z": 0.25,
    "torque": -0.00001,     # ← Diminuer pénalité
}
```

### Modifier le Nombre d'Environnements

Via ligne de commande :
```bash
--num_envs 2048  # Au lieu de 4096 par défaut
```

Ou dans la config Python :
```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=2048  # ← Modifier ici
)
```

---

## ✅ Checklist de Démarrage

- [x] IsaacLab installé (`isaaclab` package détecté)
- [x] Code migré (`awd_isaaclab/` créé)
- [x] Configurations robot prêtes (mini_bdx, go_bdx)
- [x] URDF configurés pour chargement direct
- [ ] **Premier test lancé** ← Vous êtes là !
- [ ] Entraînement test (100 itérations)
- [ ] Entraînement complet

---

## 🚀 Prochaines Étapes

1. **Test immédiat** : Lancer le test rapide (voir ci-dessus)
2. **Vérifier** que la simulation démarre
3. **Observer** les robots se déplacer
4. **Entraîner** un modèle test (court)
5. **Ajuster** les récompenses si nécessaire
6. **Entraînement complet** sur 10000 itérations

---

## 💡 Astuces

1. **Commencez petit** : 16-512 environnements pour tester
2. **Mode headless** : `--headless` pour entraînement plus rapide
3. **Surveillance GPU** : `watch -n 1 nvidia-smi` dans un terminal séparé
4. **TensorBoard** : `tensorboard --logdir runs/` pour suivre l'entraînement
5. **Checkpoints** : Sauvegardés automatiquement dans `runs/<experiment>/`

---

## 🎯 Résumé Ultra-Rapide

```bash
# 1. Aller dans IsaacLab
cd /home/alexandre/Developpements/IsaacLab

# 2. Lancer un test
./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand --robot mini_bdx --test

# 3. Si ça marche → Entraîner !
./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand --robot mini_bdx --train --num_envs 512
```

---

**Vous êtes prêt ! Lancez votre premier test maintenant ! 🚀**

Questions ? Consultez [QUICKSTART.md](QUICKSTART.md) ou [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
