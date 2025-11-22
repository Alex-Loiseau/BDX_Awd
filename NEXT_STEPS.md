# Prochaines Étapes - Installation IsaacLab

## 🚨 Statut Actuel

✅ **Migration du code complète** - Tous les fichiers IsaacLab sont créés
❌ **IsaacLab pas encore installé** - Nécessaire pour exécuter le code

## 📋 Que Faire Maintenant

### Étape 1 : Installer IsaacLab (30-45 min)

IsaacLab n'est pas encore installé sur votre système. C'est la prochaine étape critique.

```bash
# 1. Aller dans le dossier de développements
cd /home/alexandre/Developpements

# 2. Cloner IsaacLab depuis GitHub
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 3. Installer IsaacLab
# Ceci va télécharger et installer tous les composants nécessaires
./isaaclab.sh --install

# Cette commande va :
# - Télécharger Isaac Sim si nécessaire
# - Configurer l'environnement Python
# - Installer toutes les dépendances
# - Compiler les extensions nécessaires
```

**Note** : Cette installation peut prendre 30-45 minutes selon votre connexion internet.

### Étape 2 : Vérifier l'Installation (2 min)

```bash
# Vérifier que IsaacLab est bien installé
cd /home/alexandre/Developpements/IsaacLab
./isaaclab.sh -p -m pip list | grep isaac

# Vous devriez voir plusieurs packages isaac-*
# Par exemple: omni-isaac-lab, isaacsim, etc.
```

### Étape 3 : Installer les Dépendances BDX_Awd (5 min)

```bash
# Activer l'environnement IsaacLab
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Installer les dépendances du projet
cd /home/alexandre/Developpements/BDX_Awd
pip install -r requirements_isaaclab.txt

# Installer Eigen3 pour placo (optionnel)
sudo apt-get update
sudo apt-get install libeigen3-dev
pip install placo==0.6.2
```

### Étape 4 : Convertir les Assets URDF → USD (5 min)

**IMPORTANT** : Cette étape doit être exécutée APRÈS l'installation d'IsaacLab.

```bash
cd /home/alexandre/Developpements/IsaacLab

# Utiliser le wrapper isaaclab.sh pour avoir le bon environnement
./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/convert_assets.py --all

# OU (si vous préférez)
cd /home/alexandre/Developpements/BDX_Awd
source /home/alexandre/Developpements/env_isaaclab/bin/activate
python awd_isaaclab/scripts/convert_assets.py --all
```

**Résultat attendu** :
```
Converting mini_bdx...
✓ Conversion successful: awd/data/assets/mini_bdx/bdx.usd

Converting go_bdx...
✓ Conversion successful: awd/data/assets/go_bdx/go_bdx.usd

Conversion complete: 2/2 succeeded

✓ All conversions successful!
```

### Étape 5 : Test Rapide (2 min)

```bash
cd /home/alexandre/Developpements/IsaacLab

# Test avec 16 environnements (rapide)
./isaaclab.sh -p /home/alexandre/Developpements/BDX_Awd/awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test
```

**Si tout fonctionne**, vous devriez voir :
- La simulation Isaac Sim se lancer
- 16 robots Mini BDX apparaître
- Des informations sur observations/actions
- La simulation s'exécuter pendant ~5 secondes

## 🐛 Dépannage

### Problème : `./isaaclab.sh --install` échoue

**Cause possible** : Isaac Sim 5.1.0 n'est pas compatible avec votre système

**Solution 1** : Vérifier les prérequis système
```bash
# Vérifier le driver NVIDIA
nvidia-smi

# Vérifier Ubuntu
lsb_release -a
```

IsaacLab requiert :
- Ubuntu 20.04/22.04
- NVIDIA GPU avec driver 525+
- 16 GB RAM minimum

**Solution 2** : Consulter les logs d'installation
```bash
cd /home/alexandre/Developpements/IsaacLab
cat _isaac_sim/logs/Kit.log
```

### Problème : Module `omni` non trouvé

**Cause** : Vous essayez d'exécuter le code sans passer par IsaacLab

**Solution** : Toujours utiliser le wrapper `isaaclab.sh`
```bash
# ❌ INCORRECT
python awd_isaaclab/scripts/run_isaaclab.py

# ✅ CORRECT
cd /home/alexandre/Developpements/IsaacLab
./isaaclab.sh -p /path/to/script.py
```

### Problème : Conversion URDF échoue

**Cause 1** : IsaacLab pas installé → Voir ci-dessus

**Cause 2** : URDF invalide ou meshes manquants

**Solution** : Vérifier les fichiers
```bash
# Vérifier que le URDF existe
ls -lh awd/data/assets/mini_bdx/urdf/bdx.urdf

# Vérifier que les meshes existent
ls awd/data/assets/mini_bdx/meshes/
```

**Alternative** : Utiliser directement le URDF (moins performant)

Modifier `awd_isaaclab/configs/robots/mini_bdx_cfg.py` :
```python
spawn=ArticulationCfg.SpawnCfg(
    # Utiliser URDF au lieu d'USD
    usd_path="awd/data/assets/mini_bdx/urdf/bdx.urdf",
    # ... reste de la config
)
```

## 📁 Structure Attendue Après Installation

```
/home/alexandre/Developpements/
├── IsaacLab/                    # ← Nouveau (à cloner)
│   ├── isaaclab.sh
│   ├── source/
│   └── _isaac_sim/
│
├── BDX_Awd/                     # ← Votre projet
│   ├── awd/                     # Code IsaacGym (ancien)
│   ├── awd_isaaclab/            # Code IsaacLab (nouveau) ✅
│   └── awd/data/assets/
│       ├── mini_bdx/
│       │   ├── urdf/bdx.urdf
│       │   └── bdx.usd          # ← Sera créé par convert_assets.py
│       └── go_bdx/
│           ├── go_bdx.urdf
│           └── go_bdx.usd       # ← Sera créé par convert_assets.py
│
└── env_isaaclab/                # Environnement Python Isaac Sim existant
```

## ✅ Checklist d'Installation

- [ ] IsaacLab cloné dans `/home/alexandre/Developpements/IsaacLab`
- [ ] `./isaaclab.sh --install` exécuté avec succès
- [ ] `./isaaclab.sh -p -m pip list | grep isaac` montre les packages
- [ ] `pip install -r requirements_isaaclab.txt` exécuté
- [ ] Assets URDF convertis en USD (ou décision d'utiliser URDF directement)
- [ ] Test rapide exécuté avec succès

## 📞 Besoin d'Aide ?

1. **Erreur lors de l'installation IsaacLab** → Consulter [IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation.html)
2. **Erreur de conversion URDF** → Utiliser URDF directement (voir ci-dessus)
3. **Autre problème** → Consulter [INSTALL.md](INSTALL.md) pour plus de détails

## 🎯 Résumé : Ce Qui Fonctionne Déjà

✅ **Code migré** - Tous les fichiers Python IsaacLab sont créés et prêts
✅ **Documentation** - Guides complets disponibles
✅ **Scripts** - Point d'entrée et conversion prêts

**Il ne manque plus que** : Installation d'IsaacLab sur votre système

---

**Une fois IsaacLab installé, tout le reste est prêt à fonctionner !** 🚀
