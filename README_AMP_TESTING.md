# Guide de Test des Environnements AMP

## Vue d'Ensemble

Ce document explique comment tester les environnements AMP (Adversarial Motion Priors) migrés vers IsaacLab.

**État de la migration**: ✅ Complète
**Environnements disponibles**: 6 (3 de base + 3 AMP)

---

## Environnements Disponibles

### Environnements de Base (Testés ✅)

1. **DucklingCommand** - Suivi de commandes de vitesse
   - Test: ✅ Passé
   - Commande: `./run_with_isaaclab.sh DucklingCommand --test --headless`

2. **DucklingHeading** - Contrôle direction mouvement + regard
   - Test: ✅ Passé
   - Commande: `./run_with_isaaclab.sh DucklingHeading --test --headless`

3. **DucklingPerturb** - Entraînement robustesse avec perturbations
   - Test: ✅ Passé
   - Commande: `./run_with_isaaclab.sh DucklingPerturb --test --headless`

### Environnements AMP (À Tester ⏳)

4. **DucklingAMP** - Base AMP avec motion library
   - Test: ⏳ En attente (nécessite données motion)
   - Fichier: `awd_isaaclab/envs/duckling_amp.py`

5. **DucklingAMPTask** - AMP + objectifs de tâche
   - Test: ⏳ En attente (nécessite données motion)
   - Fichier: `awd_isaaclab/envs/duckling_amp_task.py`

6. **DucklingViewMotion** - Visualisation cinématique de mouvements
   - Test: ⏳ En attente (nécessite données motion)
   - Fichier: `awd_isaaclab/envs/duckling_view_motion.py`

---

## Prérequis pour Tester AMP

### 1. Données de Mouvement (Motion Data)

Les environnements AMP nécessitent des fichiers de données mocap au format JSON.

**Format attendu par AMPLoader**:
```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.0333,
  "Frames": [
    {
      "root_pos": [x, y, z],
      "root_rot": [x, y, z, w],
      "joint_positions": [...],
      "joint_velocities": [...]
    },
    ...
  ]
}
```

**Emplacement suggéré**:
```
awd/data/motions/
├── motion_walk.json
├── motion_trot.json
└── motion_run.json
```

### 2. Configuration du Motion File

Modifier `awd_isaaclab/envs/duckling_amp.py`:

```python
@configclass
class DucklingAMPCfg(DucklingCommandCfg):
    # ...

    # Motion file path (MODIFIEZ CECI)
    motion_file: str = "awd/data/motions/your_motion_file.json"
```

Ou bien, créer une variable d'environnement:
```bash
export MOTION_FILE="/path/to/your/motion/data.json"
```

### 3. Configuration des Key Bodies

Vérifier que `key_body_ids` correspond à votre robot dans la config:

```python
# IDs des corps clés pour observations AMP (pieds)
key_body_ids: list = [3, 6, 9, 12]  # FL_foot, FR_foot, RL_foot, RR_foot
```

---

## Comment Tester

### Test 1: Vérification de l'Enregistrement

Vérifier que les environnements sont enregistrés:

```bash
python3 test_amp_envs.py
```

**Résultat attendu**:
```
4. Testing environment registration in run_isaaclab.py...
   ✅ DucklingAMP registered in run_isaaclab.py
   ✅ DucklingAMPTask registered in run_isaaclab.py
   ✅ DucklingViewMotion registered in run_isaaclab.py
```

### Test 2: Test d'Import (nécessite Isaac Sim)

**Note**: Les imports des environnements AMP nécessitent Isaac Sim car ils dépendent des modules `omni`.

Pour vérifier les imports, utilisez le script run_isaaclab.py:

```bash
# Ceci chargera Isaac Sim et tentera de créer l'environnement
./run_with_isaaclab.sh DucklingAMP --test --headless
```

### Test 3: Test Complet avec Données Motion

Une fois les données motion configurées:

```bash
# Mode test rapide (100 steps avec actions aléatoires)
./run_with_isaaclab.sh DucklingAMP --test --headless --num_envs 16

# Mode test avec visualisation
./run_with_isaaclab.sh DucklingViewMotion --test --num_envs 4

# Mode entraînement complet
./run_with_isaaclab.sh DucklingAMP --train --headless --num_envs 4096
```

---

## Commandes de Test Détaillées

### DucklingAMP (Base AMP)

```bash
# Test basique headless
./run_with_isaaclab.sh DucklingAMP --test --headless --num_envs 16

# Test avec visualisation
./run_with_isaaclab.sh DucklingAMP --test --num_envs 4

# Entraînement
./run_with_isaaclab.sh DucklingAMP --train --headless --num_envs 4096 --max_iterations 10000
```

**Observations attendues**:
- Dimension: 197D (AMP multi-frame: 3 frames × 47D base + extra)
- AMP discriminator: 138D
- Actions: 12D (positions angulaires joints)

### DucklingAMPTask (AMP + Task)

```bash
# Test avec tâche spécifique
./run_with_isaaclab.sh DucklingAMPTask --test --headless --num_envs 16
```

**Observations attendues**:
- Dimension: Variable (197D AMP + task observations)
- Dépend de la tâche implémentée dans sous-classe

### DucklingViewMotion (Visualisation)

```bash
# Visualisation cinématique (recommandé avec GUI)
./run_with_isaaclab.sh DucklingViewMotion --test --num_envs 1

# Headless pour debug
./run_with_isaaclab.sh DucklingViewMotion --test --headless --num_envs 4
```

**Comportement attendu**:
- Robot suit exactement la trajectoire motion data
- Pas de contrôle physique (mode cinématique)
- Sauvegarde positions clés dans `anim.npy` et `sim.npy`

---

## Diagnostics d'Erreurs

### Erreur: "No motion file specified"

**Cause**: Le chemin `motion_file` n'est pas configuré

**Solution**:
```python
# Dans duckling_amp.py
motion_file: str = "path/to/your/motion.json"
```

### Erreur: "AMPLoader failed to load motion data"

**Cause**: Format de fichier motion incorrect ou fichier manquant

**Solutions**:
1. Vérifier que le fichier existe
2. Vérifier le format JSON (voir section Prérequis)
3. Vérifier les permissions de lecture

### Erreur: "Key body IDs out of range"

**Cause**: Les `key_body_ids` ne correspondent pas au robot

**Solution**: Ajuster `key_body_ids` dans la configuration:
```python
# Pour go_bdx, vérifier les IDs corrects
key_body_ids: list = [3, 6, 9, 12]  # Adapter selon votre robot
```

### Erreur: "AMP observations dimension mismatch"

**Cause**: Les observations AMP ne correspondent pas à la dimension attendue

**Diagnostic**:
```python
# Dans build_amp_observations(), vérifier:
# - root obs (13D)
# - dof pos (12D)
# - dof vel (12D)
# - key body pos (4 × 3 = 12D)
# Total: 13 + 12 + 12 + 12 = 49D par frame (erreur possible)
```

**Note**: Le calcul exact dépend des flags `local_root_obs` et `root_height_obs`.

---

## Vérifications Post-Migration

### Checklist de Validation

- [x] ✅ Environnements importables
- [x] ✅ Configurations créables
- [x] ✅ Enregistrés dans run_isaaclab.py
- [x] ✅ Utilitaires (torch_utils, motion_lib) fonctionnels
- [x] ✅ Fonctions JIT compilables
- [ ] ⏳ Test avec vraies données motion
- [ ] ⏳ Validation observations AMP (138D)
- [ ] ⏳ Test entraînement complet
- [ ] ⏳ Validation motion synchronization

### Fichiers Migrés

**Environnements**:
- [x] `awd_isaaclab/envs/duckling_command_env.py` (base)
- [x] `awd_isaaclab/envs/duckling_heading_env.py`
- [x] `awd_isaaclab/envs/duckling_perturb_env.py`
- [x] `awd_isaaclab/envs/duckling_amp.py` (AMP base)
- [x] `awd_isaaclab/envs/duckling_amp_task.py` (AMP + task)
- [x] `awd_isaaclab/envs/duckling_view_motion.py` (motion viz)

**Utilitaires**:
- [x] `awd_isaaclab/utils/torch_utils.py`
- [x] `awd_isaaclab/utils/motion_lib.py`
- [x] `awd_isaaclab/utils/bdx/amp_motion_loader.py`
- [x] `awd_isaaclab/utils/bdx/pose3d.py`
- [x] `awd_isaaclab/utils/bdx/motion_util.py`
- [x] `awd_isaaclab/utils/bdx/utils.py`

**Configuration**:
- [x] `awd_isaaclab/configs/robots/go_bdx_cfg.py` (USD)
- [x] `awd_isaaclab/scripts/run_isaaclab.py` (tous envs enregistrés)

---

## Prochaines Étapes

### Court Terme (Tests)

1. **Préparer données motion**
   - Obtenir ou créer fichiers JSON mocap
   - Valider format avec AMPLoader
   - Configurer chemin dans DucklingAMPCfg

2. **Tester DucklingViewMotion**
   - Environnement le plus simple (pas de contrôle)
   - Valide motion library et synchronisation
   - Commande: `./run_with_isaaclab.sh DucklingViewMotion --test`

3. **Tester DucklingAMP**
   - Valider initialisation depuis motion data
   - Vérifier observations AMP (138D)
   - Tester discriminateur (fetch_amp_obs_demo)

4. **Tester DucklingAMPTask**
   - Valider AMP + task observations
   - Implémenter tâche spécifique si nécessaire

### Moyen Terme (Optimisation)

5. **Benchmark performance**
   - Comparer vitesse avec IsaacGym
   - Optimiser si nécessaire
   - Tester scaling (4096 envs)

6. **Entraînement complet**
   - Lancer entraînement AMP
   - Valider convergence
   - Comparer avec résultats IsaacGym

### Long Terme (Nettoyage)

7. **Nettoyer ancien code**
   - Supprimer `awd/envs/` (IsaacGym)
   - Conserver uniquement utilitaires BDX réutilisés
   - Documenter architecture finale

8. **Résoudre warnings USD**
   - Warnings visuels non critiques
   - À adresser pour polish final

---

## Ressources

### Documentation

- **IsaacLab**: https://isaac-sim.github.io/IsaacLab/
- **Isaac Sim**: https://docs.omniverse.nvidia.com/isaacsim/
- **MIGRATION_STATUS.md**: Suivi détaillé de la migration

### Scripts Utiles

- `run_with_isaaclab.sh`: Script principal de lancement
- `test_amp_envs.py`: Test enregistrement environnements
- `test_amp_import.sh`: Test imports (nécessite Isaac Sim)

### Fichiers de Référence

- `awd_isaaclab/envs/duckling_amp.py`: Documentation inline complète
- `awd_isaaclab/utils/motion_lib.py`: API motion library
- `awd_isaaclab/utils/bdx/amp_motion_loader.py`: Format données attendu

---

## Support

Pour questions ou problèmes:

1. Consulter `MIGRATION_STATUS.md` pour détails migration
2. Vérifier logs Isaac Sim pour erreurs détaillées
3. Valider format données motion avec AMPLoader
4. Vérifier configuration robot (key_body_ids, etc.)

**Note**: Cette migration est complète au niveau code. Les tests complets nécessitent:
- ✅ Code migré
- ✅ Configurations créées
- ⏳ Données motion (à fournir par utilisateur)

---

**Version**: 1.0
**Date**: 2025-11-22
**Status**: Migration complète - En attente données motion pour tests
