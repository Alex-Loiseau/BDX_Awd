# Conversion Manuelle URDF → USD

## Problème Actuel

Le script automatique de conversion nécessite `isaaclab.sh` qui a un problème avec le type de terminal dans cet environnement.

## Solution : Conversion Manuelle

Vous avez **deux options** :

### Option 1 : Utiliser URDF Directement (RECOMMANDÉ - Plus Simple)

IsaacLab peut charger directement les fichiers URDF sans conversion USD. C'est moins performant mais fonctionne parfaitement.

**Aucune action requise** - Les configurations sont déjà prêtes à utiliser les URDF !

Les fichiers de configuration dans `awd_isaaclab/configs/robots/` sont configurés pour chercher :
- `awd/data/assets/mini_bdx/bdx.usd` en premier
- Si absent, tombent automatiquement sur `awd/data/assets/mini_bdx/urdf/bdx.urdf`

### Option 2 : Conversion USD via Interface Isaac Sim

Si vous voulez vraiment les fichiers USD pour de meilleures performances :

#### Étape 1 : Lancer Isaac Sim

```bash
cd /home/alexandre/Developpements/IsaacLab
# Lancer l'interface graphique Isaac Sim
./isaaclab.sh --gui
```

#### Étape 2 : Convertir dans l'interface

1. **Menu** : `Isaac Utils` → `URDF Importer`
2. **Input File** : Naviguer vers `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/mini_bdx/urdf/bdx.urdf`
3. **Output** : `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/mini_bdx/bdx.usd`
4. **Options** :
   - ✅ `Fix Base Link` : Non
   - ✅ `Make Instanceable` : Oui (pour performance)
   - ✅ `Import Inertia` : Oui
5. **Cliquer** : `Import`

6. Répéter pour Go BDX si nécessaire

#### Étape 3 : Sauvegarder

Les fichiers USD seront créés automatiquement.

### Option 3 : Script Python Standalone

Créer un script Python simple pour la conversion :

```python
#!/usr/bin/env python3
# convert_urdf_standalone.py

import subprocess
import sys

urdf_files = [
    ("awd/data/assets/mini_bdx/urdf/bdx.urdf", "awd/data/assets/mini_bdx/bdx.usd"),
    ("awd/data/assets/go_bdx/go_bdx.urdf", "awd/data/assets/go_bdx/go_bdx.usd"),
]

for urdf_path, usd_path in urdf_files:
    print(f"Converting {urdf_path} → {usd_path}")

    cmd = [
        "/home/alexandre/Developpements/IsaacLab/isaaclab.sh",
        "-p",
        "-m", "omni.isaac.lab.utils.assets.urdf_converter",
        "--input", urdf_path,
        "--output", usd_path,
        "--make-instanceable"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Success: {usd_path}")
    else:
        print(f"❌ Failed: {result.stderr}")
```

## Recommandation

**👉 Utilisez l'Option 1 (URDF direct)**

Vous pouvez commencer immédiatement à tester et entraîner sans conversion :

```bash
cd /home/alexandre/Developpements/BDX_Awd

# Test avec URDF directement
python awd_isaaclab/scripts/run_isaaclab.py \
    --task DucklingCommand \
    --robot mini_bdx \
    --test \
    --num_envs 16
```

Si vous avez des problèmes de performance plus tard, vous pourrez toujours convertir en USD.

## Vérification

Pour vérifier que tout est prêt sans conversion :

```bash
# Vérifier que les URDF existent
ls -lh awd/data/assets/mini_bdx/urdf/bdx.urdf
ls -lh awd/data/assets/go_bdx/go_bdx.urdf

# Tester directement
python awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test
```

---

**Conseil** : Commencez avec les URDF. La conversion USD est une optimisation que vous pourrez faire plus tard si nécessaire.
