# Guide : Conversion URDF → USD avec Isaac Sim GUI

## Pourquoi USD plutôt qu'URDF ?

- **Performance** : Chargement 10x plus rapide
- **Sol intégré** : Vous pouvez ajouter le sol directement dans le USD à la bonne hauteur
- **Préprocessing** : Physics et collision déjà calculées

## Étapes de Conversion

### 1. Lancer Isaac Sim

```bash
cd /isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
./isaac-sim.sh
```

### 2. Importer le URDF

1. Dans Isaac Sim, aller dans **File → Import**
2. Dans le dialogue d'import, sélectionner **URDF** comme type de fichier
3. Navigator vers votre fichier URDF :
   - Go BDX: `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/go_bdx/go_bdx.urdf`
   - Mini BDX: `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/mini_bdx/urdf/bdx.urdf`

### 3. Configurer les Options d'Import

Dans la fenêtre d'import URDF, configurez :

```
Import Options:
☐ Merge Fixed Joints: False
☐ Fix Base Link: False
☐ Import Inertia Tensor: True
☐ Self Collision: False

Joint Drive Configuration:
- Drive Type: Position
- Stiffness: 40.0 (sera écrasé par les actuators dans le code)
- Damping: 1.5
- Max Force: 100.0

Scale: 1.0
```

Cliquez sur **Import**.

### 4. Ajouter un Sol (Ground Plane)

C'est l'étape **IMPORTANTE** - c'est pour ça qu'on utilise USD !

1. Dans le menu : **Create → Physics → Ground Plane**
2. Dans le **Property Panel** à droite, ajustez la position du sol :
   - Pour **Go BDX** : Z = 0.0 (robot au sol)
   - Pour **Mini BDX** : Z = 0.0 (à ajuster selon la hauteur du robot)

3. Configurez les propriétés physiques du sol :
   - **Static Friction**: 1.0
   - **Dynamic Friction**: 1.0
   - **Restitution**: 0.0

4. Optionnel : Ajustez la taille du sol (par défaut 100m x 100m suffit)

### 5. Positionner le Robot

1. Sélectionnez le robot dans la hiérarchie (généralement nommé `/bdx` ou `/go_bdx`)
2. Dans le **Property Panel**, ajustez la position initiale :
   - **Go BDX** :
     - Position: (0, 0, 0)
     - Rotation: (0, 0, 0)
   - **Mini BDX** :
     - Position: (0, 0, 0.18)
     - Rotation: (0, -4.6°, 0) ou (0, -0.08 rad, 0)

### 6. Vérifier les Physiques

1. Cliquez sur le bouton **Play** (▶️) en haut
2. Le robot devrait :
   - Rester au-dessus du sol (pas tomber à travers)
   - Tomber doucement avec la gravité
   - Entrer en collision avec le sol

3. Si le robot traverse le sol :
   - Vérifiez que le Ground Plane a bien **Collision** activé
   - Vérifiez la position Z du sol
   - Vérifiez que le robot a bien des colliders

### 7. Sauvegarder en USD

1. **File → Save As...**
2. Sauvegarder dans le même dossier que l'URDF :
   - Go BDX: `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/go_bdx/go_bdx.usd`
   - Mini BDX: `/home/alexandre/Developpements/BDX_Awd/awd/data/assets/mini_bdx/urdf/bdx.usd`

### 8. Mettre à Jour la Configuration IsaacLab

Une fois le USD créé, modifiez `go_bdx_cfg.py` (ou `mini_bdx_cfg.py`) :

```python
from isaaclab.sim.spawners.from_files import UsdFileCfg

GO_BDX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        # USD au lieu de URDF !
        usd_path="awd/data/assets/go_bdx/go_bdx.usd",
        activate_contact_sensors=True,
        rigid_props=schemas.RigidBodyPropertiesCfg(
            # ... même config qu'avant
        ),
        articulation_props=schemas.ArticulationRootPropertiesCfg(
            # ... même config qu'avant
        ),
    ),
    # ... reste de la config identique
)
```

**Note** : Avec USD, vous n'avez **plus besoin** de créer le sol dans `_setup_scene()` car il est déjà dans le fichier USD !

Supprimez cette partie de `duckling_command_env.py` :

```python
# PLUS BESOIN de ça avec USD :
# spawn_ground_plane(...)
```

## Vérification

Pour vérifier que le USD fonctionne :

```bash
./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot go_bdx --test --num_envs 2
```

Vous devriez voir :
- Le robot se charger beaucoup plus vite
- Le sol visible dans la scène
- Le robot qui tombe et entre en collision avec le sol

## Avantages de cette Approche

✅ **Contrôle total** : Vous ajustez le sol exactement où vous voulez
✅ **Performance** : Chargement USD beaucoup plus rapide qu'URDF
✅ **Prévisualisation** : Vous voyez le résultat dans Isaac Sim avant l'entraînement
✅ **Réutilisable** : Le USD contient tout (robot + sol + physics)
✅ **Simplicité** : Plus besoin de créer le sol en code

## Troubleshooting

### Le robot traverse le sol
→ Vérifiez que les collisions sont activées sur le Ground Plane

### Le robot est trop haut/bas
→ Ajustez la position Z du robot ou du sol dans Isaac Sim, puis sauvegardez à nouveau

### Erreur de chargement USD
→ Vérifiez le chemin dans la config (relatif depuis le dossier projet)

### Physics ne fonctionnent pas
→ Vérifiez que "Import Inertia Tensor" était coché lors de l'import

---

**Prêt pour la conversion ?** Lancez Isaac Sim et suivez les étapes ci-dessus ! 🚀
