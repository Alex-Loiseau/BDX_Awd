#!/usr/bin/env python3
"""Test direct sans passer par isaaclab.sh"""

import sys
import os

# Ajouter les chemins IsaacLab
isaaclab_path = "/home/alexandre/Developpements/IsaacLab/source/isaaclab"
if os.path.exists(isaaclab_path):
    sys.path.insert(0, isaaclab_path)
    print(f"✅ Ajouté {isaaclab_path} au PYTHONPATH")

# Test import
print("\nTest des imports IsaacLab...")
try:
    import isaaclab
    print(f"✅ isaaclab: {isaaclab.__version__}")
    print(f"   Path: {isaaclab.__file__}")

    from isaaclab.envs import DirectRLEnv
    print("✅ isaaclab.envs.DirectRLEnv")

    from isaaclab.utils.configclass import configclass
    print("✅ isaaclab.utils.configclass")

    print("\n✅ Tous les imports fonctionnent!")
    print("\nMaintenant vous pouvez lancer:")
    print("  python awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test")

except ImportError as e:
    print(f"\n❌ Erreur d'import: {e}")
    print("\nLe problème persiste. Vérifiez l'installation d'IsaacLab.")
