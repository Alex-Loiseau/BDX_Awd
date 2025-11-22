#!/usr/bin/env python3
"""Test if IsaacLab is installed and accessible."""

import sys
import os

print("=" * 80)
print("Test d'installation IsaacLab")
print("=" * 80)

# Test 1: Import omni.isaac.lab
print("\n1. Test import omni.isaac.lab...")
try:
    import omni.isaac.lab
    print(f"   ✅ SUCCESS: IsaacLab trouvé!")
    print(f"   Version: {omni.isaac.lab.__version__}")
    print(f"   Path: {omni.isaac.lab.__file__}")
    isaaclab_installed = True
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
    print("\n   IsaacLab n'est PAS installé dans cet environnement Python.")
    print("\n   Solutions:")
    print("   1. Installer IsaacLab:")
    print("      cd /home/alexandre/Developpements/IsaacLab")
    print("      ./isaaclab.sh --install")
    print("\n   2. OU utiliser l'environnement Python d'IsaacLab:")
    print("      cd /home/alexandre/Developpements/IsaacLab")
    print("      ./isaaclab.sh -p /path/to/script.py")
    isaaclab_installed = False

# Test 2: Vérifier les chemins Python
print("\n2. Chemins Python actuels:")
for i, path in enumerate(sys.path[:5]):
    print(f"   [{i}] {path}")

# Test 3: Vérifier l'environnement
print("\n3. Environnement Python:")
print(f"   Executable: {sys.executable}")
print(f"   Version: {sys.version}")
print(f"   Prefix: {sys.prefix}")

# Test 4: Liste des packages isaac/omni installés
print("\n4. Packages Isaac/Omni installés:")
try:
    import pkg_resources
    isaac_packages = [
        pkg for pkg in pkg_resources.working_set
        if 'isaac' in pkg.key.lower() or 'omni' in pkg.key.lower()
    ]
    if isaac_packages:
        for pkg in isaac_packages[:10]:
            print(f"   - {pkg.key} ({pkg.version})")
    else:
        print("   ❌ Aucun package isaac/omni trouvé")
except Exception as e:
    print(f"   Erreur: {e}")

# Résumé
print("\n" + "=" * 80)
if isaaclab_installed:
    print("✅ IsaacLab est INSTALLÉ et accessible")
    print("\nVous pouvez maintenant:")
    print("  python awd_isaaclab/scripts/convert_assets.py --all")
else:
    print("❌ IsaacLab n'est PAS accessible")
    print("\nAction requise: Installer IsaacLab")
print("=" * 80)
