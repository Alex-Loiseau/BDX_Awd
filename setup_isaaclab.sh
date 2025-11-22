#!/bin/bash
# Script pour configurer IsaacLab avec Isaac Sim existant

set -e

echo "========================================"
echo "Configuration d'IsaacLab"
echo "========================================"

# Chemins
ISAAC_SIM_PATH="/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
ISAACLAB_PATH="/home/alexandre/Developpements/IsaacLab"

# Vérifier qu'Isaac Sim existe
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "❌ ERROR: Isaac Sim non trouvé à $ISAAC_SIM_PATH"
    exit 1
fi

echo "✅ Isaac Sim trouvé: $ISAAC_SIM_PATH"

# Aller dans IsaacLab
cd "$ISAACLAB_PATH"

# Créer un lien symbolique vers Isaac Sim
if [ ! -L "_isaac_sim" ]; then
    echo "📁 Création du lien symbolique _isaac_sim → $ISAAC_SIM_PATH"
    ln -s "$ISAAC_SIM_PATH" _isaac_sim
else
    echo "✅ Lien symbolique _isaac_sim existe déjà"
fi

# Vérifier python.sh
if [ -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "✅ Python Isaac Sim trouvé: $ISAAC_SIM_PATH/python.sh"
else
    echo "❌ ERROR: python.sh non trouvé dans Isaac Sim"
    echo "   Attendu: $ISAAC_SIM_PATH/python.sh"
    exit 1
fi

# Installer IsaacLab
echo ""
echo "📦 Installation d'IsaacLab..."
echo "   Ceci va installer les packages Python nécessaires..."
echo ""

# Utiliser Isaac Sim Python pour installer IsaacLab
"$ISAAC_SIM_PATH/python.sh" -m pip install -e "$ISAACLAB_PATH/source/isaaclab"

echo ""
echo "========================================"
echo "✅ Configuration terminée !"
echo "========================================"
echo ""
echo "Vous pouvez maintenant tester avec:"
echo "  cd /home/alexandre/Developpements/BDX_Awd"
echo "  ./run_with_isaac_configured.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test"
