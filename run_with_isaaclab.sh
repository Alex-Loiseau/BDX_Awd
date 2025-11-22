#!/bin/bash
# Wrapper pour lancer des scripts avec IsaacLab sans problème de terminal

# Configuration
ISAACLAB_DIR="/home/alexandre/Developpements/IsaacLab"
SCRIPT_PATH="$1"
shift  # Enlever le premier argument (le script)
SCRIPT_ARGS="$@"  # Le reste des arguments

# Vérifier que le script existe
if [ -z "$SCRIPT_PATH" ]; then
    echo "Usage: ./run_with_isaaclab.sh <script.py> [args...]"
    echo ""
    echo "Exemples:"
    echo "  ./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Erreur: Script non trouvé: $SCRIPT_PATH"
    exit 1
fi

# Convertir en chemin absolu
SCRIPT_PATH=$(readlink -f "$SCRIPT_PATH")

echo "========================================="
echo "Lancement avec IsaacLab"
echo "========================================="
echo "Script: $SCRIPT_PATH"
echo "Args: $SCRIPT_ARGS"
echo ""

# Changer de répertoire et lancer
cd "$ISAACLAB_DIR"

# Méthode 1: Essayer avec TERM=xterm
export TERM=xterm
./isaaclab.sh -p "$SCRIPT_PATH" $SCRIPT_ARGS 2>&1

# Si ça échoue, essayer sans TERM
if [ $? -ne 0 ]; then
    echo ""
    echo "Première tentative échouée, essai sans TERM..."
    unset TERM
    exec ./isaaclab.sh -p "$SCRIPT_PATH" $SCRIPT_ARGS
fi
