#!/bin/bash
# Wrapper pour lancer IsaacLab avec Isaac Sim configuré

# Chemin vers Isaac Sim
export ISAACSIM_PATH="/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
export ISAAC_PATH="$ISAACSIM_PATH"

# Chemin vers IsaacLab
ISAACLAB_DIR="/home/alexandre/Developpements/IsaacLab"

# Script à exécuter
SCRIPT_PATH="$1"
shift
SCRIPT_ARGS="$@"

if [ -z "$SCRIPT_PATH" ]; then
    echo "Usage: ./run_with_isaac_configured.sh <script.py> [args...]"
    exit 1
fi

# Convertir en chemin absolu
SCRIPT_PATH=$(readlink -f "$SCRIPT_PATH")

echo "========================================="
echo "Lancement avec IsaacLab + Isaac Sim"
echo "========================================="
echo "Isaac Sim: $ISAACSIM_PATH"
echo "IsaacLab: $ISAACLAB_DIR"
echo "Script: $SCRIPT_PATH"
echo "Args: $SCRIPT_ARGS"
echo ""

# Aller dans IsaacLab
cd "$ISAACLAB_DIR"

# Lancer avec les bonnes variables
TERM=xterm ./isaaclab.sh -p "$SCRIPT_PATH" $SCRIPT_ARGS
