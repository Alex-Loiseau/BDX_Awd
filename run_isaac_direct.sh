#!/bin/bash
# Wrapper pour lancer directement avec le Python d'Isaac Sim
# Évite les conflits NumPy

ISAAC_SIM="/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
SCRIPT="$1"
shift
ARGS="$@"

if [ -z "$SCRIPT" ]; then
    echo "Usage: ./run_isaac_direct.sh <script.py> [args...]"
    echo ""
    echo "Exemple:"
    echo "  ./run_isaac_direct.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --robot mini_bdx --test"
    exit 1
fi

# Convertir en chemin absolu
SCRIPT=$(readlink -f "$SCRIPT")

echo "========================================="
echo "Lancement avec Isaac Sim Python"
echo "========================================="
echo "Isaac Sim: $ISAAC_SIM"
echo "Script: $SCRIPT"
echo "Args: $ARGS"
echo ""

# Lancer avec le Python d'Isaac Sim
"$ISAAC_SIM/python.sh" "$SCRIPT" $ARGS
