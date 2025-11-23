#!/bin/bash
# Quick test script for AWD training

echo "================================"
echo "Testing AWD Training Setup"
echo "================================"

# Test with minimal configuration
python awd_isaaclab/scripts/train_awd.py \
    --task DucklingCommand \
    --robot go_bdx \
    --num_envs 4 \
    --max_iterations 2 \
    --headless \
    "$@"
