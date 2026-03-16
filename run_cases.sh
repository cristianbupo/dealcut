#!/bin/bash
set -e

BIN=./build/growth_iterative_lowner_cutfem
CFG=configs

cases=(
    "$CFG/growth_iterative_lowner_convex.json"         # cvx
    "$CFG/growth_iterative_lowner_concave.json"        # ccv
    "$CFG/growth_iterative_lowner_convex_hydg.json"    # cvx_hydg
    "$CFG/growth_iterative_lowner_concave_hydg.json"   # ccv_hydg
)

for cfg in "${cases[@]}"; do
    name=$(basename "$cfg" .json)
    echo "========================================"
    echo "  Running: $name"
    echo "========================================"
    $BIN "$cfg" 2>&1 | tee "output/${name}.log"
    echo ""
done

echo "All cases done."
