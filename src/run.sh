#!/usr/bin/env bash
# Runs a pipeline command inside the qml env. Usage: bash run.sh <python args...>
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
export PYTHONNOUSERSITE=1
cd "/mnt/c/Research work 2/quantum-ids/src"
python "$@"
