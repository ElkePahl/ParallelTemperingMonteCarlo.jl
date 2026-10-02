#!/bin/bash
#SBATCH --job-name=cu55-neigh-stats
#SBATCH --account=uoa02731
#SBATCH --time=10-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=28
#SBATCH --output=cu55_neigh_stats_%j.out
#SBATCH --error=cu55_neigh_stats_%j.err

set -euo pipefail

module purge
module load Julia/1.11.3-GCC-12.3.0-VTune
module load OpenBLAS/0.3.23-GCC-12.3.0

export LD_LIBRARY_PATH="$HOME/lib/runner_compat:${LD_LIBRARY_PATH:-}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

REPO="$HOME/ptmc-neighbour-matija-integration"
cd "$REPO"

julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. \
    runs/cu55_neighbour_stats_corrected/cu55_neighbour_stats_corrected_run.jl
