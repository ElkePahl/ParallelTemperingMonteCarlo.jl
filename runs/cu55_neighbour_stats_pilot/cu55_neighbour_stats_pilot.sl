#!/bin/bash
#SBATCH --job-name=cu55-stats-pilot
#SBATCH --account=uoa02731
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=28
#SBATCH --output=cu55_stats_pilot_%j.out
#SBATCH --error=cu55_stats_pilot_%j.err

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
    runs/cu55_neighbour_stats_pilot/cu55_neighbour_stats_pilot_run.jl
