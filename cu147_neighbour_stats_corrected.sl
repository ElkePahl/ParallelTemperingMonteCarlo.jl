#!/bin/bash
#SBATCH --job-name=cu147-neigh-stats
#SBATCH --account=uoa02731
#SBATCH --time=16-00:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=28
#SBATCH --output=cu147-neigh-stats-%j.out
#SBATCH --error=cu147-neigh-stats-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load Julia/1.11.3-GCC-12.3.0-VTune
module load OpenBLAS/0.3.23-GCC-12.3.0

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
julia --version

export LD_LIBRARY_PATH="$HOME/lib/runner_compat:$LD_LIBRARY_PATH"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Julia threads requested: ${SLURM_CPUS_PER_TASK}"

ldd MachineLearningPotential/lib/librunnerjulia.so

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. runs/cu147_neighbour_stats_corrected/cu147_neighbour_stats_corrected_run.jl
