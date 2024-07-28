#!/bin/bash -e
#SBATCH --job-name=sam_test    
#SBATCH --account=uoa02731    
#SBATCH --time=2:00:00         
#SBATCH --mem=1200      
#SBATCH --ntasks=1             
#SBATCH --nodes=1             
#SBATCH --cpus-per-task=2       
       
   
#SBATCH --error=error-%j.err

module load OpenBLAS
module load ScaLAPACK
module load Julia


   
srun julia -t 2 pbc32.jl > output-${SLURM_JOB_ID}.out

