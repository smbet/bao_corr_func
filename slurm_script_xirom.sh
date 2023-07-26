#!/bin/bash
#SBATCH --job-name=xiromtest
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=177gb
#SBATCH --mail-type=ALL
#SBATCH --account=PHS0336

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

qstat -f $SLURM_JOB_ID

source activate bao_simps
#
# Move to the directory where the job was submitted from
#
# Run Python
#
srun python errors_recon.py -v
