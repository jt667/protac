#!/bin/sh

 

#SBATCH --job-name=mega_array      # Job name
#SBATCH --array=1                  # Array range

 

#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=   # Where to send mail

 

#SBATCH --nodes=6                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --mem-per-cpu=8gb           # Memory per processor
#SBATCH --time=01:00:00             # Time limit hrs:min:sec

 

#SBATCH --output=array_%A-%a.out    # Standard output and error log

 

pwd; hostname; date

 

# Set the number of runs that each SLURM task should do
# PER_TASK=1000

 

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
# START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
# END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

 

# Print the job, task and run range
echo This is jobID $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID
echo Task ranges from $SLURM_ARRAY_TASK_MIN to $SLURM_ARRAY_TASK_MAX

 

# Activate conda environment
source /home/${USER}/.bashrc
source activate qc

 

# Run the loop of runs for this task.

for i in $(seq $SLURM_ARRAY_TASK_MIN $SLURM_ARRAY_TASK_MAX); do
  echo This is run number $i;
  python protac_classification_quantum.py $SLURM_ARRAY_JOB_ID $i > run_$i.log
done
