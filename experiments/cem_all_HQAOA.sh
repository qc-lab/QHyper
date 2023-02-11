#!/bin/bash
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -A plgkwantowy2-cpu
#SBATCH -o results_4/cem_all_HQAOA.txt
# the above lines are read and interpreted by sbatch, this one and leter are not
# command, this will be executed on a compute node

source ../venv/bin/active
module load python/3.10.4-gcccore-11.3.0

python3 cem_all_HQAOA.py
