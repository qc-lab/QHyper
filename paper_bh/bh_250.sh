#!/bin/bash
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -A plgkwantowy3-cpu
# the above lines are read and interpreted by sbatch, this one and leter are not
# command, this will be executed on a compute node

module load python/3.10.4-gcccore-11.3.0

python3 test_basinhopping.py -w 2 --values 2 2 1 --weights 1 1 1 --samples 40 --processes 10 --niter 250
