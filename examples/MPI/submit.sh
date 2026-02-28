#!/bin/bash
# Submission script for Lemaitre4
#SBATCH --time=01:00:00 # hh:mm:ss
#
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=1000 # megabytes

mpirun ./a.out
