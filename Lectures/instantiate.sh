#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

srun julia --project -e 'import Pkg; Pkg.instantiate()'
