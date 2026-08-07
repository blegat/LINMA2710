#!/bin/bash
#SBATCH --job-name=tracy
#SBATCH --ntasks=1
#SBATCH --time=1:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

ip=$(hostname -i)
echo "On your local computer (not the CECI cluster!), run \$ sshuttle -r $CLUSTER_NAME $ip/16"
srun --unbuffered ./OpenCLVectorAdd
