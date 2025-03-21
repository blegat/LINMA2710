#!/bin/bash
#SBATCH --job-name=pluto_gpu1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

ip=$(hostname -i)
echo "On your local computer (not the CECI cluster!), run \$ sshuttle -r $CLUSTER_NAME $ip/16"
echo "Checking node CPU info:"
julia --project -e 'using InteractiveUtils; versioninfo()'
echo "Install Julia packages and compile them:"
julia --project -e 'import Pkg; Pkg.instantiate()'
echo "Checking node GPU info:"
julia --project -e 'using CUDA; CUDA.versioninfo()'
echo "Launching Pluto:"
srun julia --project -e "import Pluto; Pluto.run(host=\"$ip\", port=1234, launch_browser=false)"
