# Lectures

[<img src="https://plutojl.org/assets/favicon.svg" height="20"/>![](https://img.shields.io/badge/Notebooks-View-blue.svg)<img src="https://plutojl.org/assets/favicon.svg" height="20"/>](https://blegat.github.io/LINMA2710/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MWAwv3qeaX97nCNBc7adeukaK7vqc_KO?usp=sharing)

This folder contains the [Pluto](https://plutojl.org/) notebooks of the course. There are two options to visualize them.

## Option 1 : Static version

[<img src="https://raw.githubusercontent.com/fonsp/Pluto.jl/dd0ead4caa2d29a3a2cfa1196d31e3114782d363/frontend/img/logo_white_contour.svg" height="16"/> notebooks](https://blegat.github.io/LINMA2710/) ← Follow this link to access a static version of the notebooks that can be visualized in your web browser without the need to install [Julia](https://julialang.org/) nor [Pluto](https://plutojl.org/).
To visualize it as slides, open the javascript console of your web browser and enter `present()`. To leave this slide mode, enter `present()` again.
To enter full screen mode, use `F11` on your keyboard.

## Option 2 : Dynamic version

First, download the `.jl` file in this repository (e.g., by [downloading a zip](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github) or using [`git clone`](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).

Now, install [Julia](https://julialang.org/) (I recommend doing this using [juliaup](https://julialang.org/downloads/#juliaup_-_julia_version_manager)).
Once this is completed, [install and then run Pluto](https://plutojl.org/#install).
You will need to install the [LLVM OpenMP Runtime Library](https://openmp.llvm.org/) for the `-fopenmp` flag to work (or you will get `"-lomp": No such file or directory` when to use the checkbox enabling `-fopenmp`). Do for instance `# apt-get install libomp-dev` on Ubuntu or `# pacman -S openmp` on ArchLinux.

## Option 3 : Run it with Google Colab.

[This Google Colab notebook](https://colab.research.google.com/drive/1MWAwv3qeaX97nCNBc7adeukaK7vqc_KO?usp=sharing) shows how to do it.

## Option 4 : Run it from the CECI cluster

Run the following lines.
You can see below that the first time I run `cat slurm-....out`, I don't see anything because it hasn't started yet as you can see in the output of `squeue --me`.
You need then to way for the output of `cat slurm-....out` to display `Go to http://....`
The, run `sshuttle -r manneback 10.3.221.102/16` on your **local** computer, not on the CECI cluster.
Then open the url given after `Go to` using a web browser on your local computer and you should see the pluto menu. Select one of the lectures and it should be running on the cluster!
```sh
(your computer) $ ssh manneback
[blegat@mbackf1 ~] git clone https://github.com/blegat/LINMA2710.git
[blegat@mbackf1 ~] cd LINMA2710/Lectures
[blegat@mbackf1 ~] sh install_julia.sh
[blegat@mbackf1 ~] sbatch pluto_gpu1.sh
[blegat@mbackf1 Lectures]$ sbatch pluto_gpu1.sh
Submitted batch job 56863614
[blegat@mbackf1 Lectures]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          56863614       gpu pluto_gp   blegat PD       0:00      1 (Priority)
[blegat@mbackf1 Lectures]$ cat slurm-56863584.out
[blegat@mbackf1 Lectures]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          56863614       gpu pluto_gp   blegat  R       0:02      1 mb-icg102
[blegat@mbackf1 Lectures]$ cat slurm-56863584.out
On your local computer (not the CECI cluster!), run $ sshuttle -r manneback 10.3.221.102/16
...
1 device:
  0: NVIDIA A10 (sm_86, 21.972 GiB / 22.488 GiB available)
Launching Pluto:
[ Info: Loading...
┌ Info:
└ Go to http://10.3.221.102:1234/?secret=AV0z9DVY in your browser to start writing ~ have fun!
┌ Info:
│ Press Ctrl+C in this terminal to stop Pluto
└
```


## 🏆 Gain bonus points by fixing typos

If you find any typo or mistakes, feel free to open an issue or make a pull requests, it will be rewarded by bonus points on your final grade of the course! Note that [you can make a pull request without having to leave your web browser](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files).
