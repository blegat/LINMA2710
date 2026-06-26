# OpenCL

Examples inspired from [these exercises](https://github.com/HandsOnOpenCL/Exercises-Solutions).

# Installation

To run the `Makefile`s of this directory, you need to install OpenCL headers:
```sh
$ sudo pacman -S opencl-headers # ArchLinux
```

To use the GPU as OpenCL device from Julia, simply install `CUDA.jl`, no need for `module load CUDA` on the CECI cluster.

To use an intel GPU as OpenCL device, install the Intel compute runtime
```sh
$ sudo pacman -S intel-compute-runtime # ArchLinux
```

To your CPU as OpenCL device, install POCL
```sh
$ sudo pacman -S pocl # ArchLinux
```
On Julia, simply run `using pocl_jll` before running any OpenCL function.

> [!WARNING]
> Avoid using the platforms provided by `pocl` installed
> by your system package manager on Julia with a `local_size` larger than 1.
> See [here](https://github.com/JuliaGPU/OpenCL.jl/issues/290) for more details.
