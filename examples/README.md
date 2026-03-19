# Profiling

We explain in this section how to profile your code with `codecarbon`, `tau2` or `tracy`.

## Profiling with `codecarbon`

The [`codecarbon` tool](https://github.com/mlco2/codecarbon) allows to track the power consumption of a process.
It is designed to track the consumption of python functions but it can also be used to
track the energy consumption of arbitrary executables (such as built from C/C++) using [monitor-emissions-c](https://github.com/adrienbanse/monitor-emissions-c):
Here is short example showing how to do it.
First connect to the cluster and clone [`monitor-emissions-c`](https://github.com/adrienbanse/monitor-emissions-c):
```sh
[local computer]$ ssh manneback
[blegat@mbackf2 ~]$ git clone https://github.com/adrienbanse/monitor-emissions-c.git
```
Now, let's install its dependencies, we show here how to do it with [`uv`](https://docs.astral.sh/uv/) so start by installing it if it's not already done:
```sh
[blegat@mbackf2 ~]$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
We now create a virtual environment and install `codecarbon` as it is in the `requirements.txt` file
```sh
[blegat@mbackf2 ~]$ cd monitor-emissions-c
[blegat@mbackf2 monitor-emissions-c]$ uv venv
[blegat@mbackf2 monitor-emissions-c]$ uv pip install -r requirements.txt --python .venv
```
Now, let's load `CUDA` and then run the [`vadd_chain` example](OpenCL/vadd_chain) on the GPU (assuming you cloned [LINMA2710](https://github.com/blegat/LINMA2710) on the parent folder and that you already compiled this example)
```sh
[blegat@mbackf2 monitor-emissions-c]$ module load CUDA
[blegat@mbackf2 monitor-emissions-c]$ srun --partition=gpu --gres=gpu:1 ./.venv/bin/python monitor.py ../LINMA2710/examples/OpenCL/vadd_chain/vadd_chain
srun: job 57726961 queued and waiting for resources
srun: job 57726961 has been allocated resources
[codecarbon WARNING @ 10:39:29] Multiple instances of codecarbon are allowed to run at the same time.
[codecarbon INFO @ 10:39:29] [setup] RAM Tracking...
[codecarbon INFO @ 10:39:29] [setup] CPU Tracking...
[codecarbon INFO @ 10:39:29] Tracking Intel CPU via RAPL interface
[codecarbon INFO @ 10:39:30] [setup] GPU Tracking...
[codecarbon INFO @ 10:39:30] Tracking Nvidia GPU via pynvml
[codecarbon INFO @ 10:39:30] The below tracking methods have been set up:
                RAM Tracking Method: RAM power estimation model
                CPU Tracking Method: RAPL
                GPU Tracking Method: pynvml

[codecarbon INFO @ 10:39:30] >>> Tracker's metadata:
[codecarbon INFO @ 10:39:30]   Platform system: Linux-5.4.286-1.el8.elrepo.x86_64-x86_64-with-glibc2.28
[codecarbon INFO @ 10:39:30]   Python version: 3.12.8
[codecarbon INFO @ 10:39:30]   CodeCarbon version: 3.0.1
[codecarbon INFO @ 10:39:30]   Available RAM : 2.000 GB
[codecarbon INFO @ 10:39:30]   CPU count: 2 thread(s) in 2 physical CPU(s)
[codecarbon INFO @ 10:39:30]   CPU model: Intel(R) Xeon(R) Gold 6346 CPU @ 3.10GHz
[codecarbon INFO @ 10:39:30]   GPU count: 1
[codecarbon INFO @ 10:39:30]   GPU model: 1 x NVIDIA A10 BUT only tracking these GPU ids : [0]
[codecarbon INFO @ 10:39:33] Emissions data (if any) will be saved to file /auto/home/users/b/l/blegat/monitor-emissions-c/emissions.csv
Invalid file vadd.cl
1 platforms found

 Device is  NVIDIA A10  GPU from  NVIDIA Corporation  with a max of 72 compute units
[codecarbon INFO @ 10:39:33] Energy consumed for RAM : 0.000001 kWh. RAM Power : 10.0 W
[codecarbon INFO @ 10:39:33] Delta energy consumed for CPU with intel_rapl : 0.000021 kWh, power : 238.74691655808266 W
[codecarbon INFO @ 10:39:33] Energy consumed for All CPU : 0.000021 kWh
[codecarbon INFO @ 10:39:33] Energy consumed for all GPUs : 0.000003 kWh. Total GPU Power : 34.75551818161019 W
[codecarbon INFO @ 10:39:33] 0.000025 kWh of electricity used since the beginning.
```
The result is stored in the `emissions.csv`. This file is automatically created if it did not exists. Otherwise, the new measurement is added as a new row.
Now, copy this file back to your local computer with `scp` or `sshfs` as detailed [here](..).

## Profiling with Tracy profiler

An example for profiling OpenCL with [Tracy](https://github.com/wolfpld/tracy) is available [here](https://github.com/wolfpld/tracy/tree/master/examples/OpenCLVectorAdd).
We also added [a slightly adapted version in here](OpenCL/tracy) where you also have instructions on how to run in on the cluster.

## Profiling with `tau2`

We detail the use of `tau2` for profiling OpenCL. See also [these slides](https://indico.ijs.si/event/1183/sessions/171/attachments/1065/1362/EuroCC_Intro_to_parallel_programming_accelerators_pt-2.pdf).

### Installing `tau2`

> [!IMPORTANT]
> `tau2` is available on the `manneback` cluster in the `releases/2023b`.
> As this is the default release, you should be able to just do `module load tau2` to use `tau_exec` and `pprof`
> In order to use the graphical interfaces `paraprof` and `jumpshot`, you can either
> 1. Use X11 forwarding on the cluster (either using `ForwardX11 yes` in your `.ssh/config` file or using `ssh -X manneback`), you will also need `module load Java` and then you can run `paraprof` or `jumpshot` on the cluster.
> 2. You can install `tau2` on your own computer, copy the profiling file to your computer and run `paraprof` or `jumpshot` on your computer.

#### Installing from source

> [!WARNING]
> Make sure to [deactivate any conda environment](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html#deactivating-an-environment) before compiling `tau2`.
> Users reported issues of installing `tau2` within a conda environment.

The software is not available on the clusters so you cannot just load them with `module load`, you will need to install it from source.
The following will install the binaries `tau_exec`, `pprof`, `paraprof` and `jumpshot` that we will need in the next sections.
First clone it:
```sh
$ git clone https://github.com/UO-OACISS/tau2.git
```
To configure, it is important to read the [GPU-specific instructions](https://github.com/UO-OACISS/tau2/blob/master/README.gpu) in order for it to catch OpenCL-specific commands.
You need to pass a folder to `-lopencl` where [it should find `include/CL/cl.h` and `lib/libOpenCL.so`](https://github.com/UO-OACISS/tau2/blob/master/configure#L3360-L3403).
In my local Linux computer, I can simply do
```sh
[local computer]$ ./configure -opencl=/usr
[blegat@mbackf2 ~]$ git clone https://github.com/adrienbanse/monitor-emissions-c.git
```
On the manneback cluster, we will use the OpenCL library that comes with CUDA. To see where it is located, we can see `Lmod`:
```sh
[blegat@mb-icg102 usr]$ module show CUDA
...
prepend_path("CMAKE_PREFIX_PATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1")
prepend_path("CPATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/include")
prepend_path("CPATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/extras/CUPTI/include")
prepend_path("CPATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/nvvm/include")
prepend_path("LD_LIBRARY_PATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/lib")
prepend_path("LD_LIBRARY_PATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/extras/CUPTI/lib64")
prepend_path("LD_LIBRARY_PATH","/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/nvvm/lib64")
```
We see that it is under `/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1`. Let's see where OpenCL is:
```sh
[blegat@mbackf1 tau2]$ find /opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/ -name cl.h
/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux/include/CL/cl.h
[blegat@mbackf1 tau2]$ find /opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/ -name libOpenCL.so
/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux/lib/libOpenCL.so
```
Seems like `/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux` is a perfect fit so we can configure with:
```sh
[blegat@mbackf1 tau2]$ ./configure -opencl=/opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux/
...
Looking for libOpenCL.so in /opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux/
found /opt/sw/arch/easybuild/2023b/software/CUDA/12.1.1/targets/x86_64-linux/lib/libOpenCL.so
...
```
It worked! Let's use `make` now and then check where the binaries are stored:
```sh
[blegat@mbackf1 tau2]$ make
```
If we try `tau_exec` not it won't work because the binary is inside a directory that is not listed as one of the directories where the shell looks for binaries
```sh
[blegat@mbackf1 tau2]$ tau_exec
-bash: tau_exec: command not found
[blegat@mbackf1 tau2]$ find . -name tau_exec
./x86_64/bin/tau_exec
./tools/src/tau_exec
```
Let's just add it to this list (this assumes that you are at the root of the `tau2` directory where you did `./configure` and `make`, otherwise `pwd` won't give the correct value).
```sh
[blegat@mbackf1 tau2]$ PATH=$(pwd)/tools/src/tau_exec:$PATH
[blegat@mbackf1 tau2]$ tau_exec

Usage: tau_exec [options] [--] <exe> <exe options>
...
```
This `PATH` environment is only updated for this shell session. In order to apply everytime to lauch a new ssh connection, add it to you `.bashrc` with.
```sh
[blegat@mbackf1 tau2]$ echo "export PATH=$(pwd)/x86_64/bin:\$PATH" >> ~/.bashrc
```

### Profiling OpenCL with `tau2`

Given a binary `a.out` that is using OpenCL, it can be profiled as follows.
Consider for instance `vadd_chain`
```sh
[local computer]$ ssh manneback
[blegat@mbackf1 ~]$ cd LINMA2710/examples/OpenCL/vadd_chain
[blegat@mbackf1 vadd_chain]$ make
cc vadd_chain.c ../common/device_info.c -O3 -lm -DCL_TARGET_OPENCL_VERSION=300 -DDEVICE=CL_DEVICE_TYPE_DEFAULT -lOpenCL -fopenmp -I ../common -o vadd_chain
vadd_chain.c:23:10: fatal error: CL/cl.h: No such file or directory
 #include <CL/cl.h>
          ^~~~~~~~~
compilation terminated.
../common/device_info.c:21:10: fatal error: CL/cl.h: No such file or directory
 #include <CL/cl.h>
          ^~~~~~~~~
compilation terminated.
make: *** [Makefile:28: vadd_chain] Error 1
```
Remember from the previous section that OpenCL is provided by CUDA. So CUDA should be loaded:
```sh
[blegat@mbackf1 vadd_chain]$ module load CUDA
[blegat@mbackf1 vadd_chain]$ make
cc vadd_chain.c ../common/device_info.c -O3 -lm -DCL_TARGET_OPENCL_VERSION=300 -DDEVICE=CL_DEVICE_TYPE_DEFAULT -lOpenCL -fopenmp -I ../common -o vadd_chain
```
Now, to generate profiles, use `srun` to run it on a compute node with a GPU
```sh
[blegat@mbackf1 vadd_chain]$ srun --partition=gpu --gres=gpu:1 tau_exec -T serial -opencl ./vadd_chain
```
You should now have `profile.0.0.0` and a `profile.0.0.1` generated in the same directory.
If you only have `profile.0.0.0` and no `profile.0.0.1` file, it means that `tau2` was not compiled with OpenCL. Make sure you use the `-opencl` option in `./configure` above and that you saw in the output of `./configure` than OpenCL was found.
You can look at the profile with `pprof`:
```sh
[blegat@mbackf1 vadd_chain]$ srun --partition=gpu --gres=gpu:1 tau_exec -T serial -opencl ./vadd_chain
[blegat@mbackf1 vadd_chain]$ pprof
Reading Profile files in profile.*

NODE 0;CONTEXT 0;THREAD 0:
...
NODE 0;CONTEXT 0;THREAD 1:
...
```
You can also view this graphically using `paraprof`. For this, sync `profile.0.0.1` with your local computer (e.g., with `sshfs`) and then
run the following in the folder where the `profile...` files are
```sh
[blegat@mbackf1 vadd_chain]$ paraprof
```

### Tracing OpenCL with `tau2`

By setting the `TAU_TRACE` environment variable to `1`, `tau_exec` will collect traces instead of profiles as in the previous section.
```sh
[blegat@mbackf1 vadd_chain]$ srun --partition=gpu --gres=gpu:1 env TAU_TRACE=1 tau_exec -T serial -opencl ./vadd_chain
```
This will generate the files `events.0.edf`, `tautrace.0.0.0.trc` and `tautrace.0.0.1.trc`.
Now run
```sh
[blegat@mbackf1 vadd_chain]$ tau_treemerge.pl
/auto/home/users/b/l/blegat/tau2/x86_64/bin/tau_merge -m tau.edf -e events.0.edf events.0.edf tautrace.0.0.0.trc tautrace.0.0.1.trc tau.trc
tautrace.0.0.0.trc: 333 records read.
tautrace.0.0.1.trc: 23 records read.
```
This will create `tau.edf` and `tau.trc`. The following needs Java so you fist need to load Java
```sh
[blegat@mbackf1 vadd_chain]$ module load Java
[blegat@mbackf1 vadd_chain]$ tau2slog2 tau.trc tau.edf -o tau.slog2
```
This will create `tau.slog2`.
You can now visualize the traces graphically using `jumpshot`. For this, sync `tau.slog2` with your local computer (e.g., with `sshfs`) and then
run the following in the folder where the `tau.slog2` file is
```sh
[blegat@mbackf1 vadd_chain]$ jumpshot tau.slog2
```
