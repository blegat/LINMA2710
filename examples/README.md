# Profiling with `tau2`

We detail the use of `tau2` for profiling OpenCL. See also [these slides](https://indico.ijs.si/event/1183/sessions/171/attachments/1065/1362/EuroCC_Intro_to_parallel_programming_accelerators_pt-2.pdf).

## Installing `tau2`

> [!IMPORTANT]
> `tau2` is available on the `manneback` cluster in the `releases/2023b`.
> As this is the default release, you should be able to just do `module load tau2` to use `tau_exec` and `pprof`
> You still need to install it on your own computer if you want to use the graphical interfaces `paraprof` and `jumpshot`.

### Installing from source

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

## Profiling OpenCL with `tau2`

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

## Tracing OpenCL with `tau2`

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
