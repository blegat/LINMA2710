Example from [here](https://github.com/wolfpld/tracy/tree/master/examples/OpenCLVectorAdd) that was slightly modified so that it works outside the `cmake` infrastructure of `tracy`

First install the *Tracy profiler* (aka *server*) and launch it.

The Tracy profiler displays profiling info emitted from a *Tracy client*.
For the application you want to profile to serve as Tracy client, you need to embed the Tracy client
in your application.
The source code of the Tracy client is in the `public` subfolder of the `tracy` repository so let's clone it:
Run the following to clone `tracy` next to the `LINMA2710` folder:
```sh
$ git clone https://github.com/wolfpld/tracy.git
```
You can also clone it elsewhere but then you need to modify the `TRACY_DIR` variable in this `Makefile`.
Now, check the version you have on the Tracy server (the GUI you have launched), say it is `v0.11.1`.
You now need to checkout the corresponding tag in the `tracy` git repo to have the good version
of the Tracy client.
```sh
$ cd tracy
$ git checkout v0.11.1
```

Then, go back in this folder and do
```sh
$ make
```
Now, click on "Connect" on the tracy application. Then run:
```sh
$ ./OpenCLVectorAdd
```
You should now see the trace in the tracy application.

## Profiling a Tracy client running on the server

> [!NOTE]
> Tracy supports 3 GUI backends : [GLFW](https://www.glfw.org/), [Emscripten](https://emscripten.org/) and [Wayland](https://wayland.freedesktop.org/).
> Manneback does not have GLFW nor Emscripten nor [xkbcommon](https://xkbcommon.org/) which is needed by Wayland.
> For this reason, we will just install the Tracy server on our own computer and **not** run in on the cluser through `ssh -X`.

Clone `tracy` on the cluster and checkout the right tag like recommended above.

## With salloc

Now, connect to a compute node and take note of its ip
```sh
[blegat@mbackf2 tracy]$ salloc
salloc: Pending job allocation 57562683
salloc: job 57562683 queued and waiting for resources
salloc: job 57562683 has been allocated resources
salloc: Granted job allocation 57562683
salloc: Waiting for resource configuration
salloc: Nodes mb-sky015 are ready for job
[blegat@mb-sky015 tracy]$ hostname -i
10.33.204.15
```
Now, use the following to allow the server running on your local computer to communicate with the client running on the compute node of the cluster:
```sh
[local computer]$ sshuttle -r manneback 10.33.204.15/16 # /!\ modify it with "the output of `hostname -i` on the compute node"/16
```
Now, on the cluster, compile with
```sh
[blegat@mb-sky015 tracy]$ module load Clang
[blegat@mb-sky015 tracy]$ module load CUDA
[blegat@mb-sky015 tracy]$ make
```
In order to run, if you don't have a NVIDIA GPU, we can use the Intel OpenCL platform for running on the CPU
with `module load intel-compilers`
```sh
[blegat@mb-sky015 tracy]$ module load intel-compilers
[blegat@mb-sky015 tracy]$ ./OpenCLVectorAdd
Waiting 30 seconds to give you time to start the Tracy server...
28 seconds left...
28 seconds left...
```
Quick! Launch the Tracy server, enter the ip of the server and then click on "Connect" before the timer expires or you'll miss the profiling information of the start of the program (or the whole of it since this example is quite small).
You can also directly launch the profiler with the ip address (replace `tracy-profiler` by the path to the binary if it is not in your `PATH`):
```sh
tracy-profiler -a 10.33.204.15 # /!\ modify it with "the output of `hostname -i` on the compute node"
```
Now, the Tracy server will record the profiling info sent by the client which is then printing
```sh
3 seconds left...
2 seconds left...
1 seconds left...
0 seconds left...
... Done waiting, let's go
OpenCL Platform: Intel(R) OpenCL
OpenCL Device: Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz
VectorAdd Kernel Enqueued
VectorAdd Kernel Enqueued
...
VectorAdd Kernel Enqueued
VectorAdd Kernel Enqueued
VectorAdd Kernel 0 tooks 560us
VectorAdd Kernel 1 tooks 503us
...
VectorAdd Kernel 98 tooks 471us
VectorAdd Kernel 99 tooks 453us
VectorAdd runtime avg: 435.634us, std: 266.001us over 100 runs.
Results are correct!
```
The program has finished, terminating the Tracy client.
As the Tracy profiler has recorded all information, you can now visualize it and save it on your computer if you want to be able to access it later.

## With sbatch

Let's start by exiting the compute node and then let's unload `intel-compilers` to
run it on an NVIDIA GPU this time:
```sh
[blegat@mb-sky015 tracy]$ exit
salloc: Relinquishing job allocation 57562683
[blegat@mbackf2 tracy]$ module rm intel-compilers
```
Let's run the program on a compute node with `sbatch` now:
```sh
[blegat@mbackf2 tracy]$ sbatch submit.sh
Submitted batch job 57562725
[blegat@mbackf2 tracy]$ cat slurm-57562725.out
On your local computer (not the CECI cluster!), run $ sshuttle -r manneback 10.3.221.102/16
```
Quick, let's run `sshuttle` and launch the Tracy server.
The timer isn't displayed as its printing is buffered somewhere between the compute node and the login node but the clock is ticking!
```sh
[local computer]$ sshuttle -r manneback 10.3.221.102/16
[local computer]$ tracy-profiler -a 10.3.221.102
```
You can see the events appearing in the Tracy server. Once it's done, you can
check which GPU you got in the log on the cluster:
```sh
[blegat@mbackf2 tracy]$ cat slurm-57562725.out
On your local computer (not the CECI cluster!), run $ sshuttle -r manneback 10.3.221.102/16
Waiting 30 seconds to give you time to start the Tracy server...
29 seconds left...
28 seconds left...
...
1 seconds left...
0 seconds left...
... Done waiting, let's go
OpenCL Platform: NVIDIA CUDA
OpenCL Device: NVIDIA A10
VectorAdd Kernel Enqueued
VectorAdd Kernel Enqueued
...
VectorAdd Kernel Enqueued
VectorAdd Kernel Enqueued
VectorAdd Kernel 0 tooks 63us
VectorAdd Kernel 1 tooks 67us
...
VectorAdd Kernel 98 tooks 66us
VectorAdd Kernel 99 tooks 67us
VectorAdd runtime avg: 68.0006us, std: 1.65031us over 100 runs.
Results are correct!
```
