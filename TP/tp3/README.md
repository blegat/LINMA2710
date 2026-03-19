# LINMA2710 - Scientific Computing
## TP3 - GPU Computing with OpenCL

| Professors | Teaching Assistants | Year |
|------------|---------------------|------|
| P.-A. Absil, B. Legat | A. Oswald, A. Bayat, B. Loucheur | 2025-2026 |

---

## Introduction

In the previous sessions, we explored SIMD, OpenMP, and MPI to speed up the same computation: applying $`f(x) = x^2`$ element-wise. Each technique exploits a different level of parallelism. Today, we introduce yet another one: **GPU computing with OpenCL**.

### What is OpenCL?

OpenCL (Open Computing Language) is a framework for writing programs that execute on heterogeneous platforms (CPUs, GPUs, FPGAs, ...). The key idea is:

- The **host** (your CPU) sets up data, compiles GPU programs, and launches them.
- The **device** (typically a GPU) executes small functions called **kernels** in parallel across many **work-items**.

A kernel is a function that runs once per work-item. Each work-item gets a unique index via `get_global_id(0)`, which it uses to know which piece of data to process.

### Memory model (simplified)

```
  Host (CPU)                    Device (GPU)
  ──────────                    ──────────────
  std::vector<float> x    ──►  cl::Buffer buf_x     (host-to-device transfer)
                                    │
                                    ▼
                               kernel executes
                                    │
                                    ▼
  std::vector<float> y    ◄──  cl::Buffer buf_y     (device-to-host transfer)
```

Data must be explicitly transferred between host and device memory. This has a cost!

### Typical OpenCL program structure

1. Query platforms and devices
2. Create a **context** and a **command queue**
3. Compile kernel source code into a **program**, extract the **kernel**
4. Create **buffers** on the device, write input data from host to device
5. Set kernel arguments and **enqueue** the kernel
6. Read results back from device to host

---

## Setup: running on the cluster

Most laptops don't have usable OpenCL support (Apple deprecated it, many systems lack GPU drivers). We will use the **CECI cluster** (manneback or Lyra).

#### Step 1: Connect to the cluster

```bash
(your computer) $ ssh manneback
```

#### Step 2: Get an interactive session with a GPU
Manneback:
```bash
salloc --partition=gpu --gres=gpu:1 --time=2:00:00 --mem-per-cpu=2000
```
Lyra:
```bash
salloc --partition=batch --gres=gpu:1 --time=2:00:00 --mem-per-cpu=2000
```


Wait for the allocation, then you should be on a compute node (e.g., `mb-gpu001` or `ly-w218`).

#### Step 3: Load modules

```bash
module load CUDA
```

If that doesn't work, try `module avail` to find the right module name, or ask a TA.

#### Step 4: Copy the TP files

Use `scp`, `git`, or `sshfs` as described in the course README.

#### Compiling

All files in this session are compiled with:
```bash
g++ -std=c++17 -O2 -I include -lOpenCL -o <file> <file>.cpp
```

We reuse the OpenCL C++ header from the project (`project/include/CL/opencl.hpp`).

---

## Part 1: Discover the GPU

##### File concerned: `query_device.cpp`

This file is **already complete**. Compile and run it:

```bash
g++ -std=c++17 -O2 -I include -lOpenCL -o query_device query_device.cpp
./query_device
```

**Questions:**
1. What is the name of the GPU on the cluster node?
2. How many compute units does it have?
3. What is the maximum work-group size?
4. How much global memory is available?

---

## Part 2: Vector Squaring on the GPU

##### File concerned: `gpu_square.cpp`

We return to our familiar dynamical system. Given $`x_0 \in \mathbb{R}^N`$, we compute:

$$y^{(i)} = \left(x_0^{(i)}\right)^2, \quad i = 1,\dots,N.$$

The host-side boilerplate (context, queue, buffers) is provided. You need to:

#### Exercise 1

Complete the OpenCL kernel `square_kernel` (marked `TODO 1`). This kernel should:
- Get the global index of the current work-item using `get_global_id(0)`
- Check that the index is within bounds (`< N`)
- Compute `y[i] = x[i] * x[i]`

#### Exercise 2

Complete the host-side code to launch the kernel (marked `TODO 2`):
- Set the kernel arguments using `kernel.setArg(...)`
- Enqueue the kernel using `queue.enqueueNDRangeKernel(...)`

> [!TiP]
> look at how the `fill` kernel is set up just above, it follows the same pattern.

#### Exercise 3

Compile, run, and verify the output:
```bash
g++ -std=c++17 -O2 -I include gpu_square.cpp -lOpenCL -o gpu_square
./gpu_square
```

The program compares GPU results against CPU results and reports timing.

---

## Part 3: Iterated Dynamics on the GPU

We now apply $`f`$ for 50 steps, as in TP1 and TP2:

$$y^{(i)} = x_{50}^{(i)}, \quad i=1,\dots,N.$$

#### Exercise 4

Complete the kernel `square_iter_kernel` (marked `TODO 3`). This kernel should:
- Load `x[i]` into a local variable `v`
- Apply `v = v * v` in a loop for `n_iter` iterations
- Store the result in `y[i]`

#### Exercise 5

Complete the host-side launch code (marked `TODO 4`), following the same pattern as Exercise 2.

#### Exercise 6

Run and observe:
1. Compare the GPU time vs the CPU time. For which values of $`N`$ does the GPU win?
2. The program also measures the **transfer time** (host-to-device + device-to-host). How does it compare to the computation time?
3. How does this compare to the speedups you observed with SIMD (TP1) and MPI (TP2)?

---

## Part 4: Kernel Launch Overhead

#### Exercise 7

Instead of looping inside the kernel (as in Exercise 4), implement the 50 iterations by **launching the kernel 50 times from the host** (marked `TODO 5`). For this approach, the kernel reads from and writes to the **same buffer** (in-place squaring).

#### Exercise 8

Compare the two approaches:
- **Approach A:** Single kernel launch with internal loop (Exercise 4)
- **Approach B:** 50 kernel launches from host (Exercise 7)

Which is faster? Why? What does this tell you about **kernel launch overhead**?

---

## Summary

After this session, you should have an intuition for:
- The **host/device** split and the cost of data transfers
- Writing simple **OpenCL kernels** with `get_global_id`
- The overhead of **kernel launches** vs doing more work per kernel
- How GPU computing compares to SIMD, OpenMP, and MPI on the same problem

---
