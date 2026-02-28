# LINMA2710: Project 2026

This project explores matrix operations implemented across four computing paradigms: sequential C++, shared-memory parallelism (OpenMP), distributed computing (MPI), and GPU computing (OpenCL). The project is divided into four parts.

The same core matrix operations are implemented three times, each time targeting a different computing paradigm:

| Class | Where data lives | How it computes | Parts |
|---|---|---|---|
| `Matrix` | In a `std::vector<double>` on the CPU | Sequential loops (Part 1), then parallelized with OpenMP threads (Part 2) | 1 & 2 |
| `DistributedMatrix` | Split by columns across MPI processes, each holding a local `Matrix` | Each process operates on its local columns; processes communicate via MPI when needed | 3 |
| `MatrixCL` | In an OpenCL buffer on the GPU | OpenCL kernels launched on the GPU | 4 |

## Part 1 — Basic Matrix Operations

In this first part, you implement a `Matrix` class with fundamental operations in C++.

### Requirements

This part must be implemented using only the C++ Standard Library. No external libraries or dependencies are allowed.

### Tasks

The `Matrix` class should implement the member functions as defined in `include/matrix.hpp` (and tested in `tests/test_matrix.cpp`).

The implementation file is `template/matrix.cpp`.

### Questions

1. Assume I use the copy constructor `Matrix(const Matrix& other)` to copy a matrix. Then, I modify an element of the copied matrix. What happens to the original matrix?
2. How would you handle special cases like sparse matrices?
3. Explain why the `Matrix` class does not need an explicitly defined destructor `~Matrix()`.
4. Can you speed up matrix operations using SIMD instructions? Measure the speedup compared to the non-SIMD version.

### Leaderboard

The fastest 3 implementations for this part will obtain respectively 0.5, 1 and 1.5 bonus points (over 20) for the course.
The benchmarking will be run automatically on Inginious for any submission that passes all the tests.

> Here is the output of `lscpu` on an Inginious runner:
> ```bash
> $ lscpu
> Architecture:        x86_64
> CPU op-mode(s):      32-bit, 64-bit
> Byte Order:          Little Endian
> CPU(s):              16
> On-line CPU(s) list: 0-15
> Thread(s) per core:  2
> Core(s) per socket:  8
> Socket(s):           1
> NUMA node(s):        1
> Vendor ID:           GenuineIntel
> CPU family:          6
> Model:               165
> Model name:          Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
> Stepping:            5
> CPU MHz:             2900.000
> CPU max MHz:         4800.0000
> CPU min MHz:         800.0000
> BogoMIPS:            5799.77
> Virtualization:      VT-x
> L1d cache:           32K
> L1i cache:           32K
> L2 cache:            256K
> L3 cache:            16384K
> NUMA node0 CPU(s):   0-15
> Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp pku ospke md_clear flush_l1d arch_capabilities
> ```

## Part 2 — Parallel Matrix Operations (OpenMP)

In this part, you parallelize the Matrix operations from Part 1 using OpenMP. You work in the **same file** `template/matrix.cpp`.

### Requirements

This part requires OpenMP in addition to the C++ Standard Library. Compile with the `-fopenmp` flag.

### Tasks

- Add `#pragma omp parallel for` directives to parallelize the matrix multiplication and other suitable operations.
- Implement a benchmark that measures the execution time of matrix multiplication for different matrix sizes with varying numbers of threads.

### Questions

1. What is the speedup you observe when using OpenMP for matrix multiplication? How does it vary with matrix size and number of threads?
2. Explain Amdahl's law and how it applies to the parallelization of matrix multiplication.
3. For small matrices, OpenMP parallelization may actually slow things down. Explain why and discuss potential solutions.

## Part 3 — Distributed Matrix Operations (MPI)

In this part, you implement a `DistributedMatrix` class that distributes a matrix by columns across MPI processes. It provides the same common operations as `Matrix`, plus distributed-specific operations like `multiplyTransposed` and `gather`.

### Requirements

This part requires MPI in addition to the C++ Standard Library. Compile with `mpic++`. Your code needs to be tested on the CECI cluster.

### Tasks

The `DistributedMatrix` class should implement the member functions as defined in `include/distributed_matrix.hpp` (and tested in `tests/test_distributed.cpp`).

The matrix is split by columns into parts as equal as possible across processes. Both matrices involved in operations have the same column partitioning across processes.

The implementation file is `template/distributed_matrix.cpp` (look for `TODO` markers).

### Questions

1. Profile and analyze the communication overhead (MPI operations) versus actual computation time in `DistributedMatrix::multiplyTransposed`.
2. What is the expected speedup for the distributed `DistributedMatrix::multiplyTransposed` operation? Compare this with the speedup you measure in your numerical experiments.
3. Compare this distributed approach (splitting columns) with an alternative where data is partitioned among processes and gradients are synchronized afterward.

## Part 4 — GPU Matrix Operations (OpenCL)

In this part, you implement a `MatrixCL` class that performs the same matrix operations as `Matrix`, but on the GPU using OpenCL kernels.

### Requirements

This part requires OpenCL in addition to the C++ Standard Library. Compile with `-lOpenCL`.

### Tasks

The `MatrixCL` class should implement the methods as defined in `include/matrix_opencl.hpp` (and tested in `tests/test_opencl.cpp`).

A `MatrixCL` object stores:
- `rows_` and `cols_` for the number of rows/columns
- `context_` a reference to an OpenCL context for managing device memory
- `queue_` a reference to an OpenCL CommandQueue for launching kernels
- `buffer_` an OpenCL buffer storing matrix elements on the device

All operations are performed directly on device memory. OpenCL kernel code is compiled once at initialization with `initializeKernels` and stored in a shared `KernelCache`.

The implementation file is `template/matrix_opencl.cpp` (look for `TODO` markers). You need to write both the OpenCL kernel source strings and the host-side methods that invoke them.

### Questions

1. Implement 2 versions of the matrix-matrix multiplication: a simple and a faster one. Be ready to show your two OpenCL kernel codes and explain them briefly.
2. Profile and analyze your two implementations on a GPU. It may also be useful to query the [profiling info](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html) (don't forget to enable profiling in the queues with `CL_QUEUE_PROFILING_ENABLE`).
3. Measure the impact of the kernel implementation on the power consumption of the GPU. To measure the power consumption, different tools are available, [codecarbon](https://github.com/mlco2/codecarbon) is an example.

## Building and Running

The `Makefile` supports all parts. By default it compiles the solution files from `src/`. To compile the student template instead, use `SRC_DIR=template`:

```bash
# Compile and run Part 1 & 2 tests (solution)
make run_matrix

# Compile and run Part 1 & 2 tests (student template)
make run_matrix SRC_DIR=template

# Compile and run Part 3 tests
make run_distributed

# Compile and run Part 4 tests
make run_opencl

# Clean all binaries
make clean
```

## Deadline

Monday, May 4th 2026 at 12:00.

## Submission

You submit your implementation files on Inginious. The parts are:

| Part | File to submit |
|---|---|
| Part 1 — Basic + SIMD | `matrix.cpp` (run with 1 thread) |
| Part 2 — OpenMP | `matrix.cpp` (run with multiple threads) |
| Part 3 — MPI | `distributed_matrix.cpp` |
| Part 4 — OpenCL | `matrix_opencl.cpp` |

Parts 1 and 2 share the same source file `template/matrix.cpp`, compiled with `-fopenmp`. On Inginious, they are separate tasks: Part 1 runs your code with a single thread, Part 2 runs it with multiple threads.

## Guidelines

 - **Collaboration and AI**: You are allowed to discuss and exchange ideas with other people or AIs but you are not allowed to share code with other people and do not copy-paste code from AI output or use coding agents. In the grading, we will put a strong emphasis on your complete and precise understanding of the code, benchmarks and profiles you have produced. This grading favors students that write their own code and comment its design decisions. On the other hand, copy-pasting code from internet, from your colleagues or from AI output will usually not improve your grade as we might notice at the oral examination that your understanding is not on par with your code. So for this project (and this is also a good advice for any project even outside of grading considerations), prefer doing less but better and with a deeper understanding.
 - **Code Submission**: On Inginious, submit your implementation files. You are allowed to make as many submissions as you need, only the last submission will be taken into account. You are advised to verify that your submission passes the tests in Inginious early before the deadline. Note that, even if submitting the code on Inginious is mandatory, the Inginious automatic grading has no influence on the final grading. The tests on Inginious are similar to those included in the `tests/` directory. Since these tests are minimalist, passing them is a necessary but not sufficient condition for having correct code.
 - **Oral Evaluation**: There will be an oral evaluation at the end of the year on the project. The questions listed under each part are provided as food for thought to deepen your understanding; the actual oral questions may be different.
 - **Language**: English is the default language. However, since the course is French-friendly, French is accepted without penalty.
 - **Questions**: If you have any questions, please contact the TA: `benoit.loucheur@uclouvain.be`, `antonin.oswald@uclouvain.be` and `amir.bayat@uclouvain.be`.
