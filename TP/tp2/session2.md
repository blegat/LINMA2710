# LINMA2710 - Scientific Computing  
## TP2 - Distributed Computing 
**Professors:** P.-A. Absil, B. Legat  
**Teaching Assistants:** A. Oswald, A. Bayat, B. Loucheur  
**Year:** 2025-2026

---

## MPI
Consider a scalar discrete dynamical system defined by the map $`f(x) = x^2`$. We apply this dynamics element-wise to a vector of states (but we keep the notation $`f`$ by abuse of notation).

Let $`x_0 = \{x_0^{(i)}\}_{i=1}^N  \in R^N`$ represent a vector of initial conditions. We compute the system's evolution for 50 steps:

$$x_{k+1} = f(x_k), \quad k=0, \dots, 49
$$

where the map $`f`$ is applied element-wise (by abuse of notation). The final result is stored in $`y = x_{50}`$. Specifically, for $`y \in R^N`$ let:

$$y^{(i)} = x_{50}^{(i)}, \quad i=1,\dots,N,
$$

where each $`x_{50}^{(i)}`$ is obtained by propagating the dynamics through the map $`f`$ as defined above.

##### Files concerned: `distributed.cpp` and `example.cpp`.

##### Exercise 1

We now want to use MPI to compute efficiently $`y`$. Draw the situation by hand and think of a natural way to distribute the computation accross nodes.
 
##### Exercise 2

Implement missing parts of the file `distributed.cpp`. A minimal example of MPI initialization is provided in `example.cpp`. 

`Hint` Check the following functions :
* `MPI_Gatherv(...)` and `MPI_Gather(...)`,
* `MPI_Scatterv(...)` and `MPI_Scatter(...)`,
* `MPI_Reduce(...)` and `MPI_Allreduce(...)`,
* `MPI_Broadcast(...)`.

Run your implementation on your personal computer. Try to increase the number of processes. Go as high as you can. What do you observe?

`Hint` To compile your code with MPI, use

```
    mpicxx -[O1 or O2 or O3] distributed.cpp -o distributed
```
To run the resulting script, use
```
    mpirun -np [NUM_OF_PROCESSES] ./distributed
```
##### Exercise 3

Copy and run your code on the cluster of your choice. 

`Hint` Look at the course README for detailed information about clusters access and job submissions.

##### Exercise 4

When introducing MPI, you add communication costs between processes. These costs are not negligible and should therefore be quantified. The following model is usually used to describe such costs:

$$T = \alpha + \beta m,
$$

where we denote
* $`\alpha`$ is the latency (i.e. fixed cost) in seconds,
* $`\beta`$ is the inverse bandwidth (i.e. the time to send one byte) in seconds per byte,
* $`m`$ is the message size in bytes,
* $`T`$ is the communication time in seconds.

In the problem above, assume that each processes have the same chunck size. In such situation, at some point, you would've used the collective `MPI_Gather(...)`. Compute the lower complexity bound of this operation.

##### Exercise 5

Inside each MPI process, can you still speed up the computation? Think about parallelism and vectorization. Can you enable both on your computer? And on a cluster? As in the previous session, how would you connect the three techniques: distributed, parallel, and vectorized computing?

Implement parallelism and vectorization, and run it on the cluster. Compute the speedup for each configuration, and compare it to the naive sequential implementation, for $`N \in \left\{ 10^{5}, \, 10^{6}, \, 10^{7}\right\}`$.



