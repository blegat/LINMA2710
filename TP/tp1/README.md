# LINMA2710 - Scientific Computing  
## TP1 - Parallel Computing 
**Professors:** P.-A. Absil, B. Legat  
**Teaching Assistants:** A. Oswald, A. Bayat, B. Loucheur  


---

## Part 1: SIMD

Consider a scalar discrete dynamical system defined by the map $`f(x) = x^2`$. We apply this dynamics element-wise to a vector of states (but we keep the notation $`f`$ by abuse of notation).

Let $`x_0 = \{x_0^{(i)}\}_{i=1}^N  \in \mathbb{R}^N`$ represent a vector of initial conditions. We compute the system's evolution for 50 steps:

$$ x_{k+1} = f(x_k), \quad k=0, \dots, 49
$$

where the map $`f`$ is applied element-wise (by abuse of notation). The final result is stored in $`y = x_{50}`$.


#### Exercise 1
##### Files concerned

- `first_no_simd.cpp`,   
- `first_simd.cpp`,
- `first.hpp`, 
- `first.cpp`.

Consider a single step of the dynamics. Given a vector of initial conditions $`x_0 = \{x_0^{(i)}\}_{i=1}^N  \in \mathbb{R}^N`$, we compute the output vector $`y \in \mathbb{R}^N`$ defined by the component-wise squaring operation:

$$y^{(i)} = \left(x_0^{(i)}\right)^2, \quad i = 1,\dots,N.
$$


1. Start by implementing the missing part of the files `first_no_simd.cpp, first_simd.cpp`. Do you see what could be theoretically vectorized?
2. To check your intuition, execute the following commands on a terminal.
```bash
clang++ -O3 -march=native -S -emit-llvm first_no_simd.cpp -o first_no_simd.ll
clang++ -O3 -march=native -S -emit-llvm first_simd.cpp -o first_simd.ll
```
The resulting file is something similar to assembly (though a bit more high-level). What do you observe? Highlight the differences. Can you see a link with the theoretical insights?

**Remark:** The flag `-Rpass=loop-vectorize -S` on `clang` allows you to check if a loop is effectively vectorized.

#### Exercise 2
##### Files concerned

- `second_no_simd.cpp`,   
- `second_simd.cpp`,
- `second.hpp`, 
- `second.cpp`.

Observe the behavior when storing the final state (after 50 steps) for each initial condition. Specifically, let:

$$
    y^{(i)} = x_{50}^{(i)}, \quad i=1,\dots,N,
$$

where each $`x_{50}^{(i)}`$ is obtained by propagating the dynamics through the map $`f`$ defined above. Compile and run the file to determine whether you obtain a gain in running time.
```bash
clang++ -O3 -march=native second.cpp second_no_simd.cpp second_simd.cpp -o second_demo
```


---

## Part 2: OpenMP

##### File concerned
- `pi.cpp`.

Consider the the integral and the infinite series used to compute $\pi$,

$$
\pi = 4 \int_{0}^{1} \frac{1}{1+x^2} \mathrm{d}x = 4\sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{2n-1}
= 4\left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots \right).
$$

This series is known as the **Gregory–Leibniz series**.  
As the number of terms tends to infinity, the series converges to the value of $\pi$.

However, the convergence of this series is **very slow**, meaning that a large number of terms is required to obtain high numerical accuracy.

In this exercise session, we will compute the first $10^9$ terms of this series in two different ways.


#### Exercise 1. Serial Computation

Implement the summation using standard sequential code (without parallelism) and record the runtime. You are expected to implement the `for` loop that computes the Gregory–Leibniz series.



#### Exercise 2. Parallel Computation

Create multiple threads using **OpenMP** to parallelize the summation and observe the performance improvement. Analyze how the runtime and speedup change as the number of threads increases.


#### Exercise 3. Performance comparison

Vary the number of terms in the sum. Create a `csv` file with four columns : Number of threads, number of terms, runtime for serial computation, runtime for parallel computation. Then, import this `csv` and plot it with your preferred plotting tool. Hint: look at `fstream`.

---
### Performance Improvement

The performance gain (speedup) is defined as

$$
\text{Speedup} = \frac{t_{\text{serial}}}{t_{\text{parallel}}}.
$$

---

### Calling OpenMP

We will study the following OpenMP directive:

`#pragma omp parallel for reduction(+ : sum) num_threads(#)`

This directive parallelizes the loop and safely combines partial sums using a reduction operation.

---
