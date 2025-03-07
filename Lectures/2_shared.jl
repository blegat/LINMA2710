### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 78503ab7-f0a3-4231-8b4e-5fd30715ec27
import Pkg

# ╔═╡ 58758402-50e7-4d7b-b4aa-4b0dcb137869
Pkg.activate(@__DIR__)

# ╔═╡ 34519b36-0e60-4c2c-92d6-3b8ed71e6ad1
using MyUtils, PlutoUI, PlutoUI.ExperimentalLayout, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools, Markdown

# ╔═╡ d537aa7e-f38a-11ef-3bef-b7291789fea9
header("LINMA2710 - Scientific Computing
Shared-Memory Multiprocessing", "P.-A. Absil and B. Legat")

# ╔═╡ 3887824b-7c7f-4c24-bf6d-7a55ed7adc89
section("Memory layout")

# ╔═╡ dbe9c6d8-611e-46f9-9a8e-2b3647e813fa
begin
    dir = mktempdir()
    file = joinpath(dir, "topo.png")
    run(`lstopo $file`)
    img(file)
end

# ╔═╡ 37d9b5f0-48b6-4ff3-873d-592230687995
frametitle("Hierarchy")

# ╔═╡ 138caa9b-1d53-4c01-a3b9-c1a097413736
img(URL("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/hierarchy.jpg"))

# ╔═╡ 81465bf1-8e54-461f-892c-2769bf94fdfe
md"""Latency of `n` bytes of data is given by
```math
\alpha + \beta n
```
where ``\alpha`` is the start up time and ``\beta`` is the inverse of the bandwidth.
"""

# ╔═╡ a32ba8f2-a9c9-41c6-99b4-577f0823bd9f
frametitle("Cache lines and prefetch")

# ╔═╡ 02be0de6-70dc-4cf4-b630-b541a304eecd
img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/prefetch.jpeg")

# ╔═╡ 658ca396-2d73-4c93-8138-33c101deee7b
md"""
* Accessing value not in the cache → *cache miss*
* This value is then loaded along with a whole cache line (e.g., 64 or 128 contiguous bytes)
* Following cache lines may also be anticipated and prefetched

This shows the importance of *data locality*. An algorithm performs better if it accesses data close in memory and in a predictable pattern.
"""

# ╔═╡ f26f0a70-c16b-491d-b4cf-45ca146727c2
frametitle("Illustration with matrices")

# ╔═╡ 81da94b8-1bbf-4773-ba53-d229452cef75
mat = rand(Cfloat, 2^8, 2^8)

# ╔═╡ 98a65469-573e-43b5-9043-f3d0f3198bcc
aside(
	Foldable(
		md"What is the performance issue of this code ?",
		md"The way matrices are represented by Julia in memory is by concatenating all columns as single continuous memory. This means that it is more efficient to access the matrix column by column !
		Switch to column-wise sum $(@bind column_wise CheckBox(default = false))",
	), v_offset = -275
)

# ╔═╡ ccfd4488-a32a-4b35-a922-2e830f91ca08
function c_sum_matrix(T; column_wise)
	code = """
#include <stdio.h>

$T sum($T *mat, int n, int m) {
  $T total = 0;
"""
	idx = column_wise ? 'j' : 'i'
	len = column_wise ? 'm' : 'n'
	code *= """
  for (int $idx = 0; $idx < $len; $idx++) {
"""
	idx = column_wise ? 'i' : 'j'
	len = column_wise ? 'n' : 'm'
	code *= """
	for (int $idx = 0; $idx < $len; $idx++) {
"""
	code *= """
	  total += mat[i + j * n];
	}
  }
  return total;
}
"""
	return CCode(code)
end;

# ╔═╡ fa017c45-6410-4c14-b9a2-ede33759d396
sum_matrix_code, sum_matrix_lib = compile_lib(c_sum_matrix("float"; column_wise), lib = true, cflags = ["-O3", "-mavx2", "-ffast-math"]);

# ╔═╡ 19943be2-1633-48c9-8cb3-2a73fb96e4ae
c_sum(x::Matrix{Cfloat}) = ccall(("sum", sum_matrix_lib), Cfloat, (Ptr{Cfloat}, Cint, Cint), x, size(x, 1), size(x, 2));

# ╔═╡ 5d7cd5e3-5fc2-4835-bea1-c4897467365b
aside(sum_matrix_code, v_offset = -470)

# ╔═╡ c0bda86a-136b-45ca-84ba-7365c367d265
frametitle("Arithmetic intensity")

# ╔═╡ 11b1c6a8-3918-4dda-9028-17af2d6c44c4
md"""
Consider a program requiring `m` load / store operations with memory for `o` arithmetic operations.

* The *arithmetic intensity* is the ratio ``a = o / m``.
* The arithmetic time is ``t_\text{arith} = o / \text{frequency}``
* The data transfer time is ``t_\text{mem} = m / \text{bandwidth} = o / (a \cdot \text{bandwidth})``

As arithmetic operations and data transfer are done in parallel, the time per iteration is
```math
\min(t_\text{arith} / o, t_\text{mem} / o) = 1/\max(\text{frequency}, a \cdot \text{bandwidth})
```
So the number of operations per second is ``\max(\text{frequency}, a \cdot \text{bandwidth})``.

This piecewise linear function in ``a`` gives the *roofline model*.
"""

# ╔═╡ 6e8865f5-84ad-4083-bb19-57ad1b561fab
frametitle("The roofline model")

# ╔═╡ d8238145-9787-40f0-a151-1ef73d8c97ee
hbox([
	img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/roofline1.jpeg", :height => "260px"),
	img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/roofline3.jpeg", :height => "260px"),
])

# ╔═╡ d221bad8-98fb-4c1d-9c9c-66e1b697f023
md"""
* *compute-bound* : For large arithmetic intensity (Alg2 in above picture), performance determined by processor characteristics
* *bandwidth-bound* : For low arithmetic intensity (Alg1 in above picture), performance determined by memory characteristics
* Bandwidth line may be be lowered by inefficient memory access (e.g., no locality)
* Peak performance line may be lowered by inefficient use of CPU (e.g., not using SIMD)
"""

# ╔═╡ 9e78f2a1-0811-4f61-957d-ad4718430f7f
frametitle("Cache hierarchy for a multi-core CPU")

# ╔═╡ 6f70144e-5240-41ef-a719-8a8942e18fee
img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/cache-hierarchy.jpg")

# ╔═╡ e90fd21d-d046-4852-823c-5d7210068923
md"""
*Cache coherence* : Update L1 cache when the corresponding memory is modified by another core.
"""

# ╔═╡ e7445ed8-cbf7-475d-bd67-3df8d9015de2
section("Parallel sum")

# ╔═╡ a144f991-3af4-4f88-a7e8-c1f3e9d7f7e2
aside(md"""
Low level implementation using POSIX Threads (pthreads) covered in "LEPL1503 : Projet 3".
We use the high level $(img(URL("https://upload.wikimedia.org/wikipedia/commons/e/eb/OpenMP_logo.png"), :width => "45pt")) library in this course.
""", v_offset = -1000)

# ╔═╡ d5432907-3e55-4035-9c91-183c37d886ea
aside(vbox([
md"`log_size` = $(@bind log_size Slider(14:24, default = 16, show_value = true))",
md"`num_threads` = $(@bind num_threads Slider(2:8, default = 2, show_value = true))",
]), v_offset = -900)

# ╔═╡ 3a5d674d-7c5b-4dac-b9ae-d65a1e9a5cba
vec = rand(Cfloat, 2^log_size)

# ╔═╡ 1b9fb8aa-71cf-4e69-ad84-666c1b66bb5e
begin
	no_diff = Foldable(
		md"Wait, these didn't make any difference in the benchmark, how can it be ?",
		md"""
The compiler most probably use a register (actually a SIMD register here (if there is any) since we used `#pragma omp simd`) as accumulator for the `for` loop and only stored the value of that register into `total`, `local_results[thread_num]` or `no_false_sharing` (depending on the version).
Despite all this, it is still important to be careful about this issue and not trust the execution on one environment or rely too much on compiler optimizations for the code to be portable.
""",
	)
	false_sharing = Foldable(
		md"This is still a performance issue, can you see why ?",
		md"""
The entries of `local_results` are close to each other in memory. There are therefore very likely going to be part of the same block on cache. This means that when one threads modifies it, the cache block will need to be written and then other threads will need to refresh the value of this block in their cache.  variable `total` is shared between the threads, so its value in the register should be sync'ed between the threads! This is called *false sharing*. Let's fix this ? $(@bind no_false_sharing CheckBox(default = false))
$no_diff
""",
	)
aside(Foldable(
	md"Can you spot the issue in the code ?",
	md"""
The same variable `total` is shared between the threads, so its value in the register should be sync'ed between the threads, this is a performance issue! More importantly, the access to the `total` variable are not **atomic**. Therefore, two threads may read the value, `add`, and then store → only one of the two `add` will then be accounted for! Let's fix this ? $(@bind local_results CheckBox(default = false))
$false_sharing
	""",
), v_offset = -800)
end

# ╔═╡ 19655acd-5880-44fa-ac29-d56faf43e87b
function c_sum_code(T; local_results::Bool, no_false_sharing::Bool, simd::Bool)
	code = """
#include <vector>
#include <stdint.h>
#include <omp.h>
#include <stdio.h>

extern "C" {
$T sum($T *vec, int length, int num_threads, int verbose) {
  $T total = 0;
  omp_set_dynamic(0); // Force the value `num_threads`
  omp_set_num_threads(num_threads);
"""
	if local_results
		code *= """
  std::vector<$T> local_results(num_threads);
"""
	end
	code *= """
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
	int stride = length / num_threads;
    int last = stride * (thread_num + 1);
    if (thread_num + 1 == num_threads)
      last = length;
	if (verbose >= 1)
      fprintf(stderr, "thread id : %d / %d %d:%d\\n", thread_num, omp_get_num_threads(), stride * thread_num, last - 1);
"""
	if no_false_sharing
		code *= """
	$T no_false_sharing = 0;
"""
	end
	if simd
		code *= """
    #pragma omp simd
"""
	end
    code *= """
    for (int i = stride * thread_num; i < last; i++)
      $(local_results ? (no_false_sharing ? "no_false_sharing" : "local_results[thread_num]") : "total") += vec[i];
"""
	if local_results && no_false_sharing
		code *= """
	local_results[thread_num] = no_false_sharing;
"""
	end
	code *= """
  }
"""
	if local_results
		code *= """
  for (int i = 0; i < local_results.size(); i++)
    total += local_results[i];
"""
	end
	code *= """
  return total;
}
}
"""
	return CppCode(code)
end;

# ╔═╡ ebe1cd42-ba25-4538-acbe-353e0e47009e
sum_md_code, sum_lib = compile_lib(c_sum_code("float"; local_results, no_false_sharing, simd = true), lib = true, cflags = ["-O3", "-mavx2", "-I/usr/lib/llvm-18/lib/clang/18/include", "-fopenmp"]);

# ╔═╡ 4b9cfb4d-2355-42e3-be2f-35e2638e984b
sum_md_code

# ╔═╡ 253bd533-99b7-4012-b3f4-e86a2466a919
c_sum(x::Vector{Cfloat}; num_threads = 1, verbose = 0) = ccall(("sum", sum_lib), Cfloat, (Ptr{Cfloat}, Cint, Cint, Cint), x, length(x), num_threads, verbose);

# ╔═╡ 4ae73a28-d945-4c1b-a281-aa4931bf0bfd
@btime c_sum($mat)

# ╔═╡ 62297efe-3a15-4dab-ac17-b823ab3e7933
@btime c_sum($vec, num_threads = 1, verbose = 0)

# ╔═╡ 31727049-ac0a-45a6-aae0-934d4549b541
@btime c_sum($vec; num_threads, verbose = 0)

# ╔═╡ b5f1d18c-53ad-4441-8ca8-e02d6ab840d0
@time c_sum(vec; num_threads, verbose = 1)

# ╔═╡ 1f45bab8-afb7-4cd2-8a37-1f258f37ad8f
frametitle("Many processors")

# ╔═╡ db24839c-eb42-4d5c-8545-3714abc01bc5
frametitle("Benchmark")

# ╔═╡ d718f117-41da-42ff-9bcd-8bef0e7e6974
md"""
If we have many processors, we may want to speed up the last part as well:
"""

# ╔═╡ 6c021710-5828-4ac0-8619-ce690ba89d5f
aside(vbox([
md"`many_log_size` = $(@bind many_log_size Slider(14:24, default = 16, show_value = true))",
md"`base_num_threads` = $(@bind base_num_threads Slider(2:8, default = 2, show_value = true))",
md"`factor` = $(@bind factor Slider(2:8, default = 2, show_value = true))",
]), v_offset = -500)

# ╔═╡ 96bffd66-24fc-46f7-b211-57e7d27bc316
many_vec = rand(Cfloat, 2^many_log_size)

# ╔═╡ a7118fbb-66d6-44a1-a6ae-839f0e42a3ec
@btime c_sum($many_vec)

# ╔═╡ 050a67f8-7f02-4ac9-8ac4-20327d46c5e8
function many_sum_code(T)
	code = """
#include <omp.h>
#include <stdio.h>

extern "C" {
void sum_to($T *vec, int length, $T *local_results, int num_threads, int verbose) {
  omp_set_dynamic(0); // Force the value `num_threads`
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
	int stride = length / num_threads;
    int last = stride * (thread_num + 1);
    if (thread_num + 1 == num_threads)
      last = length;
	if (verbose >= 1)
      fprintf(stderr, "thread id : %d / %d %d:%d\\n", thread_num, omp_get_num_threads(), stride * thread_num, last - 1);
	$T no_false_sharing = 0;
    #pragma omp simd
    for (int i = stride * thread_num; i < last; i++)
      no_false_sharing += vec[i];
	local_results[thread_num] = no_false_sharing;
  }
}

$T sum($T *vec, int length, int num_threads, int factor, int verbose) {
  $T* buffers[2] = {new $T[num_threads], new $T[num_threads / factor]};
  sum_to(vec, length, buffers[0], num_threads, verbose);
  int prev = num_threads, cur;
  int buffer_idx = 0;
  for (cur = num_threads / factor; cur > 0; cur /= factor) {
	sum_to(buffers[buffer_idx % 2], prev, buffers[(buffer_idx + 1) % 2], cur, verbose);
	prev = cur;
	buffer_idx += 1;
  }
  if (prev == 1)
	return buffers[buffer_idx % 2][0];
  sum_to(buffers[buffer_idx % 2], prev, buffers[(buffer_idx + 1) % 2], 1, verbose);
  return buffers[(buffer_idx + 1) % 2][0];
}
}
"""
	return CppCode(code)
end;

# ╔═╡ 8e337fad-abcf-4ad3-bf75-ab3980f36baa
many_sum_md_code, many_sum_lib = compile_lib(many_sum_code("float"), lib = true, cflags = ["-O3", "-mavx2", "-fopenmp"]);

# ╔═╡ 258817e3-8495-4136-8cb9-00a4475245b2
many_sum_md_code

# ╔═╡ 6657e4dd-f5c2-47c4-b0d6-a2a56aac7b96
many_sum(x::Vector{Cfloat}; base_num_threads = 1, factor = 2, verbose = 0) = ccall(("sum", many_sum_lib), Cfloat, (Ptr{Cfloat}, Cint, Cint, Cint, Cint), x, length(x), base_num_threads, factor, verbose);

# ╔═╡ 947b8e5c-9cb7-4fe6-aff6-48416879fb43
@time many_sum(vec; base_num_threads, factor, verbose = 1)

# ╔═╡ 910fc9b2-c57d-4874-b8b2-df440fc921c0
@btime many_sum($many_vec; base_num_threads, factor)

# ╔═╡ f95dd40b-8c56-4e10-abbc-3dbb58148e1f
section("Amdahl's law")

# ╔═╡ 2a1f3d29-4d6b-4634-86f3-4ecd4a7821a2
frametitle("Speed-up and efficency")

# ╔═╡ b2b3beda-c8bf-4616-b1bd-bdd907d11636
hbox([
definition("Speed-up", md"""
```math
S_p = \frac{T_1}{T_p}
```"""),
Div(definition("Efficiency", md"""
```math
E_p = \frac{S_p}{p}
```"""); style = Dict("margin-left" => "30px")),
	Div(md"""
Let ``T_p`` bet the time with ``p`` processes
* ``E_p > 1`` → Unlikely
* ``E_p = 1`` → Ideal
* ``E_p < 1`` → Realistic
	"""; style = Dict("margin" => "30px", "flex-grow" => "1")),

]; style = Dict("align-items" => "center", "justify-content" => "center"))

# ╔═╡ b26ab400-ce89-4a76-ad48-464ac6821dd2
frametitle("Amdahl's law")

# ╔═╡ 4b7a62a4-1e88-410b-8549-3021f6cdf6da
md"""
* ``F_s`` : Percentage of ``T_1`` that is sequential
* ``F_p = 1 - F_s`` : Percentage of ``T_1`` that is parallelizable

```math
\begin{align}
T_p &= T_1F_s + T_1F_p/p\\
S_p &= \frac{1}{F_s + F_p/p} & E_p &= \frac{1}{pF_s + F_p}\\
\lim_{p \to \infty} S_p &= \frac{1}{F_s}
\end{align}
```
"""

# ╔═╡ e83baa29-ad2b-4ffc-99f5-cdbca9e31233
frametitle("Application to parallel sum")

# ╔═╡ f7da896d-089c-4430-b82c-db86c380b171
md"""
The first `sum_to` takes ``n/p`` operations.
Assuming `factor` is `2`, there is one operation for each of the ``\log_2(p)`` subsequent `sum_to`.
```math
\begin{align}
  T_1 & = n\\
  T_p & = n/p + \log_2(p)\\
  S_p & = \frac{1}{1/p + \log_2(p)/n} & E_p & = \frac{1}{1 + p\log_2(p)/n}
\end{align}
```"""

# ╔═╡ d1aef3d4-33d1-4151-8ba3-2169f734ea6b
Foldable(md"How to get ``1/F_s = \lim_{p \to \infty} S_p`` ?", md"""
The algorithm cannot use more than ``n`` processes so if ``p \ge n``, we have
``T_p = \log_2(n)``.
Therefore, ``\lim_{p \to \infty} S_p = \frac{1}{\log_2(n)}`` hence ``F_s = \log_2(n)``.
""")

# ╔═╡ dadaf83a-ac35-4a04-827a-1e4e69177e04
biblio = load_biblio!()

# ╔═╡ 8a51b9c5-8888-4578-ae40-cf906ec9b5fa
bibrefs(biblio, "eijkhout2010Introduction")

# ╔═╡ e867d9be-5668-4756-af7f-c23c48962f08
aside(bibcite(biblio, "eijkhout2010Introduction", "Figure 1.5"), v_offset = -300)

# ╔═╡ caec43a3-9bac-4f73-a8e5-288cfa9e1606
aside(bibcite(biblio, "eijkhout2010Introduction", "Figure 1.11"), v_offset = -280)

# ╔═╡ de0bbef2-1240-4f85-889f-0af509d6cfff
	aside(tip(md"""
See examples in $(bibcite(biblio, "eijkhout2010Introduction", "Section 1.7.1")).
"""), v_offset=-300)

# ╔═╡ ea9ff1a9-615d-4e18-a4c8-9aad20447156
aside(bibcite(biblio, "eijkhout2010Introduction", "Figure 1.16"), v_offset = -360)

# ╔═╡ c6ea9bc5-bb15-4e25-a854-de3417d736a6
aside(bibcite(biblio, "eijkhout2010Introduction", "Figure 1.13"), v_offset = -250)

# ╔═╡ 8b98b33e-f65d-4cbd-9e80-20a7132cd349
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.2

# ╔═╡ Cell order:
# ╟─d537aa7e-f38a-11ef-3bef-b7291789fea9
# ╟─8a51b9c5-8888-4578-ae40-cf906ec9b5fa
# ╟─3887824b-7c7f-4c24-bf6d-7a55ed7adc89
# ╟─dbe9c6d8-611e-46f9-9a8e-2b3647e813fa
# ╟─37d9b5f0-48b6-4ff3-873d-592230687995
# ╟─138caa9b-1d53-4c01-a3b9-c1a097413736
# ╟─81465bf1-8e54-461f-892c-2769bf94fdfe
# ╟─e867d9be-5668-4756-af7f-c23c48962f08
# ╟─a32ba8f2-a9c9-41c6-99b4-577f0823bd9f
# ╟─02be0de6-70dc-4cf4-b630-b541a304eecd
# ╟─658ca396-2d73-4c93-8138-33c101deee7b
# ╟─caec43a3-9bac-4f73-a8e5-288cfa9e1606
# ╟─f26f0a70-c16b-491d-b4cf-45ca146727c2
# ╠═4ae73a28-d945-4c1b-a281-aa4931bf0bfd
# ╠═81da94b8-1bbf-4773-ba53-d229452cef75
# ╠═19943be2-1633-48c9-8cb3-2a73fb96e4ae
# ╟─5d7cd5e3-5fc2-4835-bea1-c4897467365b
# ╟─98a65469-573e-43b5-9043-f3d0f3198bcc
# ╟─fa017c45-6410-4c14-b9a2-ede33759d396
# ╟─ccfd4488-a32a-4b35-a922-2e830f91ca08
# ╟─c0bda86a-136b-45ca-84ba-7365c367d265
# ╟─11b1c6a8-3918-4dda-9028-17af2d6c44c4
# ╟─de0bbef2-1240-4f85-889f-0af509d6cfff
# ╟─6e8865f5-84ad-4083-bb19-57ad1b561fab
# ╟─d8238145-9787-40f0-a151-1ef73d8c97ee
# ╟─d221bad8-98fb-4c1d-9c9c-66e1b697f023
# ╟─ea9ff1a9-615d-4e18-a4c8-9aad20447156
# ╟─9e78f2a1-0811-4f61-957d-ad4718430f7f
# ╟─6f70144e-5240-41ef-a719-8a8942e18fee
# ╟─e90fd21d-d046-4852-823c-5d7210068923
# ╟─c6ea9bc5-bb15-4e25-a854-de3417d736a6
# ╟─e7445ed8-cbf7-475d-bd67-3df8d9015de2
# ╟─4b9cfb4d-2355-42e3-be2f-35e2638e984b
# ╠═62297efe-3a15-4dab-ac17-b823ab3e7933
# ╠═31727049-ac0a-45a6-aae0-934d4549b541
# ╠═b5f1d18c-53ad-4441-8ca8-e02d6ab840d0
# ╠═3a5d674d-7c5b-4dac-b9ae-d65a1e9a5cba
# ╠═253bd533-99b7-4012-b3f4-e86a2466a919
# ╟─a144f991-3af4-4f88-a7e8-c1f3e9d7f7e2
# ╟─d5432907-3e55-4035-9c91-183c37d886ea
# ╟─1b9fb8aa-71cf-4e69-ad84-666c1b66bb5e
# ╟─ebe1cd42-ba25-4538-acbe-353e0e47009e
# ╟─19655acd-5880-44fa-ac29-d56faf43e87b
# ╟─1f45bab8-afb7-4cd2-8a37-1f258f37ad8f
# ╟─258817e3-8495-4136-8cb9-00a4475245b2
# ╟─db24839c-eb42-4d5c-8545-3714abc01bc5
# ╟─d718f117-41da-42ff-9bcd-8bef0e7e6974
# ╠═947b8e5c-9cb7-4fe6-aff6-48416879fb43
# ╠═a7118fbb-66d6-44a1-a6ae-839f0e42a3ec
# ╠═910fc9b2-c57d-4874-b8b2-df440fc921c0
# ╠═96bffd66-24fc-46f7-b211-57e7d27bc316
# ╠═6657e4dd-f5c2-47c4-b0d6-a2a56aac7b96
# ╟─6c021710-5828-4ac0-8619-ce690ba89d5f
# ╟─8e337fad-abcf-4ad3-bf75-ab3980f36baa
# ╟─050a67f8-7f02-4ac9-8ac4-20327d46c5e8
# ╟─f95dd40b-8c56-4e10-abbc-3dbb58148e1f
# ╟─2a1f3d29-4d6b-4634-86f3-4ecd4a7821a2
# ╟─b2b3beda-c8bf-4616-b1bd-bdd907d11636
# ╟─b26ab400-ce89-4a76-ad48-464ac6821dd2
# ╟─4b7a62a4-1e88-410b-8549-3021f6cdf6da
# ╟─e83baa29-ad2b-4ffc-99f5-cdbca9e31233
# ╟─f7da896d-089c-4430-b82c-db86c380b171
# ╟─d1aef3d4-33d1-4151-8ba3-2169f734ea6b
# ╟─78503ab7-f0a3-4231-8b4e-5fd30715ec27
# ╟─58758402-50e7-4d7b-b4aa-4b0dcb137869
# ╟─34519b36-0e60-4c2c-92d6-3b8ed71e6ad1
# ╟─dadaf83a-ac35-4a04-827a-1e4e69177e04
# ╠═8b98b33e-f65d-4cbd-9e80-20a7132cd349
