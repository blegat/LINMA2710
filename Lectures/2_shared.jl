### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 34519b36-0e60-4c2c-92d6-3b8ed71e6ad1
using PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools, Markdown, SimpleClang

# ╔═╡ 3bff6f16-ad66-4aed-9a29-4bdcfe41b57f
@htl("""
<p align=center style=\"font-size: 40px;\">LINMA2710 - Scientific Computing
Shared-Memory Multiprocessing</p><p align=right><i>P.-A. Absil and B. Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ 8a51b9c5-8888-4578-ae40-cf906ec9b5fa
md"[Eij10] V. Eijkhout. [Introduction to High Performance Scientific Computing.](https://theartofhpc.com/istc.html) 3 Edition, Vol. 1 (Lulu.com, 2010). "

# ╔═╡ d52b5227-0ef4-4dc6-b443-03364b0caeff
bibcite(what) = "[Eij10; " * what * "]";

# ╔═╡ 3887824b-7c7f-4c24-bf6d-7a55ed7adc89
md"# Memory layout"

# ╔═╡ 655d59f3-665b-410b-b534-afadae96295b
aside(md"Try it on your laptop!
```sh
$ lstopo
```", v_offset = -200)

# ╔═╡ 37d9b5f0-48b6-4ff3-873d-592230687995
md"## Hierarchy"

# ╔═╡ 81465bf1-8e54-461f-892c-2769bf94fdfe
md"""Latency of `n` bytes of data is given by
```math
\alpha + \beta n
```
where ``\alpha`` is the start up time and ``\beta`` is the inverse of the bandwidth.
"""

# ╔═╡ e867d9be-5668-4756-af7f-c23c48962f08
aside(bibcite("Figure 1.5"), v_offset = -300)

# ╔═╡ a32ba8f2-a9c9-41c6-99b4-577f0823bd9f
md"## Cache lines and prefetch"

# ╔═╡ 658ca396-2d73-4c93-8138-33c101deee7b
md"""
* Accessing value not in the cache → *cache miss*
* This value is then loaded along with a whole cache line (e.g., 64 or 128 contiguous bytes)
* Following cache lines may also be anticipated and prefetched

This shows the importance of *data locality*. An algorithm performs better if it accesses data close in memory and in a predictable pattern.
"""

# ╔═╡ caec43a3-9bac-4f73-a8e5-288cfa9e1606
aside(bibcite("Figure 1.11"), v_offset = -280)

# ╔═╡ f26f0a70-c16b-491d-b4cf-45ca146727c2
md"## Illustration with matrices"

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
md"## Arithmetic intensity"

# ╔═╡ 11b1c6a8-3918-4dda-9028-17af2d6c44c4
md"""
Consider a program requiring `m` load / store operations with memory for `o` arithmetic operations.

* The *arithmetic intensity* is the ratio ``a = o / m``.
* The arithmetic time is ``t_\text{arith} = o / \text{frequency}``
* The data transfer time is ``t_\text{mem} = m / \text{bandwidth} = o / (a \cdot \text{bandwidth})``

As arithmetic operations and data transfer are done in parallel, the time per iteration is
```math
\max(t_\text{arith}, t_\text{mem}) / o = 1/\min(\text{frequency}, a \cdot \text{bandwidth})
```
So the number of operations per second is ``\min(\text{frequency}, a \cdot \text{bandwidth})``.

This piecewise linear function in ``a`` gives the *roofline model*.
"""

# ╔═╡ de0bbef2-1240-4f85-889f-0af509d6cfff
	aside(tip(md"""
See examples in $(bibcite("Section 1.6.1")).
"""), v_offset=-300)

# ╔═╡ 6e8865f5-84ad-4083-bb19-57ad1b561fab
md"## The roofline model"

# ╔═╡ d221bad8-98fb-4c1d-9c9c-66e1b697f023
md"""
* *compute-bound* : For large arithmetic intensity (Alg2 in above picture), performance determined by processor characteristics
* *bandwidth-bound* : For low arithmetic intensity (Alg1 in above picture), performance determined by memory characteristics
* Bandwidth line may be lowered by inefficient memory access (e.g., no locality)
* Peak performance line may be lowered by inefficient use of CPU (e.g., not using SIMD)
"""

# ╔═╡ ea9ff1a9-615d-4e18-a4c8-9aad20447156
aside(bibcite("Figure 1.16"), v_offset = -360)

# ╔═╡ 9e78f2a1-0811-4f61-957d-ad4718430f7f
md"## Cache hierarchy for a multi-core CPU"

# ╔═╡ e90fd21d-d046-4852-823c-5d7210068923
md"""
*Cache coherence* : Update L1 cache when the corresponding memory is modified by another core.
"""

# ╔═╡ c6ea9bc5-bb15-4e25-a854-de3417d736a6
aside(bibcite("Figure 1.13"), v_offset = -250)

# ╔═╡ e7445ed8-cbf7-475d-bd67-3df8d9015de2
md"# Parallel sum"

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
md"## Many processors"

# ╔═╡ db24839c-eb42-4d5c-8545-3714abc01bc5
md"## Benchmark"

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
md"# Amdahl's law"

# ╔═╡ 2a1f3d29-4d6b-4634-86f3-4ecd4a7821a2
md"## Speed-up and efficency"

# ╔═╡ b26ab400-ce89-4a76-ad48-464ac6821dd2
md"## Amdahl's law"

# ╔═╡ 4b7a62a4-1e88-410b-8549-3021f6cdf6da
md"""
* ``F_s`` : Fraction of ``T_1`` that is sequential
* ``F_p = 1 - F_s`` : Fraction of ``T_1`` that is parallelizable

```math
\begin{align}
T_p &= T_1F_s + T_1F_p/p\\
S_p &= \frac{1}{F_s + F_p/p} & E_p &= \frac{1}{pF_s + F_p}\\
\lim_{p \to \infty} S_p &= \frac{1}{F_s}
\end{align}
```
"""

# ╔═╡ e83baa29-ad2b-4ffc-99f5-cdbca9e31233
md"## Application to parallel sum"

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
``T_p = 1 + \log_2(n)``.
Therefore, ``\lim_{p \to \infty} S_p = S_n = \frac{n}{1 + \log_2(n)}``. Ignoring the constant ``1``, we get ``F_s = \log_2(n)/n``.
""")

# ╔═╡ 8b98b33e-f65d-4cbd-9e80-20a7132cd349
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.2

# ╔═╡ 91f7f3fe-cb4d-4461-9147-81c72b650a4b
begin
struct Path
    path::String
end

function imgpath(path::Path)
    file = path.path
    if !('.' in file)
        file = file * ".png"
    end
    return joinpath(joinpath(@__DIR__, "images", file))
end

function img(path::Path, args...; kws...)
    return PlutoUI.LocalResource(imgpath(path), args...)
end

struct URL
    url::String
end

function save_image(url::URL, html_attributes...; name = split(url.url, '/')[end], kws...)
    path = joinpath("cache", name)
    return PlutoTeachingTools.RobustLocalResource(url.url, path, html_attributes...), path
end

function img(url::URL, args...; kws...)
    r, _ = save_image(url, args...; kws...)
    return @htl("<a href=$(url.url)>$r</a>")
end

function img(file::String, args...; kws...)
    if startswith(file, "http")
        img(URL(file), args...; kws...)
    else
        img(Path(file), args...; kws...)
    end
end
end

# ╔═╡ dbe9c6d8-611e-46f9-9a8e-2b3647e813fa
begin
    dir = mktempdir()
    file = joinpath(dir, "topo.png")
    run(`lstopo $file`)
    img(file)
end

# ╔═╡ 138caa9b-1d53-4c01-a3b9-c1a097413736
img(URL("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/hierarchy.jpg"))

# ╔═╡ 02be0de6-70dc-4cf4-b630-b541a304eecd
img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/prefetch.jpeg")

# ╔═╡ d8238145-9787-40f0-a151-1ef73d8c97ee
hbox([
	img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/roofline1.jpeg", :height => "260px"),
	img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/roofline3.jpeg", :height => "260px"),
])

# ╔═╡ 6f70144e-5240-41ef-a719-8a8942e18fee
img("https://github.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/raw/refs/heads/main/booksources/graphics/cache-hierarchy.jpg")

# ╔═╡ a144f991-3af4-4f88-a7e8-c1f3e9d7f7e2
aside(md"""
Low level implementation using POSIX Threads (pthreads) covered in "LEPL1503 : Projet 3".
We use the high level $(img(URL("https://upload.wikimedia.org/wikipedia/commons/e/eb/OpenMP_logo.png"), :width => "45pt")) library in this course.
""", v_offset = -1000)

# ╔═╡ cd6c2bda-e80d-46ba-8242-f0f61a250471
function definition(name, content)
    return Markdown.MD(Markdown.Admonition("key-concept", "Def: $name", [content]))
end

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SimpleClang = "d80a2e99-53a4-4f81-9fa2-fda2140d535e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~1.0.0"
Luxor = "~4.4.1"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.79"
SimpleClang = "~0.1.0"
StaticArrays = "~1.9.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "dd10f104cd42c7c2e99d0e31aeb629e0242d1f87"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a21c5464519504e41e0cbc91f0188e8ca23d7440"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+1"

[[deps.Clang_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML", "Zlib_jll", "libLLVM_jll"]
git-tree-sha1 = "f85df021a5fd31ac59ea7126232b2875a848544f"
uuid = "0ee61d77-7f21-5576-8119-9fcc46b10100"
version = "18.1.7+4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "Cairo_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pango_jll", "XML2_jll", "gdk_pixbuf_jll"]
git-tree-sha1 = "e6ab5dda9916d7041356371c53cdc00b39841c31"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.54.7+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "DataStructures", "Dates", "FFMPEG", "FileIO", "PolygonAlgorithms", "PrecompileTools", "Random", "Rsvg"]
git-tree-sha1 = "e820980fe5635ec27cc96d2cd407f16e72169866"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "4.4.1"

    [deps.Luxor.extensions]
    LuxorExtLatex = ["LaTeXStrings", "MathTeXEngine"]
    LuxorExtTypstry = ["Typstry"]

    [deps.Luxor.weakdeps]
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MultilineStrings]]
git-tree-sha1 = "8c49220ba78101000fcbbf9cb858010dd9b74a7b"
uuid = "1e8d2bf6-9821-4900-9a2f-4d87552df2bd"
version = "1.0.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PolygonAlgorithms]]
git-tree-sha1 = "5608c3c5b78134cd5da29571ef3736077408031f"
uuid = "32a0d02f-32d9-4438-b5ed-3a2932b48f96"
version = "0.3.5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "e53dad0507631c0b8d5d946d93458cbabd0f05d7"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.1.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SimpleClang]]
deps = ["Clang_jll", "InteractiveUtils", "LLVMOpenMP_jll", "Markdown", "MultilineStrings"]
git-tree-sha1 = "b3d3225c2513bedab65df13f7968c3ab48e785cc"
uuid = "d80a2e99-53a4-4f81-9fa2-fda2140d535e"
version = "0.1.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0f529006004a8be48f1be25f3451186579392d47"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.17"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "895f21b699121d1a57ecac57e65a852caf569254"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.13+0"

[[deps.libLLVM_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8f36deef-c2a5-5394-99ed-8e07531fb29a"
version = "18.1.7+5"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╟─3bff6f16-ad66-4aed-9a29-4bdcfe41b57f
# ╟─8a51b9c5-8888-4578-ae40-cf906ec9b5fa
# ╟─d52b5227-0ef4-4dc6-b443-03364b0caeff
# ╟─3887824b-7c7f-4c24-bf6d-7a55ed7adc89
# ╟─dbe9c6d8-611e-46f9-9a8e-2b3647e813fa
# ╟─655d59f3-665b-410b-b534-afadae96295b
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
# ╠═ea9ff1a9-615d-4e18-a4c8-9aad20447156
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
# ╠═34519b36-0e60-4c2c-92d6-3b8ed71e6ad1
# ╠═8b98b33e-f65d-4cbd-9e80-20a7132cd349
# ╟─91f7f3fe-cb4d-4461-9147-81c72b650a4b
# ╟─cd6c2bda-e80d-46ba-8242-f0f61a250471
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
