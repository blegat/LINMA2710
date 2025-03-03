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

# ╔═╡ 406dcb0d-b68d-40ec-8844-c5445cff88f6
import Pkg

# ╔═╡ dfc87f8b-1b29-4f3a-b674-6ddf16b9d1ce
Pkg.activate(".")

# ╔═╡ d3c2d2b7-8f23-478b-b36b-c92552a6cf01
using MyUtils, PlutoUI, PlutoUI.ExperimentalLayout, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools

# ╔═╡ 49aca9a0-ed40-11ef-1cf9-635242dfa821
header("LINMA2710 - Scientific Computing
Single Instruction Multiple Data (SIMD)", "P.-A. Absil and B. Legat")

# ╔═╡ 4d0f2c46-4651-4ba9-b08d-44c8494d2b60
section("Motivation")

# ╔═╡ 74d55b53-c917-460a-b59c-71b1f07f7cba
frametitle("The need for parallelism")

# ╔═╡ b348eb57-446b-42ec-9292-d5a77cd26e0c
RobustLocalResource("https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png", "cache/42-years-processor-trend.png")

# ╔═╡ b6ae6fcc-a77e-49c5-b380-06854844469e
md"[Image source](https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/)."

# ╔═╡ 74ae5855-85e8-4615-bf98-e7819bc053d2
frametitle("A bit of historical context")

# ╔═╡ 0d17955c-6ddc-4d57-8600-8ad3229d4631
hbox([md"""
* **1972** : C language created by Dennis Ritchie and Ken Thompson to ease development of Unix (previously developed in **assembly**)
* **1985** : C++ created by Bjarne Stroustrup
* **2003** : LLVM started at University of Illinois
* **2005** : Apple hires Chris Lattner from the university
* **2007** : He then creates the LLVM-based compiler Clang
* **2009** : Mozilla start developing an LLVM-based compiler for Rust
* **2009** : Develpment starts on Julia, with LLVM-based compiler
""",
	Div(md"""$(@draw begin
	    placeimage_from_url("https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg", Point(-100, -100), scale = 0.1)
	    arrow(Point(-85, -85), Point(-20, -15))
	    placeimage_from_url("https://raw.githubusercontent.com/rust-lang/www.rust-lang.org/master/static/images/rust-social-wide-light.svg", Point(0, -100), scale = 0.15)
	    arrow(Point(0, -85), Point(0, -15))
	    placeimage_from_url("https://julialang.org/assets/infra/logo.svg", Point(100, -100), scale = 0.15)
	    arrow(Point(85, -85), Point(20, -15))
		placeimage_from_url("https://llvm.org/img/LLVMWyvernSmall.png", Point(-30, 30), scale = 0.08)
		boxed("LLVM Intermediate Representation (IR)", Point(0, 0))
		arrow(Point(0, 10), Point(0, 85))
		boxed("Assembly", Point(0, 100))
	end 300 300)"""; style = Dict("width" => "100%")),
])

# ╔═╡ f6674345-4b71-40f3-8d42-82697990d534
frametitle("A sum function in C and Julia")

# ╔═╡ baf29a4d-337c-430c-b382-9b2dab7ce69a
function julia_sum(v::Vector{T}) where {T}
	total = zero(T)
	for i in eachindex(v)
		total += v[i]
	end
	return total
end

# ╔═╡ ec98ab34-cb2b-48c1-a9d2-3fa9c7821d11
frametitle("Let's make a small benchmark")

# ╔═╡ 8b7c3a6e-bd6a-425e-8040-340fdb6b0dd0
vec_float = rand(Float32, 2^16)

# ╔═╡ 691d01a2-12fc-4782-a9f9-a732746285c6
@btime c_sum($vec_float)

# ╔═╡ 0b4c686c-912b-42ff-a7ef-970030808a74
@btime julia_sum($vec_float)

# ╔═╡ 8f4e6abd-8da8-42a5-b69f-ae76fa8fcf6b
aside(tip(md"As accessing global variables is slow in Julia, it is important to add `$` in front of them when using `btime`. This is less critical in Pluto though as it handles global variables differently. To see why, try removing the `$`, you should see `1` allocations instead of zero."); v_offset=-400)

# ╔═╡ 9956af59-12e9-4eb6-bf63-03e2936a5912
sum_float_options = hbox([
	Div(vbox([
		md"""OpenMP : $(@bind sum_float_pragma_openmp Select(["No pragma", "simd"]))""",
		md"""$(@bind sum_float_pragma_fastmath Select(["No pragma", "float_control(precise, off)"]))""",
		md"""Vectorize : $(@bind sum_float_pragma_vectorize Select(["No pragma", "vectorize(disable)", "vectorize(enable)", "vectorize_width(1)", "vectorize_width(2)", "vectorize_width(4)", "vectorize_width(8)", "vectorize_width(16)"]))""",
		md"""Interleave : $(@bind sum_float_pragma_interleave Select(["No pragma", "interleave(disable)", "interleave(enable)", "interleave_count(1)", "interleave_count(2)", "interleave_count(4)", "interleave_count(8)"]))"""]),
		; style = Dict("flex-grow" => "1")
	),
	vbox([
		hbox([
	    	md"""$(@bind sum_float_opt Select(["-O0", "-O1", "-O2", "-O3"], default = "-O0"))""",
			md"""$(@bind sum_float_flags MultiCheckBox(["-ffast-math", "-fopenmp"]))""",
		]),
	    md"""$(@bind sum_float_m MultiCheckBox(["-msse3", "-mavx2", "-mavx512f"]))""",
	])
]);

# ╔═╡ 0a19c69e-d9f1-4630-a8b4-5718e4f1abfa
qa(md"How to speed up the C code ?",
md"""
Try passing the following flags to Clang by selecting them and waiting for the benchmark timing to refresh: $(sum_float_options)

What are they doing ? We'll see in the slide...
"""
)

# ╔═╡ 2a404744-686c-4b8a-988a-8ff99603f2d4
frametitle("Summing with SIMD")

# ╔═╡ 2acc14b4-4e65-4dc1-950a-df9ed3a0892d
Resource(
	"https://i0.wp.com/juliacomputing.com/assets/img/new/auto-vectorization2.png",
	:alt => "SIMD"
)

# ╔═╡ 66a18765-b8a4-41af-8711-80d08b0ef4c4
frametitle("Faster Julia code")

# ╔═╡ f853de2d-ca27-42d6-af9a-194ee6bb7d89
Foldable(md"How to get the same speed up from the Julia code ?", md"
* The `-O3` option need to be passed to the `julia` session that started Pluto, `-O2` is used by default.
* Instead of applying `-fast-math` to the whole library, the macro `@fastmath` allows to apply it to a selected part of the code.
* In order to accurately throw the out of bound error for the **first** index that is out of bound, Julia will prevent SIMD to be applied. The bound checking also makes it harder to parallelise. To circumvent this, check the bounds outside of the loop and then use `@inbounds` to disable bound checks inside the loop.
* The use of SIMD can also be forced with `@simd`.")

# ╔═╡ e437157d-e30a-498f-a031-a603048caed0
function julia_sum_fast(v::Vector{T}) where {T}
	total = zero(T)
	for i in eachindex(v)
		@fastmath total += @inbounds v[i]
	end
	return total
end

# ╔═╡ cce70070-5938-4f44-8181-2fb6158c419b
@btime julia_sum_fast($vec_float)

# ╔═╡ ad4e2ac1-6a51-4338-ae38-15a2b817020d
function julia_sum_simd(v::Vector{T}) where {T}
	total = zero(T)
	@simd for i in eachindex(v)
		total += v[i]
	end
	return total
end

# ╔═╡ 70ab5cde-5856-451d-9095-864367b6c207
@btime julia_sum_simd($vec_float)

# ╔═╡ e432159e-f3f2-412d-b559-155674f732f6
frametitle("Careful with fast math")

# ╔═╡ b19154d8-cb88-4aac-b76a-18f647672d70
Foldable(md"Why are the three elements in the center of the vector ignored in this example ?", md"In a large sum, the `total` variable become much larger than each summand. Because of this, significant roundoff errors can occur. These roundoff errors cannot be added to the `total` variable as it is too large but it may be added to the summands as they are smaller so as to compensate the error. Here, instead of considering a large sum, we just used a large first summand to simplify but you can consider `1` as being the sum of a large amounds of preceding elements in the sum to make it more realistic.")

# ╔═╡ 4cd17588-8f3c-447e-890b-fc881575db8d
test_kahan = Cfloat[1.0, eps(Cfloat)/4, eps(Cfloat)/4, eps(Cfloat)/4, 1000eps(Cfloat)]

# ╔═╡ 3469f9fe-2512-4fb9-81b8-dd1d39e20c38
sum(Float64.(test_kahan))

# ╔═╡ 8df0ed24-b5bc-4cf8-b507-37bd8fc79be2
md"To improve the accuracy this, we consider the [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)."

# ╔═╡ c8ae3959-6428-4937-9212-171ea6ab0888
hbox([Div(
		md"""Optimization level : $(@bind sum_kahan_opt Select(["-O0", "-O1", "-O2", "-O3"], default = "-O0"))"""; style = Dict("flex-grow" => "1")),
		md"""Enable `-ffast-math` ? $(@bind sum_kahan_fastmath CheckBox())"""
],
)

# ╔═╡ 52cd9d6e-0e24-45ae-a602-1b9d9edc67ae
Foldable(md"What happens when `-ffast-math` is enabled ?", md"The flag allows LLVM to optimize out the code to be exactly the as the code of `c_sum`! This does not happen at `-O0`, so the optimization level also needs to be increased to see this.")

# ╔═╡ 1a4f7389-9d1b-4008-8896-76ecc409ab1f
md"For further details, see [this blog post](https://simonbyrne.github.io/notes/fastmath/)."

# ╔═╡ a0389720-9ed7-4534-87ac-5b61e5c2470d
aside(tip(md"`eps` gives the difference between `1` and the number closest to `1`. See also `prevfloat` and `nextfloat`."), v_offset = -600)

# ╔═╡ abf284e9-75f1-42f4-b559-8720f56b02a2
sum_kahan_code, sum_kahan_lib = compile_lib("""
float sum_kahan(float* vec, int length) {
    float total, c, t, y;
    int i;
    total = c = 0.0f;
    for (i = 0; i < length; i++) {
      y = vec[i] - c;
      t = total + y;
      c = (t - total) - y;
      total = t;
   }
   return total;
}
""", lib = true, cflags = String[sum_kahan_opt; ifelse(sum_kahan_fastmath, ["-ffast-math"], String[])]);

# ╔═╡ 839f5630-a405-4a2e-9046-cd0d1fd9c37e
c_sum_kahan(x::Vector{Cfloat}) = ccall(("sum_kahan", sum_kahan_lib), Cfloat, (Ptr{Cfloat}, Cint), x, length(x));

# ╔═╡ 919045cb-90cc-4cbc-be2a-5b2580a93de9
c_sum_kahan(test_kahan)

# ╔═╡ ee269b38-a5e1-467a-a91e-f7a7f1f54509
aside(sum_kahan_code, v_offset = -400)

# ╔═╡ 8a552e21-e51b-457d-b974-148537db6cae
section("SIMD inspection")

# ╔═╡ 639e0ece-502b-4379-a932-32c0d119cc2f
frametitle("Instruction sets")

# ╔═╡ 1ddcda8b-fa23-4802-852c-e70b1777c2e4
md"""
The data is **packed** on a single SIMD unit whose width and register depends on the instruction set family.
The single instruction is then run in parallel on all elements of this small **vector** stored in the SIMD unit.
These give the prefix `vp` to the instruction names that stands from *Vectorized Packed*.

| Instruction Set Family | Width of SIMD unit | Register |
|-----------------|-------------------|----------|
| Streaming SIMD Extension (SSE) | 128-bit           | `%xmm`   |
| Advanced Vector Extensions (AVX) | 256-bit           | `%ymm`   |
| AVX-512  | 512-bit           | `%zmm`   |
"""

# ╔═╡ 3afaf82a-4843-4afa-8541-1a26d7e943a1
run(pipeline(`lscpu`, `grep Flag`))

# ╔═╡ 3d4335d5-f526-4869-b3e7-a0b36443cc41
aside(tip(md"To determine which instruction set is supported for your computer, look at the `Flags` list in the output of `lscpu`.
We can check in the [Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#) that `avx`, `avx2` and `avx_vnni` are in the AVX family."), v_offset = -280)

# ╔═╡ c66fc30b-355d-43fa-9950-f943e3a095a6
frametitle("SIMD at LLVM level")

# ╔═╡ 7a5620c8-2ca0-4422-851d-39c5b65226e5
md"How can you check that SIMD is enable ? Let's check at the level of LLVM IR."

# ╔═╡ fc278dad-6133-466b-8c3a-775353bdd64a
function f(x1, x2, x3, x4, y1, y2, y3, y4)
	z1 = x1 + y1
	z2 = x2 + y2
	z3 = x3 + y3
	z4 = x4 + y4
	return z1, z2, z3, z4
end

# ╔═╡ 6869a1d9-b662-4c66-9adb-fc72932eb6c5
@code_llvm debuginfo=:none f(1, 2, 3, 4, 5, 6, 7, 8)

# ╔═╡ 1fa393f5-ccea-4199-bf23-16fc1d6a1969
aside(tip(md"If we see `add i64`, it means that each `Int64` is added independently"), v_offset = -200)

# ╔═╡ 220127a6-dba3-448a-a12d-f9c523009f74
frametitle("Packing the data to enable SIMD")

# ╔═╡ e4ef3a2b-ba92-4c86-9ff2-2b968de27ea5
function f_broadcast(x, y)
	z = x .+ y
	return z
end

# ╔═╡ 765aef1b-ffd1-4851-9b15-0ad9df4980f4
@code_llvm debuginfo=:none f_broadcast((1, 2, 3, 4), (1, 2, 3, 4))

# ╔═╡ 7bbb30c8-3407-4a18-aa50-8b8f6f37e8a3
aside(tip(md"`load <4 x i64>` means that 4 `Int64` are loaded into a 256-bit wide SIMD unit."), v_offset = -200)

# ╔═╡ c7a4a182-6503-4d3d-9f49-8b1b2e3dc499
frametitle("SIMD at assembly level")

# ╔═╡ 9e48a50e-e120-4838-91d7-264522ac1723
@code_native debuginfo=:none f_broadcast((1, 2, 3, 4), (1, 2, 3, 4))

# ╔═╡ 7530ea93-11fd-4931-9dd4-a5e820f8b540
aside(tip(md"The suffic `v` in front of the instruction stands for `vectorized`. It means it is using a SIMD unit."), v_offset = -300)

# ╔═╡ a0abb64b-6dc2-4e98-bdfd-5de9b5c97897
frametitle("Tuples implementing the array interface")

# ╔═╡ a7e4be26-d088-47fe-b0ce-e12cb9936599
md"`N` = $(@bind N Slider(2:4, default=2, show_value = true))"

# ╔═╡ 3a7df4f4-0f7b-4a51-8d6a-dcba9a97c18f
let
    T = Float64
	A = rand(SMatrix{N,N,T})
	x = rand(SVector{N,T})
	@code_llvm debuginfo=:none A * x
end

# ╔═╡ 4d0ba8c4-2d94-400e-a106-467db6e3fc0c
aside(tip(md"Small arrays that are allocated on the stack like tuples and implemented in `StaticArrays.jl`. Operating on them leverages SIMD."), v_offset = -400)

# ╔═╡ 403bb0f1-5514-486e-9f81-fba9d6031ee1
section("Auto-Vectorization")

# ╔═╡ b9ad74c5-d99d-4129-afa2-4ff62eedf796
frametitle("LLVM Loop Vectorizer for a C array")

# ╔═╡ 41d1448e-72c9-431c-a614-c7922e35c883
frametitle("LLVM Loop Vectorizer for a C++ vector")

# ╔═╡ 7ab127df-8afd-4ebe-8403-9ca3bcc2f8e3
@btime cpp_sum($vec_float)

# ╔═╡ 49ca9d35-cce8-45fd-8c2e-1dd92f056c93
aside(tip(md"Easily call C++ code from Julia or Python by adding a C interface like the `c_sum` in this example."), v_offset = -170)

# ╔═╡ 48d3e554-28f3-4ca3-a111-8a9904771426
function cpp_sum_code(T; pragmas = String[], loop_pragmas = String[])
	code = """
#include <vector>

$T my_sum(std::vector<$T> vec) {
  $T total = 0;
"""
	for pragma in loop_pragmas
		code *= """
  #pragma clang loop $pragma
"""
	end
	code *= """
  for (int i = 0; i < vec.size(); i++) {
"""
	for pragma in pragmas
		code *= """
	#pragma $pragma
"""
	end
    code *= """
    total += vec[i];
  }
  return total;
}

extern "C" {
$T c_sum($T *array, int length) {
  std::vector<$T> v;
  v.assign(array, array + length);
  return my_sum(v);
}}"""
	return code
end;

# ╔═╡ 529ba439-40fe-4d93-88c5-797c0a9fc6ee
frametitle("LLVM Superword-Level Parallelism (SLP) Vectorizer")

# ╔═╡ 69c872e1-966a-4a7a-a90f-d13bc108b801
f(a, b) = (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])

# ╔═╡ bfb3b635-85b2-4a1e-a16c-5106b6495d09
@code_llvm debuginfo=:none f((1, 2, 3, 4), (5, 6, 7, 8))

# ╔═╡ 594cb702-35ff-4932-93cb-8cdbd53b7e27
frametitle("Inspection with godbolt Compiler Explorer")

# ╔═╡ aa153cd9-0118-4f2a-802e-fae8c302ad4b
html"""<iframe width="800px" height="400px" src="https://godbolt.org/e#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYMQAJlIOAMngMmABy7gBGmMQgAGykAA6oCoS2DM5uHt7xickCAUGhLBFRsZaY1ilCBEzEBGnunj6l5QKV1QR5IeGRMRZVNXUZjX3tgZ2F3dEAlBaorsTI7BwAbqh46ADU/KgQgQTrTFyk67v7PseC62GH53thZycAVACCk%2BsApADsAEJvGk/r6yebwArF8NCCACLvADMUIO6we6wg8LeXi%2Bly4kze0J%2BfwBQNBXEhMLhXgRSKYZNR6LuWJxv3%2BgJBXy8xOxcK45ORnOpGLpuMZBK%2B0LZsNOXMp7zRly8/IZnwhHGmtE4wN4ng4WlIqE4jnWClm80wUuhPFIBE0SumAGsQNDJAA6SQADi8Gkk0WiGmB0I%2Bzud0hVHEk6st2s4vAUIA05st0zgsCQaBYcTokXIlGTqfoUWQyAMRi4AE4uDGaLQCJEoxAwmGwoFqgBPThm%2BvMYiNgDyYW0ZQt3F4ybYgk7DFozc1vCwYVcwEcYloUYHpCwLEMwHEk5XeGIfbwS0wS61mFUZVclZbvF2mCDWtoeDCxCbziwl/NxDwLDfB%2BIYUSmAhTA1yMe8jDjPgDGABQADU8EwAB3Ts4kYN9%2BEEEQxHYKQZEERQVHULddEOAsTDMfQHyjSBplQOIbAEJcAFpO2hdYGLXJZVC8XhUB/D8sEoiBpiaOjPAgBwBgaHx/FGAoihAD4siSESJO8HwEiUlIOlk7oFOEiphhUrxGhvPcWmGLSuiiXSDJcepVN6NoLPGKyhMNBYJGVVVQy3HUOHWVRnWiBjokkdZ83XdZiwdLgHQ0JFcEIEgTUxXh%2By0SYbTtaEHWhaJnSkaEi2dItgQ%2BaFiuBfROBDUgv1LGMNS1XzI2jWNJ3jGBEBQVAUzTMgKAgLM%2BpQYApB8ctK2Iatay3NsmzfOaO27XtrDfIdGAIUdxzDadZ3nWhFzfVd103LV8F3coDyPXgTzPC9l2vW9eHvR9nwwRYzQID8v2XH8/yUQDgI3QJQHaiCmCg2CEKQlDlzQ4RRHEbD4bwtQw10HwSJAUxjHMF6BOo2iUkY5jWPYzjuN4tZD3gISTOaUTxNsjJDmk/JLL0dSclSZnPEOLmRKcuTDj0sy2hUkX6ZE1oaiF7oRZs9I%2BYc2WZI5zEZjmdyNaDNVSEa7jOH8wLgtC4A80ix0yQgBKiGIZLJlSuNMuhLxoskIqNCLSRvXdD4vAUoMaoN8MOBamM0qtKqOC42qJA0Bqw2atr0umH8kjsSQgA%3D%3D"></iframe>"""

# ╔═╡ ea10cb8a-a95e-400c-be86-1633a3833ec5
md"[Example source](https://llvm.org/docs/Vectorizers.html)"

# ╔═╡ 7dd1fa44-ed35-4abe-853f-58fe4085b441
frametitle("Further readings")

# ╔═╡ fcf5c210-c100-4534-a65b-9bee23c518da
md"""
Slides inspired from:
* [SIMD in Julia](https://www.youtube.com/watch?v=W1hXttRmuks&t=337s)
* [Demystifying Auto-vectorization in Julia](https://www.juliabloggers.com/demystifying-auto-vectorization-in-julia/)
* [Auto-Vectorization in LLVM](https://llvm.org/docs/Vectorizers.html)
"""

# ╔═╡ 8d24ad58-fd1a-43f2-b1ce-ab02dd3a5df6
options = vbox([
	md"""$(@bind sum_pragma_fastmath Select(["No pragma", "float_control(precise, off)"]))""",
	md"""$(@bind sum_pragma_vectorize Select(["No pragma", "vectorize(disable)", "vectorize(enable)", "vectorize_width(1)", "vectorize_width(2)", "vectorize_width(4)", "vectorize_width(8)", "vectorize_width(16)"]))""",
	md"""$(@bind sum_pragma_interleave Select(["No pragma", "interleave(disable)", "interleave(enable)", "interleave_count(1)", "interleave_count(2)", "interleave_count(4)", "interleave_count(8)"]))""",
	md"""Element type : $(@bind sum_type Select(["char", "short", "int", "float", "double"], default = "int"))""",
	md"""Optimization level : $(@bind sum_opt Select(["-O0", "-O1", "-O2", "-O3"], default = "-O0"))""",
	md"""$(@bind sum_flags MultiCheckBox(["-msse3", "-mavx2", "-mavx512f", "-ffast-math"], orientation = :column))""",
]);

# ╔═╡ bc8bc245-6c10-4759-a85b-b407ef016c60
aside(options, v_offset = -260)

# ╔═╡ 1cb7d80a-84a0-41a3-b089-6ffefa44f041
aside(options, v_offset = -330)

# ╔═╡ 8e3738ac-d742-4c60-ade8-f5565ea2d1bf
cpp_sum_float_code, cpp_sum_float_lib = compile_lib(cpp_sum_code("float"), lib = true, cflags = [sum_opt; sum_flags], language = CppLanguage());

# ╔═╡ 57005169-054b-4912-b0ba-742a56ee3f5f
cpp_sum(x::Vector{Cfloat}) = ccall(("c_sum", cpp_sum_float_lib), Cfloat, (Ptr{Cfloat}, Cint), x, length(x));

# ╔═╡ 8c23d4b7-9580-4563-9586-1e32358b9802
cpp_sum_code_for_llvm = cpp_sum_code(
	sum_type,
	pragmas = filter(!isequal("No pragma"), [sum_pragma_fastmath]),
	loop_pragmas = filter(!isequal("No pragma"), [sum_pragma_vectorize, sum_pragma_interleave]),
);

# ╔═╡ 972c1194-9d5f-438a-964f-176713bab912
aside(md_code(cpp_sum_code_for_llvm, CppLanguage()), v_offset = -700)

# ╔═╡ 174407b5-75be-4930-a476-7f2bfa35cdf0
function c_sum_code(T; loop_pragmas = String[], openmp_pragmas = String[], pragmas = String[])
	code = """
$T sum($T *vec, int length) {
    $T total = 0;
"""
	for pragma in loop_pragmas
		code *= """
	#pragma clang loop $pragma
"""
	end
	for pragma in openmp_pragmas
		code *= """
	#pragma omp $pragma
"""
	end
	code *= """
    for (int i = 0; i < length; i++) {
"""
	for pragma in pragmas
		code *= """
	    #pragma $pragma
"""
	end
	code *= """
        total += vec[i];
    }
    return total;
}"""
	return code
end;

# ╔═╡ 1548a494-80a9-4295-a012-88be6de7fcfa
sum_float_code, sum_float_lib = compile_lib(c_sum_code("float",
	pragmas = filter(!isequal("No pragma"), [sum_float_pragma_fastmath]),
	loop_pragmas = filter(!isequal("No pragma"), [sum_float_pragma_vectorize, sum_float_pragma_interleave]),
	openmp_pragmas = filter(!isequal("No pragma"), [sum_float_pragma_openmp]),
), lib = true, cflags = [sum_float_opt; sum_float_flags]);

# ╔═╡ a38807e2-d901-4467-b35e-248da491abff
sum_float_code

# ╔═╡ a841d535-c32b-4bb6-8132-600253038508
c_sum(x::Vector{Cfloat}) = ccall(("sum", sum_float_lib), Cfloat, (Ptr{Cfloat}, Cint), x, length(x));

# ╔═╡ c80ad92b-853d-4bc1-ad7c-0dd1ad48d1c4
c_sum(test_kahan[[1, 5]])

# ╔═╡ 570b50d9-64d8-408a-8f05-6f81716f20c2
c_sum(test_kahan)

# ╔═╡ 1e494794-7c9f-42bb-a06c-d617ee271c9b
aside(sum_float_code, v_offset=-200)

# ╔═╡ 9cfd52a7-f5b9-424a-b1a4-b81f63e3b30c
c_sum_code_for_llvm = c_sum_code(
	sum_type,
	pragmas = filter(!isequal("No pragma"), [sum_pragma_fastmath]),
	loop_pragmas = filter(!isequal("No pragma"), [sum_pragma_vectorize, sum_pragma_interleave]),
);

# ╔═╡ e6fac999-9f54-42f9-a1b7-3fd883b891ab
emit_llvm(c_sum_code_for_llvm, cflags = [sum_opt; sum_flags]);

# ╔═╡ a7421d94-6966-4b71-b8c2-7553b209f146
aside(md_c(c_sum_code_for_llvm), v_offset = -480)

# ╔═╡ 69bdd3ba-dbeb-4ef8-acb7-6314bee13c8c
emit_llvm(c_sum_code_for_llvm, cflags = [sum_opt; sum_flags], language = CppLanguage());

# ╔═╡ Cell order:
# ╟─49aca9a0-ed40-11ef-1cf9-635242dfa821
# ╟─4d0f2c46-4651-4ba9-b08d-44c8494d2b60
# ╟─74d55b53-c917-460a-b59c-71b1f07f7cba
# ╟─b348eb57-446b-42ec-9292-d5a77cd26e0c
# ╟─b6ae6fcc-a77e-49c5-b380-06854844469e
# ╟─74ae5855-85e8-4615-bf98-e7819bc053d2
# ╟─0d17955c-6ddc-4d57-8600-8ad3229d4631
# ╟─f6674345-4b71-40f3-8d42-82697990d534
# ╟─a38807e2-d901-4467-b35e-248da491abff
# ╠═a841d535-c32b-4bb6-8132-600253038508
# ╠═baf29a4d-337c-430c-b382-9b2dab7ce69a
# ╟─1548a494-80a9-4295-a012-88be6de7fcfa
# ╟─ec98ab34-cb2b-48c1-a9d2-3fa9c7821d11
# ╠═8b7c3a6e-bd6a-425e-8040-340fdb6b0dd0
# ╠═691d01a2-12fc-4782-a9f9-a732746285c6
# ╠═0b4c686c-912b-42ff-a7ef-970030808a74
# ╟─0a19c69e-d9f1-4630-a8b4-5718e4f1abfa
# ╟─8f4e6abd-8da8-42a5-b69f-ae76fa8fcf6b
# ╟─1e494794-7c9f-42bb-a06c-d617ee271c9b
# ╟─9956af59-12e9-4eb6-bf63-03e2936a5912
# ╟─2a404744-686c-4b8a-988a-8ff99603f2d4
# ╟─2acc14b4-4e65-4dc1-950a-df9ed3a0892d
# ╟─66a18765-b8a4-41af-8711-80d08b0ef4c4
# ╟─f853de2d-ca27-42d6-af9a-194ee6bb7d89
# ╠═e437157d-e30a-498f-a031-a603048caed0
# ╠═cce70070-5938-4f44-8181-2fb6158c419b
# ╠═ad4e2ac1-6a51-4338-ae38-15a2b817020d
# ╠═70ab5cde-5856-451d-9095-864367b6c207
# ╟─e432159e-f3f2-412d-b559-155674f732f6
# ╟─b19154d8-cb88-4aac-b76a-18f647672d70
# ╠═4cd17588-8f3c-447e-890b-fc881575db8d
# ╠═3469f9fe-2512-4fb9-81b8-dd1d39e20c38
# ╠═c80ad92b-853d-4bc1-ad7c-0dd1ad48d1c4
# ╠═570b50d9-64d8-408a-8f05-6f81716f20c2
# ╟─8df0ed24-b5bc-4cf8-b507-37bd8fc79be2
# ╠═919045cb-90cc-4cbc-be2a-5b2580a93de9
# ╟─c8ae3959-6428-4937-9212-171ea6ab0888
# ╟─52cd9d6e-0e24-45ae-a602-1b9d9edc67ae
# ╟─1a4f7389-9d1b-4008-8896-76ecc409ab1f
# ╟─839f5630-a405-4a2e-9046-cd0d1fd9c37e
# ╟─a0389720-9ed7-4534-87ac-5b61e5c2470d
# ╟─ee269b38-a5e1-467a-a91e-f7a7f1f54509
# ╟─abf284e9-75f1-42f4-b559-8720f56b02a2
# ╟─8a552e21-e51b-457d-b974-148537db6cae
# ╟─639e0ece-502b-4379-a932-32c0d119cc2f
# ╟─1ddcda8b-fa23-4802-852c-e70b1777c2e4
# ╠═3afaf82a-4843-4afa-8541-1a26d7e943a1
# ╟─3d4335d5-f526-4869-b3e7-a0b36443cc41
# ╟─c66fc30b-355d-43fa-9950-f943e3a095a6
# ╟─7a5620c8-2ca0-4422-851d-39c5b65226e5
# ╠═fc278dad-6133-466b-8c3a-775353bdd64a
# ╠═6869a1d9-b662-4c66-9adb-fc72932eb6c5
# ╟─1fa393f5-ccea-4199-bf23-16fc1d6a1969
# ╟─220127a6-dba3-448a-a12d-f9c523009f74
# ╠═e4ef3a2b-ba92-4c86-9ff2-2b968de27ea5
# ╠═765aef1b-ffd1-4851-9b15-0ad9df4980f4
# ╟─7bbb30c8-3407-4a18-aa50-8b8f6f37e8a3
# ╟─c7a4a182-6503-4d3d-9f49-8b1b2e3dc499
# ╠═9e48a50e-e120-4838-91d7-264522ac1723
# ╟─7530ea93-11fd-4931-9dd4-a5e820f8b540
# ╟─a0abb64b-6dc2-4e98-bdfd-5de9b5c97897
# ╟─a7e4be26-d088-47fe-b0ce-e12cb9936599
# ╠═3a7df4f4-0f7b-4a51-8d6a-dcba9a97c18f
# ╟─4d0ba8c4-2d94-400e-a106-467db6e3fc0c
# ╟─403bb0f1-5514-486e-9f81-fba9d6031ee1
# ╟─b9ad74c5-d99d-4129-afa2-4ff62eedf796
# ╟─e6fac999-9f54-42f9-a1b7-3fd883b891ab
# ╟─a7421d94-6966-4b71-b8c2-7553b209f146
# ╟─bc8bc245-6c10-4759-a85b-b407ef016c60
# ╟─9cfd52a7-f5b9-424a-b1a4-b81f63e3b30c
# ╟─41d1448e-72c9-431c-a614-c7922e35c883
# ╟─69bdd3ba-dbeb-4ef8-acb7-6314bee13c8c
# ╠═7ab127df-8afd-4ebe-8403-9ca3bcc2f8e3
# ╠═57005169-054b-4912-b0ba-742a56ee3f5f
# ╟─972c1194-9d5f-438a-964f-176713bab912
# ╟─1cb7d80a-84a0-41a3-b089-6ffefa44f041
# ╟─49ca9d35-cce8-45fd-8c2e-1dd92f056c93
# ╟─8e3738ac-d742-4c60-ade8-f5565ea2d1bf
# ╟─48d3e554-28f3-4ca3-a111-8a9904771426
# ╟─8c23d4b7-9580-4563-9586-1e32358b9802
# ╟─529ba439-40fe-4d93-88c5-797c0a9fc6ee
# ╠═69c872e1-966a-4a7a-a90f-d13bc108b801
# ╠═bfb3b635-85b2-4a1e-a16c-5106b6495d09
# ╟─594cb702-35ff-4932-93cb-8cdbd53b7e27
# ╟─aa153cd9-0118-4f2a-802e-fae8c302ad4b
# ╟─ea10cb8a-a95e-400c-be86-1633a3833ec5
# ╟─7dd1fa44-ed35-4abe-853f-58fe4085b441
# ╟─fcf5c210-c100-4534-a65b-9bee23c518da
# ╟─8d24ad58-fd1a-43f2-b1ce-ab02dd3a5df6
# ╟─174407b5-75be-4930-a476-7f2bfa35cdf0
# ╟─406dcb0d-b68d-40ec-8844-c5445cff88f6
# ╟─dfc87f8b-1b29-4f3a-b674-6ddf16b9d1ce
# ╟─d3c2d2b7-8f23-478b-b36b-c92552a6cf01
