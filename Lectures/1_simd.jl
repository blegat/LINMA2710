### A Pluto.jl notebook ###
# v0.20.22

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

# ╔═╡ d3c2d2b7-8f23-478b-b36b-c92552a6cf01
using SimpleClang, PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools

# ╔═╡ 0ffdfb82-79d4-490f-819a-0281db822038
@htl("""
<p align=center style=\"font-size: 40px;\">LINMA2710 - Scientific Computing
Single Instruction Multiple Data (SIMD)</p><p align=right><i>P.-A. Absil and B. Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ 4d0f2c46-4651-4ba9-b08d-44c8494d2b60
md"# Motivation"

# ╔═╡ 74d55b53-c917-460a-b59c-71b1f07f7cba
md"## The need for parallelism"

# ╔═╡ b6ae6fcc-a77e-49c5-b380-06854844469e
aside(md"[Image source](https://www.karlrupp.net/2018/02/42-years-of-microprocessor-trend-data/)", v_offset = -300)

# ╔═╡ 74ae5855-85e8-4615-bf98-e7819bc053d2
md"## A bit of historical context"

# ╔═╡ f6674345-4b71-40f3-8d42-82697990d534
md"## A sum function in C and Julia"

# ╔═╡ baf29a4d-337c-430c-b382-9b2dab7ce69a
function julia_sum(v::Vector{T}) where {T}
	total = zero(T)
	for i in eachindex(v)
		total += v[i]
	end
	return total
end

# ╔═╡ ec98ab34-cb2b-48c1-a9d2-3fa9c7821d11
md"## Let's make a small benchmark"

# ╔═╡ 8b7c3a6e-bd6a-425e-8040-340fdb6b0dd0
vec_float = rand(Float32, 2^16)

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

# ╔═╡ 2a404744-686c-4b8a-988a-8ff99603f2d4
md"## Summing with SIMD"

# ╔═╡ 2acc14b4-4e65-4dc1-950a-df9ed3a0892d
Resource(
	"https://i0.wp.com/juliacomputing.com/assets/img/new/auto-vectorization2.png",
	:alt => "SIMD"
)

# ╔═╡ 66a18765-b8a4-41af-8711-80d08b0ef4c4
md"## Faster Julia code"

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
md"## Careful with fast math"

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
sum_kahan_code, sum_kahan_lib = compile_lib(c"""
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
md"# SIMD inspection"

# ╔═╡ 639e0ece-502b-4379-a932-32c0d119cc2f
md"## Instruction sets"

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
md"## SIMD at LLVM level"

# ╔═╡ 7a5620c8-2ca0-4422-851d-39c5b65226e5
md"How can you check that SIMD is enabled? Let's check at the level of LLVM IR."

# ╔═╡ fc278dad-6133-466b-8c3a-775353bdd64a
function f(x1, x2, x3, x4, y1, y2, y3, y4)
	z1 = x1 + y1
	z2 = x2 + y2
	z3 = x3 + y3
	z4 = x4 + y4
	return z1, z2, z3, z4
end

# ╔═╡ 1fa393f5-ccea-4199-bf23-16fc1d6a1969
aside(tip(md"If we see `add i64`, it means that each `Int64` is added independently"), v_offset = -200)

# ╔═╡ 220127a6-dba3-448a-a12d-f9c523009f74
md"## Packing the data to enable SIMD"

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
md"## SIMD at assembly level"

# ╔═╡ 9e48a50e-e120-4838-91d7-264522ac1723
@code_native debuginfo=:none f_broadcast((1, 2, 3, 4), (1, 2, 3, 4))

# ╔═╡ 7530ea93-11fd-4931-9dd4-a5e820f8b540
aside(tip(md"The suffix `v` in front of the instruction stands for `vectorized`. It means it is using a SIMD unit."), v_offset = -300)

# ╔═╡ a0abb64b-6dc2-4e98-bdfd-5de9b5c97897
md"## Tuples implementing the array interface"

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
md"# Auto-Vectorization"

# ╔═╡ b9ad74c5-d99d-4129-afa2-4ff62eedf796
md"## LLVM Loop Vectorizer for a C array"

# ╔═╡ 41d1448e-72c9-431c-a614-c7922e35c883
md"## LLVM Loop Vectorizer for a C++ vector"

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
	return CppCode(code)
end;

# ╔═╡ 529ba439-40fe-4d93-88c5-797c0a9fc6ee
md"## LLVM Superword-Level Parallelism (SLP) Vectorizer"

# ╔═╡ 69c872e1-966a-4a7a-a90f-d13bc108b801
f(a, b) = (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])

# ╔═╡ 6869a1d9-b662-4c66-9adb-fc72932eb6c5
@code_llvm debuginfo=:none f(1, 2, 3, 4, 5, 6, 7, 8)

# ╔═╡ bfb3b635-85b2-4a1e-a16c-5106b6495d09
@code_llvm debuginfo=:none f((1, 2, 3, 4), (5, 6, 7, 8))

# ╔═╡ 594cb702-35ff-4932-93cb-8cdbd53b7e27
frametitle("Inspection with godbolt Compiler Explorer")

# ╔═╡ aa153cd9-0118-4f2a-802e-fae8c302ad4b
html"""<iframe width="800px" height="400px" src="https://godbolt.org/e#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYMQAJlIOAMngMmABy7gBGmMQgAGykAA6oCoS2DM5uHt7xickCAUGhLBFRsZaY1ilCBEzEBGnunj6l5QKV1QR5IeGRMRZVNXUZjX3tgZ2F3dEAlBaorsTI7BwAbqh46ADU/KgQgQTrTFyk67v7PseC62GH53thZycAVACCk%2BsApADsAEJvGk/r6yebwArF8NCCACLvADMUIO6we6wg8LeXi%2Bly4kze0J%2BfwBQNBXEhMLhXgRSKYZNR6LuWJxv3%2BgJBXy8xOxcK45ORnOpGLpuMZBK%2B0LZsNOXMp7zRly8/IZnwhHGmtE4wN4ng4WlIqE4jnWClm80wUuhPFIBE0SumAGsQNDJAA6SQADi8Gkk0WiGmB0I%2Bzud0hVHEk6st2s4vAUIA05st0zgsCQaBYcTokXIlGTqfoUWQyAMRi4AE4uDGaLQCJEoxAwmGwoFqgBPThm%2BvMYiNgDyYW0ZQt3F4ybYgk7DFozc1vCwYVcwEcYloUYHpCwLEMwHEk5XeGIfbwS0wS61mFUZVclZbvF2mCDWtoeDCxCbziwl/NxDwLDfB%2BIYUSmAhTA1yMe8jDjPgDGABQADU8EwAB3Ts4kYN9%2BEEEQxHYKQZEERQVHULddEOAsTDMfQHyjSBplQOIbAEJcAFpO2hdYGLXJZVC8XhUB/D8sEoiBpiaOjPAgBwBgaHx/FGAoihAD4siSESJO8HwEiUlIOlk7oFOEiphhUrxGhvPcWmGLSuiiXSDJcepVN6NoLPGKyhMNBYJGVVVQy3HUOHWVRnWiBjokkdZ83XdZiwdLgHQ0JFcEIEgTUxXh%2By0SYbTtaEHWhaJnSkaEi2dItgQ%2BaFiuBfROBDUgv1LGMNS1XzI2jWNJ3jGBEBQVAUzTMgKAgLM%2BpQYApB8ctK2Iatay3NsmzfOaO27XtrDfIdGAIUdxzDadZ3nWhFzfVd103LV8F3coDyPXgTzPC9l2vW9eHvR9nwwRYzQID8v2XH8/yUQDgI3QJQHaiCmCg2CEKQlDlzQ4RRHEbD4bwtQw10HwSJAUxjHMF6BOo2iUkY5jWPYzjuN4tZD3gISTOaUTxNsjJDmk/JLL0dSclSZnPEOLmRKcuTDj0sy2hUkX6ZE1oaiF7oRZs9I%2BYc2WZI5zEZjmdyNaDNVSEa7jOH8wLgtC4A80ix0yQgBKiGIZLJlSuNMuhLxoskIqNCLSRvXdD4vAUoMaoN8MOBamM0qtKqOC42qJA0Bqw2atr0umH8kjsSQgA%3D%3D"></iframe>"""

# ╔═╡ ea10cb8a-a95e-400c-be86-1633a3833ec5
md"[Example source](https://llvm.org/docs/Vectorizers.html)"

# ╔═╡ 7dd1fa44-ed35-4abe-853f-58fe4085b441
md"## Further readings"

# ╔═╡ fcf5c210-c100-4534-a65b-9bee23c518da
md"""
Slides inspired from:
* [SIMD in Julia](https://www.youtube.com/watch?v=W1hXttRmuks&t=337s)
* [Demystifying Auto-vectorization in Julia](https://www.juliabloggers.com/demystifying-auto-vectorization-in-julia/)
* [Auto-Vectorization in LLVM](https://llvm.org/docs/Vectorizers.html)
"""

# ╔═╡ 9d86cb9c-396c-4357-a336-2773ee84dc2e
html"<p align=center style=\"font-size: 20px; margin-bottom: 5cm; margin-top: 5cm;\">The End</p>"

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
cpp_sum_float_code, cpp_sum_float_lib = compile_lib(cpp_sum_code("float"), lib = true, cflags = [sum_opt; sum_flags]);

# ╔═╡ 57005169-054b-4912-b0ba-742a56ee3f5f
cpp_sum(x::Vector{Cfloat}) = ccall(("c_sum", cpp_sum_float_lib), Cfloat, (Ptr{Cfloat}, Cint), x, length(x));

# ╔═╡ 7ab127df-8afd-4ebe-8403-9ca3bcc2f8e3
@btime cpp_sum($vec_float)

# ╔═╡ 8c23d4b7-9580-4563-9586-1e32358b9802
cpp_sum_code_for_llvm = cpp_sum_code(
	sum_type,
	pragmas = filter(!isequal("No pragma"), [sum_pragma_fastmath]),
	loop_pragmas = filter(!isequal("No pragma"), [sum_pragma_vectorize, sum_pragma_interleave]),
);

# ╔═╡ 972c1194-9d5f-438a-964f-176713bab912
aside(cpp_sum_code_for_llvm, v_offset = -700)

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
	return CCode(code)
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

# ╔═╡ 691d01a2-12fc-4782-a9f9-a732746285c6
@btime c_sum($vec_float)

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
aside(c_sum_code_for_llvm, v_offset = -480)

# ╔═╡ 69bdd3ba-dbeb-4ef8-acb7-6314bee13c8c
emit_llvm(c_sum_code_for_llvm, cflags = [sum_opt; sum_flags]);

# ╔═╡ 8d89bdcb-8bc8-4cff-99a2-9b2f7fccb706
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

# ╔═╡ b348eb57-446b-42ec-9292-d5a77cd26e0c
img("https://www.karlrupp.net/wp-content/uploads/2018/02/42-years-processor-trend.png")

# ╔═╡ 5d84df6f-fb95-48df-bb3b-a8a9e7adb8aa
begin
function qa(question, answer)
    return @htl("<details><summary>$question</summary>$answer</details>")
end
function _inline_html(m::Markdown.Paragraph)
    return sprint(Markdown.htmlinline, m.content)
end
function qa(question::Markdown.MD, answer)
    # `html(question)` will create `<p>` if `question.content[]` is `Markdown.Paragraph`
    # This will print the question on a new line and we don't want that:
    h = HTML(_inline_html(question.content[]))
    return qa(h, answer)
end
end

# ╔═╡ 0a19c69e-d9f1-4630-a8b4-5718e4f1abfa
qa(md"How to speed up the C code ?",
md"""
Try passing the following flags to Clang by selecting them and waiting for the benchmark timing to refresh: $(sum_float_options)

What are they doing ? We'll see in the slide...
"""
)

# ╔═╡ d4ca3ff1-5676-45d5-9c96-4f4a5d24bd3c
function Luxor.placeimage(url::URL, pos; scale = 1.0, centered = true, kws...)
    r, path = save_image(url; kws...)
    if r.mime isa MIME"image/svg+xml"
        img = Luxor.readsvg(path)
    else
        img = Luxor.readpng(path)
    end
    Luxor.gsave()
    Luxor.scale(scale)
    Luxor.placeimage(img, pos / scale; centered)
    Luxor.grestore() # undo `Luxor.scale`
    return
end

# ╔═╡ 7f04c516-316a-49b9-9141-981f943dfb80
function CenteredBoundedBox(str)
    xbearing, ybearing, width, height, xadvance, yadvance =
        Luxor.textextents(str)
    lcorner = Luxor.Point(xbearing - width/2, ybearing + height/2)
    ocorner = Luxor.Point(lcorner.x + width, lcorner.y + height)
    return Luxor.BoundingBox(lcorner, ocorner)
end

# ╔═╡ fe8d312c-81b2-489b-b170-26fcdb11ad9e
function boxed(str::AbstractString, p; hue = "lightgrey")
    Luxor.gsave()
    Luxor.translate(p)
    Luxor.sethue(hue)
    Luxor.poly(CenteredBoundedBox(str) + 5, action = :stroke, close=true)
    Luxor.sethue("black")
    Luxor.text(str, Luxor.Point(0, 0); halign = :center, valign = :middle)
    #settext("<span font='26'>$str</span>", halign="center", markup=true)
    Luxor.origin()
    Luxor.grestore() # strokecolor
end

# ╔═╡ 0d17955c-6ddc-4d57-8600-8ad3229d4631
hbox([md"""
* **1972** : C language created by Dennis Ritchie and Ken Thompson to ease development of Unix (previously developed in **assembly**)
* **1985** : C++ created by Bjarne Stroustrup
* **2003** : LLVM started at University of Illinois
* **2005** : Apple hires Chris Lattner from the university
* **2007** : He then creates the LLVM-based compiler Clang
* **2009** : Mozilla start developing an LLVM-based compiler for Rust
* **2009** : Development starts on Julia, with LLVM-based compiler
""",
	Div(md"""$(@draw begin
	    placeimage(URL("https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg"), Point(-100, -100), scale = 0.1)
	    arrow(Point(-85, -85), Point(-20, -15))
	    placeimage(URL("https://raw.githubusercontent.com/rust-lang/www.rust-lang.org/master/static/images/rust-social-wide-light.svg"), Point(0, -100), scale = 0.15)
	    arrow(Point(0, -85), Point(0, -15))
	    placeimage(URL("https://julialang.org/assets/infra/logo.svg"), Point(100, -100), scale = 0.15)
	    arrow(Point(85, -85), Point(20, -15))
		placeimage(URL("https://llvm.org/img/LLVMWyvernSmall.png"), Point(-30, 30), scale = 0.08)
		boxed("LLVM Intermediate Representation (IR)", Point(0, 0))
		arrow(Point(0, 10), Point(0, 85))
		boxed("Assembly", Point(0, 100))
	end 300 300)"""; style = Dict("width" => "100%")),
])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SimpleClang = "d80a2e99-53a4-4f81-9fa2-fda2140d535e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
Luxor = "~4.4.1"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.79"
SimpleClang = "~0.1.0"
StaticArrays = "~1.9.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "25cf5193938de9fc035716f5afbcbc45556b0627"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

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
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

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

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "80580012d4ed5a3e8b18c7cd86cebe4b816d17a6"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.9"

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

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "65ae3db6ab0e5b1b5f217043c558d9d1d33cc88d"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.0"

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
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "10c258e189b8d097c1404ed59f6c171281a39b85"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.7"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

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
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
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

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "14d1bfb0a30317edc77e11094607ace3c800f193"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.13.2"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

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
# ╟─0ffdfb82-79d4-490f-819a-0281db822038
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
# ╠═1548a494-80a9-4295-a012-88be6de7fcfa
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
# ╟─9d86cb9c-396c-4357-a336-2773ee84dc2e
# ╟─8d24ad58-fd1a-43f2-b1ce-ab02dd3a5df6
# ╟─174407b5-75be-4930-a476-7f2bfa35cdf0
# ╟─d3c2d2b7-8f23-478b-b36b-c92552a6cf01
# ╠═8d89bdcb-8bc8-4cff-99a2-9b2f7fccb706
# ╠═5d84df6f-fb95-48df-bb3b-a8a9e7adb8aa
# ╠═d4ca3ff1-5676-45d5-9c96-4f4a5d24bd3c
# ╠═7f04c516-316a-49b9-9141-981f943dfb80
# ╠═fe8d312c-81b2-489b-b170-26fcdb11ad9e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
