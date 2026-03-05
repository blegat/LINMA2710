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

# ╔═╡ 4034621b-b836-43f6-99ec-2f7ac88cf4e3
using OpenCL, pocl_jll # `pocl_jll` provides the POCL OpenCL platform for CPU devices

# ╔═╡ 584dcbdd-cfed-4e19-9b7c-0e5256d051fa
using SimpleClang, PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools

# ╔═╡ 2861935c-c989-434b-996f-f2c99d785315
@htl("""
<p align=center style=\"font-size: 40px;\">LINMA2710 - Scientific Computing
Graphics processing unit (GPU)</p><p align=right><i>P.-A. Absil and B. Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ 6a09a11c-6ddd-4302-b371-7a947f339b52
md"""
Sources

* [OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl)
* [HandsOnOpenCL](https://github.com/HandsOnOpenCL/Lecture-Slides)
* [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
* [Parallel Computation Patterns (Reduction)](sshuttle -r manneback 10.3.221.102/16)
* [Profiling, debugging and optimization](https://indico.ijs.si/event/1183/sessions/171/attachments/1065/1362/EuroCC_Intro_to_parallel_programming_accelerators_pt-2.pdf)
* [How to use TAU for Performance Analysis](https://www.olcf.ornl.gov/wp-content/uploads/2019/08/3_tau_day_1.pdf)
"""

# ╔═╡ 093f3598-0fbc-4236-af12-d02d361bde1b
md"# Introduction"

# ╔═╡ 2b7036fe-2cd6-45bb-8124-b805b85fd0ba
md"## Context"

# ╔═╡ a3f31283-1054-4abe-9ec3-1e753905b83f
md"## General-Purpose computing on GPU (GPGPU)"

# ╔═╡ 2eba97cf-56c2-457c-b07d-1ec5678476b1
md"## Standard Portable Intermediate Representation (SPIR)"

# ╔═╡ 426b14a2-218a-4639-a36a-0188e8f8328a
md"""
Similar to LLVM IR : Intermediate representation for accelerated computation.
"""

# ╔═╡ 2cfe65d7-669f-426e-af8a-473bc5f36318
md"## Hierarchy"

# ╔═╡ 1d6d90e1-c720-49c2-9eb0-e8a3b81b32ef
md"""
| compute device    | compute unit     | processing element |
|-------------------|------------------|--------------------|
| `get_global_id`   | `get_group_id`   | `get_local_id`     |
| `get_global_size` | `get_num_groups` | `get_local_size`   |
"""

# ╔═╡ 11444947-ce05-47c2-8f84-8ed3af3d8665
md"## Memory"

# ╔═╡ 2e0ffb06-536b-402c-9ee8-8980c6f08d37
md"## OpenCL Platforms and Devices"

# ╔═╡ 269eadc2-77ea-4329-ae77-a2df4d2af8cb
md"""
* Platforms are OpenGL implementations, listed in `/etc/OpenCL/vendors`
* Devices are actual CPUs/GPUs
* ICD allows to change platform at runtime
"""

# ╔═╡ 7e29d33b-9956-4663-9985-b89923fbf1f8
OpenCL.versioninfo()

# ╔═╡ 05372b0b-f03c-4b50-99c2-51559da18137
md"See also `clinfo` command line tool and `examples/OpenCL/common/device_info.c`."

# ╔═╡ b8f2ae64-8e2a-4ac3-9635-4761077cb834
aside(tip(Foldable(md"tl;dr To refresh the list of platforms, you need to quit Julia and open a new session", md"The OpenCL ICD Loader will compute the list of available platforms once the first time it is needed and it will never recompute it again. You can indeed see [here](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/d547426c32f9af274ec1369acd1adcfd8fe0ee40/loader/linux/icd_linux.c#L234-L238) it it sets a global `initialized` variable to `true`. This means that, if you do `using pocl_jll` or install the required GPU drivers and look at the list of platforms again from the same Julia sessions, you won't see any changes! There is unfortunately no way to set this `initialized` variable back to `false` so you'll need to restart Julia and make sure you do `using pocl_jll` before using OpenCL. Fortunately, Pluto does this in the right order.")), v_offset = -400)

# ╔═╡ 7c6a4307-610b-461e-b63a-e1b10fade204
md"## Important stats"

# ╔═╡ 6e8e7d28-f788-4fd7-80f9-1594d0502ad0
aside((@bind info_platform Select([p => p.name for p in cl.platforms()])), v_offset = -300)

# ╔═╡ 0e932c41-691c-4a0a-b2e7-d2e2972de5b8
aside((@bind info_device Select([d => d.name for d in cl.devices(info_platform)])), v_offset = -300)

# ╔═╡ c7ba2764-0921-4426-96be-6d7cf323684b
function get_scalar(prop, typ)
    scalar = Ref{typ}()
    cl.clGetDeviceInfo(info_device, prop, sizeof(typ), scalar, C_NULL)
    return Int(scalar[])
end;

# ╔═╡ ff473748-ed4a-4cef-9681-10ba978a3525
md"""
* Platform
  - name: $(info_platform.name)
  - profile: $(info_platform.profile)
  - vendor: $(info_platform.vendor)
  - version: $(info_platform.version)
* Device
  - name: $(info_device.name)
  - type: $(info_device.device_type)

  | [`clGetDeviceInfo`](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html) | Value |
  | ---- | ---- |
  | `CL_DEVICE_GLOBAL_MEM_SIZE` | $(BenchmarkTools.prettymemory(info_device.global_mem_size)) |
  | `CL_DEVICE_MAX_COMPUTE_UNITS`   | $(info_device.max_compute_units) |
  | `CL_DEVICE_LOCAL_MEM_SIZE` | $(BenchmarkTools.prettymemory(info_device.local_mem_size)) |
  | `CL_DEVICE_MAX_WORK_GROUP_SIZE` | $(info_device.max_work_group_size) |
  | `CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF` | $(get_scalar(cl.CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl.cl_uint)) |
  | `CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT` | $(get_scalar(cl.CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl.cl_uint)) |
  | `CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE` | $(get_scalar(cl.CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl.cl_uint)) |
  | `CL_DEVICE_MAX_CLOCK_FREQUENCY` | $(info_device.max_clock_frequency) MHz |
  | `CL_DEVICE_PROFILING_TIMER_RESOLUTION` | $(BenchmarkTools.prettytime(info_device.profiling_timer_resolution)) |
"""

# ╔═╡ 5932765a-f69c-4281-80a0-dab181492b98
md"# Examples"

# ╔═╡ 7f24b243-c4d0-4ff7-9289-74eafcd6b617
md"## Vectorized sum"

# ╔═╡ c9832cda-cb4a-4ffd-b093-ea440e85de20
hbox([
	md"""`vadd_size` = $(@bind vadd_size Slider(2 .^ (4:16), default = 512, show_value = true))""",
	Div(html"  ", style = Dict("flex-grow" => "1")),
	md"""`vadd_verbose` = $(@bind vadd_verbose Slider(0:16, default = 0, show_value = true))""",
])

# ╔═╡ 4c6dce77-890a-4cf2-a7e1-f5ac2507f679
aside((@bind vadd_platform Select([p => p.name for p in cl.platforms()])), v_offset = -250)

# ╔═╡ 74ada0d5-8f5e-4958-a012-2ce507778b32
aside((@bind vadd_device Select([d => d.name for d in cl.devices(vadd_platform)])), v_offset = -250)

# ╔═╡ ee9ca02c-d431-4194-ba96-67a855d0f7b1
md"## Mandelbrot"

# ╔═╡ 3e0f2c68-c766-4277-8e3b-8ada91050aa3
hbox([
	md"""`mandel_size` = $(@bind mandel_size Slider(2 .^ (4:16), default = 512, show_value = true))""",
	Div(html" "; style = Dict("flex-grow" => "1")),
	md"""`maxiter` = $(@bind maxiter Slider(1:200, default = 100, show_value = true))""",
])

# ╔═╡ c902f1de-5659-4518-b3ac-534844e9a93c
q = [ComplexF32(r,i) for i=1:-(2.0/mandel_size):-1, r=-1.5:(3.0/mandel_size):0.5];

# ╔═╡ 5cb87ab9-5ce8-4ca7-9779-f9092fef31b2
aside((@bind mandel_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ c034c5e1-ff03-4e8d-a519-cda42e52d59f
aside((@bind mandel_device Select([d => d.name for d in cl.devices(mandel_platform)])), v_offset = -400)

# ╔═╡ 322b070d-4a1e-4e8b-80fe-85b1f69c451e
md"## Compute π"

# ╔═╡ c3db554a-a910-404d-b54c-5d24c20b9800
aside((@bind π_platform Select([p => p.name for p in cl.platforms()])), v_offset = -200)

# ╔═╡ 4eee8256-c989-47f4-94b8-9ad1b3f89357
aside((@bind π_device Select([d => d.name for d in cl.devices(π_platform)])), v_offset = -200)

# ╔═╡ 948a2fe6-1dfc-4d8a-a754-cff40756fe9d
md"## First element"

# ╔═╡ 964e125c-5d09-49c0-bd24-1c25568eb661
md"Let's write a simple kernel that returns the first element of a vector in global memory."

# ╔═╡ 8d5446c4-d283-4774-833f-338b5361fa7e
aside((@bind first_el_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ ec2536b2-7198-4986-acd2-8ffd300a9ace
aside((@bind first_el_device Select([d => d.name for d in cl.devices(first_el_platform)])), v_offset = -400)

# ╔═╡ 124c0aa7-7e82-461e-a000-a47f387ddfd4
aside(md"`first_el_len` = $(@bind first_el_len Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ c61c2407-b9c7-4eb6-a056-54b69ec01540
md"## Copy to local memory"

# ╔═╡ 19869f7f-cc98-45d5-aec4-64faa40e5ede
aside((@bind copy_to_local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -300)

# ╔═╡ e5232de1-fb2f-492e-bee2-1911a662eabe
aside((@bind copy_to_local_device Select([d => d.name for d in cl.devices(copy_to_local_platform)])), v_offset = -300)

# ╔═╡ 7fcce948-ccd0-4276-bb8f-f4fd27fbf1e8
aside(md"`copy_global_len` = $(@bind copy_global_len Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -300)

# ╔═╡ 154a0565-13ad-4fe1-8f3e-9c8c0ed83ca4
aside(md"`copy_local_len` = $(@bind copy_local_len Slider((2).^(1:min(8, round(Int, log2(copy_global_len)))), default = min(256, copy_global_len), show_value = true))", v_offset = -300)

# ╔═╡ 8181ffb4-57db-494f-b749-dd937608800b
md"# Reduction on GPU"

# ╔═╡ b13fdb24-1593-438a-a282-600750a5731c
md"""
Many operations can be framed in terms of a [MapReduce](https://en.wikipedia.org/wiki/MapReduce) operation.
* Given a vector of data
* It first map each elements through a given function
* It then reduces the results into a single element

The mapping part is easily embarassingly parallel but the reduction is harder to parallelize. Let's see how this reduction step can be achieved using arguably the simplest example of `mapreduce`, the sum (corresponding to an identity map and a reduction with `+`).
"""

# ╔═╡ ed441d0c-7f33-4c61-846c-a60195a77f97
md"## Sum"

# ╔═╡ 15418031-5e3d-419a-aa92-8f2b69593c69
aside((@bind local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 5a9e881e-479c-4b5a-af0a-8f543bf981f3
aside((@bind local_device Select([d => d.name for d in cl.devices(local_platform)])), v_offset = -400)

# ╔═╡ 4293e21c-ffd1-4bf8-8797-23b0dec5a0c3
aside(md"`global_len` = $(@bind global_len Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ 15bd7314-9ce8-4042-aea8-1c6a736d12a7
aside(md"`local_len` = $(@bind local_len Slider((2).^(1:min(8, round(Int, log2(global_len)))), default = min(256, global_len), show_value = true))", v_offset = -400)

# ╔═╡ d2de3aca-47e3-48be-8e37-5dd55338b4ce
md"## Blocked sum"

# ╔═╡ b275155c-c876-4ec0-b2e4-2c87f248562f
Foldable(
	md"Was it beneficial in terms of performance for GPUs like in the case of OpenMP ?",
	md"""
No, GPU threads are much cheaper and easier to synchronize than threads running on different CPU cores like we had in the OpenMP lecture.
""",
)

# ╔═╡ 901cb94a-1cf1-4193-805c-b04d4feb51d2
aside((@bind block_local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 1aa810e8-6017-4ed8-af33-5ea58f9393f3
aside((@bind block_local_device Select([d => d.name for d in cl.devices(block_local_platform)])), v_offset = -400)

# ╔═╡ 93453907-4072-4ae9-9fb9-38c859bd21a3
aside(md"`block_global_len` = $(@bind block_global_len Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ 11fb0663-b61d-41fc-9688-b31ff283df23
aside(md"`block_local_len` = $(@bind block_local_len Slider((2).^(1:min(8, round(Int, log2(block_global_len)))), default = min(256, block_global_len), show_value = true))", v_offset = -400)

# ╔═╡ 328db68d-aa1e-456b-9fed-65c4527e7f37
aside(md"`factor` = $(@bind factor Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ 8e9911a9-337e-49ab-a6ef-5cbffea8b227
md"## Back to SIMD"

# ╔═╡ 9ed8f1ba-8c9b-4d9d-b73c-66b327dc13a5
md"""
* Also called Single Instruction Multiple Threads (SIMT)
* [CUDA Warp](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) : width of 32 threads
* AMD wavefront : width of 64 threads
* In general : [`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelWorkGroupInfo.html)
* Consecutive `get_local_id()` starting from 0
  - So the thread of local id from 0 to 31 are in the same CUDA warp.
* Threads execute the **same instruction** at the same time so no need for `barrier`.
"""

# ╔═╡ d8943644-3795-4761-8021-8dafe7c358a9
md"## Warp divergence"

# ╔═╡ fca83c6f-bb3b-4b30-9050-fc365be9f3ec
md"Suppose a kernel is executed on a nvidia GPU with `global_size` threads. How much time will it take to execute it ?"

# ╔═╡ c9f81594-93f9-431d-812e-c30d51c74002
Foldable(
	md"How much time will it take to execute it if `global_size` is 32 and `n` is 16 ?",
	md"""
There are 32 threads so they are on the same warp. However, they are not executing the same instructions so `do_task_A` and `do_task_B` cannot be run in parallel. The total time will then be `a + b ns`.
"""
)

# ╔═╡ 8bc85b9b-a74e-4c6a-a8e1-0cfc57856ab5
Foldable(
	md"How much time will it take to execute it if `global_size` is 64 and `n` is 32 ?",
	md"""
There are 64 threads so they are on two different warps. All the threads of the first warp satisfy `item < 32` so they all execute `do_task_A`. All the threads of the second one go to the `else` clause so they all execute `do_task_B`. So all the threads in each task execute the same instructions and the two warps can execute in parallel. The total time will then be `max(a, b) ns`.
"""
)

# ╔═╡ 501b4fff-4905-4663-9cf5-d094761a27d2
md"Are the threads that are still active in the same warp for you sum example ?"

# ╔═╡ 48ae3222-cf66-487f-9d9e-09ae601d425b
md"## Warp diversion for our sum"

# ╔═╡ fff22d71-2732-4206-8bed-26dee00d6c48
Foldable(
	md"We are still using different warps until the end. Is that a good thing ?",
	md"""
No. First, we need to use a `barrier` until the end so it will not be good for performance. Furthermore, most of the threads of each warps are unused which also means that we use more warps than necessary. This is wasteful in terms of performance as these other warps could be used for something else but also for power consumption as a warp consume as much whether all its threads are used or not.
""")

# ╔═╡ 1933ee88-c6cc-460b-938b-9046e6fc8f67
md"How should we change the sum to keep the working threads on the same warp ?"

# ╔═╡ 954d5f75-e7cd-4d8c-b07a-018b1f5a8b70
md"## No warp divergence"

# ╔═╡ da8ffc39-d7ca-46da-81c8-f172c853427c
md"Now the same warp is used for all threads so we don't need `barrier` and it frees other warps to stay idle (reducing power consumption) or do other tasks."

# ╔═╡ 8a999999-0312-4d38-bdc4-2e4b569165a4
md"## Reordered local sum"

# ╔═╡ d7147373-238f-48c4-9dbd-6be3d6290fab
aside((@bind reordered_local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 42f69447-d0f0-44ca-a862-350d3c3dad73
aside((@bind reordered_local_device Select([d => d.name for d in cl.devices(reordered_local_platform)])), v_offset = -400)

# ╔═╡ fd21958c-8e56-48dd-9327-f79b51860785
aside(md"`reordered_global_size` = $(@bind reordered_global_size Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ 8f7c528e-7a1e-4f78-9e2d-39b0522d3287
aside(md"`reordered_local_size` = $(@bind reordered_local_size Slider((2).^(1:min(8, round(Int, log2(reordered_global_size)))), default = min(256, reordered_global_size), show_value = true))", v_offset = -400)

# ╔═╡ dede8676-17d8-4cb4-9673-dcb00df7c9e7
md"## SIMT sum"

# ╔═╡ 1d84c896-5208-4d50-9fdd-b82a2233f834
Foldable(md"Why don't we check any condition on `item`, aren't some thread computing data that won't be used ?", md"As a SIMT unit is the smallest unit of computation, even if only on thread is executing, it's all SIMT unit will be executing anyway. So it's better to save the evaluation of the `if` condition if all the threads are in the same SIMT unit anyway.")

# ╔═╡ e18d0f97-4339-4388-b9fb-fe58ed701845
aside((@bind simt_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 335ab0f7-91ac-407d-a5c8-2455ee38bf5a
aside((@bind simt_device Select([d => d.name for d in cl.devices(simt_platform)])), v_offset = -400)

# ╔═╡ 0ae62b2b-25e7-41e7-bafc-6c2efb4a5c27
aside(md"`simt_global_size` = $(@bind simt_global_size Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ ce06cd23-4d7d-43c9-80a4-119ac9ab2c30
aside(md"`simt_local_size` = $(@bind simt_local_size Slider((2).^(1:min(8, round(Int, log2(simt_global_size)))), default = min(256, simt_global_size), show_value = true))", v_offset = -400)

# ╔═╡ 27e92271-9c5a-42a9-a522-946d1ad5b676
aside(danger(md"POCL does not synchronize, even for `simt_len <= 8`"), v_offset = -400)

# ╔═╡ b099820a-3be5-4bc0-a0da-8564558c61dc
aside(
	Foldable(md"Why do we need `volatile` ?", md"`barrier(CLK_LOCAL_MEM_FENCE)` does two synchronizations: It first makes sure that all threads reach the barriers but it also makes sure that their register memory are synced with the local memory. In a SIMT unit, they are always at the same instruction but they may have values in a register that is not synced with the local memory. `volatile` makes sure that the local memory stays in sync."),
	v_offset = -300,
)

# ╔═╡ 0e3a4f79-9d07-46cb-8368-a693304334c1
md"## Unrolled sum"

# ╔═╡ 2ab1717e-afc2-4b1a-86e6-e64143546a94
Foldable(
	md"How to have portable code using unrolling ?",
	md"The compiler could do the unrolling automatically when compiling for a specific platform if it knows the relevant sizes at **compile time**.",
)

# ╔═╡ cf439f0a-5e14-45d6-8611-1b84e7574437
aside((@bind unrolled_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 79935f80-8c05-4849-8489-ea5c279a6e05
aside((@bind unrolled_device Select([d => d.name for d in cl.devices(unrolled_platform)])), v_offset = -400)

# ╔═╡ 8603d859-9f64-4932-a02f-32ba4258174c
aside(md"`unrolled_global_size` = $(@bind unrolled_global_size Slider((2).^(1:16), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ c99f5342-4536-4d89-828c-2f70934c4b0c
aside(md"`unrolled_local_size` = $(@bind unrolled_local_size Slider((2).^(1:min(8, round(Int, log2(unrolled_global_size)))), default = min(256, unrolled_global_size), show_value = true))", v_offset = -400)

# ╔═╡ 09f6479a-bc27-436c-a3b3-12b84e084a86
md"## Utils"

# ╔═╡ e1741ba3-cc15-4c0b-96ac-2b8621be2fa6
_pretty_time(x) = BenchmarkTools.prettytime(minimum(x))

# ╔═╡ 1c9f78c8-e71d-4bfa-a879-c1ea58d95886
md"`num_runs` = $(@bind num_runs Slider(1:100, default=1, show_value = true))"

# ╔═╡ e4f9813d-e171-4d04-870a-3802e0ee1728
function timed_clcall(kernel, args...; kws...)
	info = cl.work_group_info(kernel, cl.device())
	# See https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelWorkGroupInfo.html
	println("CL_KERNEL_WORK_GROUP_SIZE                    | ", info.size)
	println("CL_KERNEL_COMPILE_WORK_GROUP_SIZE            | ", info.compile_size)
	println("CL_KERNEL_LOCAL_MEM_SIZE                     | ", BenchmarkTools.prettymemory(info.local_mem_size))
	println("CL_KERNEL_PRIVATE_MEM_SIZE                   | ", BenchmarkTools.prettymemory(info.private_mem_size))
	println("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE | ", info.prefered_size_multiple)

	# `:profile` sets `CL_QUEUE_PROFILING_ENABLE` to the command queue
	queued_submit = Float64[]
	submit_start = Float64[]
	start_end = Float64[]
	cl.queue!(:profile) do
		for _ in 1:num_runs
        	evt = clcall(kernel, args...; kws...)
        	wait(evt)
	
			# See https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html
			push!(queued_submit, evt.profile_submit - evt.profile_queued)
			push!(submit_start, evt.profile_start - evt.profile_submit)
			push!(start_end, evt.profile_end - evt.profile_start)
		end
	end
	println("Send command from host to device  | $(_pretty_time(queued_submit))")
	println("Including data transfer           | $(_pretty_time(submit_start))")
    println("Execution of kernel               | $(_pretty_time(start_end))")
end

# ╔═╡ 3ce993a9-8354-47a5-8c63-ff0b0b70caa5
import Random, PlutoPlotly, PlotlyBase

# ╔═╡ 9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
function local_sum(global_size, local_size, code, device)
	cl.device!(device)
	T = Float32
	Random.seed!(0)
	x = rand(T, global_size)
    global_x = CLArray(x)
	local_x = cl.LocalMem(T, local_size)
    result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = code.code) |> cl.build!
    k = cl.Kernel(prg, "sum")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}, CLPtr{T}}, global_x, local_x, result; global_size, local_size)

    return (; OpenCL = Array(result)[], Classical = sum(x))
end

# ╔═╡ c5a0a524-64cd-40fa-9d26-91e744083286
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

# ╔═╡ d235c08c-5508-4da1-9863-dcc75775b28d
hbox([
	md"""
* Most *dedicated* GPUs produced by $(img("https://upload.wikimedia.org/wikipedia/commons/a/a4/NVIDIA_logo.svg", :height => "15pt")) and $(img("https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg", :height => "15pt"))
* *Integrated* GPUs by $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "15pt")) used in laptops to reduce power consumption
* Designed for 3D rendering through ones of the APIs : $(img("https://upload.wikimedia.org/wikipedia/commons/7/7f/Microsoft-DirectX-Logo-wordmark.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/21/OpenGL_logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/25/WebGL_Logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/2f/WebGPU_logo.svg", :height => "25pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/f/fe/Vulkan_logo.svg", :height => "20pt")) or Apple's Metal $(img("https://upload.wikimedia.org/wikipedia/commons/8/8d/Metal_3_Logo.png", :height => "20pt"))
* Illustration on the right is from [Charge's film](https://studio.blender.org/blog/charge-poster/?utm_medium=homepage), it shows how 3D modeling works.
""",
	img("https://upload.wikimedia.org/wikipedia/commons/f/fd/Charge-movie_poster.jpg", :width => "120"),
])

# ╔═╡ ed8768e0-4b3c-4a13-8533-2219cbd1d1a1
hbox([
	md"""
Also known as *compute shader* as they abuse the programmable shading of GPUs by treating the data as texture maps.
""",
	img("https://upload.wikimedia.org/wikipedia/commons/3/3d/Phong-shading-sample_%28cropped%29.jpg", :height => "100"),
])

# ╔═╡ 277130d7-1e4f-44e7-bcf5-3a70baa45f36
grid([
	md"Hardware-specific" img("https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg", :height => "70pt") img("https://upload.wikimedia.org/wikipedia/commons/7/7b/ROCm_logo.png", :height => "60pt") md"""` ` $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "30pt")) $(img("https://upload.wikimedia.org/wikipedia/en/f/fa/OneAPI-rgb-3000.png", :height => "60pt"))"""
	md"Common interface" img("https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenCL_logo.svg") img("https://upload.wikimedia.org/wikipedia/commons/1/12/SYCL_logo.svg") img("https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/designworks/blog1/OpenACC-logo.png", :height => "50pt")
])

# ╔═╡ a48fb960-3a78-435f-9167-78d831667252
img("https://www.khronos.org/assets/uploads/apis/2024-spirv-language-ecosystem.jpg")

# ╔═╡ adf26494-b700-4867-8c74-f8d520bbd29d
hbox([
	md"""
* CPUs:
   - All CPUs part of same device
   - 1 Compute Unit per core
   - Number of processing elements equal to SIMD width
* GPUs:
   - One device per GPU
""",
	img("https://upload.wikimedia.org/wikipedia/de/9/96/Platform_architecture_2009-11-08.svg", :width => "400pt"),
])

# ╔═╡ f161cf4d-f516-4db8-a54f-c757f50d4d83
img("https://upload.wikimedia.org/wikipedia/de/d/d1/OpenCL_Memory_model.svg")

# ╔═╡ d05b1a35-c49e-4e62-a166-a64f4b480290
begin
struct CLCode <: Code
    code::String
end

macro cl_str(s)
    return :($CLCode($(esc(s))))
end

SimpleClang.source_extension(::CLCode) = "cl"
end

# ╔═╡ 4d7e430d-8106-4f69-8c36-608754f223e5
cl"""
__kernel void diverge(n)
{
  int item = get_local_id(0);
  if (item < n) {
    do_task_A(); // `a` ns
  } else {
    do_task_B(); // `b` ns
  }
}
"""

# ╔═╡ d59192da-22c6-4e0f-8e0a-8d9dc70e1ffa
begin
struct Example
    name::String
end

function code(example::Example)
    code = read(joinpath(dirname(@__DIR__), "examples", example.name), String)
    ext = split(example.name, '.')[end]
    if ext == "c"
        return CCode(code)
    elseif ext == "cpp" || ext == "cc"
        return CppCode(code)
    elseif ext == "cl"
        return CLCode(code)
    else
        error("Unrecognized extension `$ext`.")
    end
end

function SimpleClang.compile_and_run(example::Example; kws...)
    return SimpleClang.compile_and_run(code(example); kws...)
end

function SimpleClang.compile_lib(example::Example; kws...)
    return SimpleClang.compile_lib(code(example); kws...)
end
end

# ╔═╡ e176f74e-b1c7-42fd-b150-966ef2c59835
vadd_source = code(Example("OpenCL/vadd/vadd.cl"));

# ╔═╡ e1435446-a7ea-4a51-b7cd-60a526f3b0ef
codesnippet(vadd_source)

# ╔═╡ 8bcfca40-b4b6-4ef6-94a9-dbdba8b6ca7b
function vadd(len, verbose)
	a = round.(rand(Float32, len) * 100)
	b = round.(rand(Float32, len) * 100)
	c = similar(a)

	cl.device!(vadd_device)
	vadd_kernel = cl.Kernel(cl.Program(; source = vadd_source.code) |> cl.build!, "vadd")

	d_a = CLArray(a)
	d_b = CLArray(b)
	d_c = CLArray(c)

	timed_clcall(vadd_kernel, Tuple{CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, Cint},
       d_a, d_b, d_c, verbose; global_size=(len,))
	return
end

# ╔═╡ 48943ec0-f596-4e82-a161-5062a2852a1d
evt = vadd(vadd_size, vadd_verbose);

# ╔═╡ 0c3de497-aa34-441c-9e8d-8007809c05e4
mandel_source = code(Example("OpenCL/mandelbrot/mandel.cl"));

# ╔═╡ b4bb6be6-fbe9-4500-8c0e-d5adbbcda20e
codesnippet(mandel_source)

# ╔═╡ 3f0383c1-f5e7-4f84-8b86-f5823c37e5eb
function mandel(q::Array{ComplexF32}, maxiter::Int64, device; kws...)
	cl.device!(device)
    q = CLArray(q)
    o = CLArray{Cushort}(undef, size(q))

    prg = cl.Program(; source = mandel_source.code) |> cl.build!
    k = cl.Kernel(prg, "mandelbrot")

    timed_clcall(k, Tuple{CLPtr{ComplexF32}, CLPtr{Cushort}, Cushort},
           q, o, maxiter; kws...)

    return Array(o)
end

# ╔═╡ 02a4d1b9-b8ec-4fd5-84fa-4cf67d947419
mandel_image = mandel(q, maxiter, mandel_device; global_size=length(q));

# ╔═╡ 81e9d99a-c6ce-48ff-9caa-9b1869b36c2a
aside(PlutoPlotly.plot(PlotlyBase.heatmap(z=mandel_image; showscale=false), PlotlyBase.Layout(; height=300)), v_offset = -400)

# ╔═╡ 1fc9096b-52f9-4a4b-a3aa-388fd1e427dc
π_code = code(Example("OpenCL/pi/pi_ocl.cl"));

# ╔═╡ b525aeff-5d9f-49bf-b948-dc8de3f23c5d
Foldable(md"How to compute π with a kernel ?", codesnippet(π_code))

# ╔═╡ 64359922-c9ce-48a3-9f93-1626251e3d2d
function mypi(; niters = 262144, in_nsteps = 512*512*512)
	cl.device!(π_device)

    prg = cl.Program(; source = π_code.code) |> cl.build!
    pi_kernel = cl.Kernel(prg, "pi")
	work_group_size = cl.device().max_work_group_size
	nwork_groups = in_nsteps ÷ (work_group_size * niters)
	nsteps = work_group_size * niters * nwork_groups

	nwork_groups = in_nsteps ÷ (work_group_size * niters)

	if nwork_groups < 1
    	# you can get opencl object info through the getproperty syntax
    	nwork_groups = cl.device().max_compute_units
    	work_group_size = in_nsteps ÷ (nwork_groups * niters)
	end

	nsteps = work_group_size * niters * nwork_groups

	
	step_size = 1.0 / nsteps

	global_size = (nwork_groups * work_group_size,)
	local_size  = (work_group_size,)
	localmem    = cl.LocalMem(Float32, work_group_size)

	h_psum = Vector{Float32}(undef, nwork_groups)
	d_partial_sums = CLArray{Float32}(undef, length(h_psum))
    timed_clcall(pi_kernel, Tuple{Int32, Float32, cl.LocalMem{Float32}, CLPtr{Float32}},
    niters, step_size, localmem, d_partial_sums; global_size, local_size)
	cl.copy!(h_psum, d_partial_sums)

	return sum(h_psum) * step_size
end

# ╔═╡ 6144d563-10c6-449b-a20e-92c2b11da4e6
mypi()

# ╔═╡ a88cc545-7780-47b2-9eb8-a5a39d5d8f0e
first_el_code = code(Example("OpenCL/sum/first_el.cl"));

# ╔═╡ e6e73837-a474-45ce-8fee-14479a0ebe7f
codesnippet(first_el_code)

# ╔═╡ 38f12061-5308-4e57-9064-af19f97e1bae
function first_el(x::Vector{T}) where {T}
	cl.device!(first_el_device)
	result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = first_el_code.code) |> cl.build!
    k = cl.Kernel(prg, "first_el")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}}, CLArray(x), result; global_size=length(x))

	return Array(result)[]
end

# ╔═╡ df052a84-5065-40fe-9828-dfa60b936482
first_el(rand(Float32, first_el_len))

# ╔═╡ fb8cfe2a-7e0d-4258-bd4a-ae7193dacfdd
copy_to_local_code = code(Example("OpenCL/sum/copy_to_local.cl"));

# ╔═╡ 236e17f9-5c4a-471d-97d0-e8e57abb6c10
codesnippet(copy_to_local_code)

# ╔═╡ c8794d8d-c3da-4761-8360-e8e8b71d1b06
function copy_to_local(global_size, local_size)
	cl.device!(first_el_device)
	T = Float32
	x = rand(T, global_size)
	local_x = cl.LocalMem(T, local_size)

    prg = cl.Program(; source = copy_to_local_code.code) |> cl.build!
    k = cl.Kernel(prg, "copy_to_local")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}}, CLArray(x), local_x; global_size, local_size)
end

# ╔═╡ 8f88521c-4793-4b50-8bbc-8d799336a5ec
copy_to_local(copy_global_len, copy_local_len)

# ╔═╡ cefe3234-28ef-4591-87ad-a4b3468610d7
local_code = code(Example("OpenCL/sum/local_sum.cl"));

# ╔═╡ a8f39218-e414-4d0e-a577-5d2a01b13c0c
local_sum(global_len, local_len, local_code, local_device)

# ╔═╡ 9195adff-cc5d-4504-9a31-ba19b18639a0
Foldable(
	md"How to compute the sum an array in **local** memory with a kernel ?",
	codesnippet(local_code),
)

# ╔═╡ d1c5c1e6-ab41-45b7-9983-e36a444105ee
block_local_sum_code = code(Example("OpenCL/sum/block_local_sum.cl"));

# ╔═╡ 040af2e8-fc93-40e6-a0f1-70c96d864609
Foldable(
	md"How to reduce the amount of `barrier` synchronizations ?",
	codesnippet(block_local_sum_code),
)

# ╔═╡ 0855eaeb-c6e4-40f9-80d2-930c960bbd3c
function block_local_sum(global_size, local_size, factor)
	cl.device!(block_local_device)
	T = Float32
	Random.seed!(0)
	x = rand(T, global_size)
    global_x = CLArray(x)
	local_x = cl.LocalMem(T, local_size)
    result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = block_local_sum_code.code) |> cl.build!
    k = cl.Kernel(prg, "sum")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}, CLPtr{T}, Cint}, global_x, local_x, result, factor; global_size, local_size)

    return (; OpenCL = Array(result)[], Classical = sum(x))
end

# ╔═╡ b151cf64-7297-44a1-ad7e-a6c9505ff7df
block_local_sum(block_global_len, block_local_len, factor)

# ╔═╡ bc42547f-5b8a-4b18-8f04-04fcd67bd61b
reordered_local_sum_code = code(Example("OpenCL/sum/reordered_local_sum.cl"));

# ╔═╡ 69bb37db-054e-48b9-9a7a-307ded792b2b
codesnippet(reordered_local_sum_code)

# ╔═╡ 1028e0b0-8357-4e92-86fc-7357114aba8e
local_sum(reordered_global_size, reordered_local_size, reordered_local_sum_code, reordered_local_device)

# ╔═╡ d97d7a7e-c4f5-4675-a40e-05289a55927c
simt_code = code(Example("OpenCL/sum/warp.cl"));

# ╔═╡ 124251a9-4052-4e4b-a0b4-1476fd19731d
codesnippet(simt_code)

# ╔═╡ c6d5d3b9-c293-4f40-8d2a-11cadf9c50e2
local_sum(simt_global_size, simt_local_size, simt_code, simt_device)

# ╔═╡ c73b5b7b-b017-4692-860f-a98ac7a3cd5e
unrolled_code = code(Example("OpenCL/sum/unrolled.cl"));

# ╔═╡ 243570f7-eaa5-4c08-8ac2-004274271e2c
Foldable(
	md"How to get even faster performance by assuming that `items` is a power of 2 smaller than 512 and that the SIMT width is 32 ?",
	codesnippet(unrolled_code),
)

# ╔═╡ a553fdca-8a38-47a7-a1c4-fafa4d8e1939
local_sum(unrolled_global_size, unrolled_local_size, unrolled_code, unrolled_device)

# ╔═╡ 46fd1e65-a256-49ac-af02-93a46a8ce20b
function CenteredBoundedBox(str)
    xbearing, ybearing, width, height, xadvance, yadvance =
        Luxor.textextents(str)
    lcorner = Luxor.Point(xbearing - width/2, ybearing + height/2)
    ocorner = Luxor.Point(lcorner.x + width, lcorner.y + height)
    return Luxor.BoundingBox(lcorner, ocorner)
end

# ╔═╡ 126d2bb5-b740-4212-8ccf-088169db5938
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

# ╔═╡ aa51b708-525e-467f-a20f-45e860c206cc
function reduce(n, good; x_scale = 30, y_scale = 160, offset = 0.1, thread_hue = "orange")
	off(a, b) = a + sign(b - a) * offset
	p(i, j) = Point((i - 2^n/2) * x_scale, (j - n/2) * y_scale)
	c(m, i, j; kws...) = boxed(m, (p(i, j)); kws...)
	a(i1, j1, i2, j2) = arrow(p(off(i1, i1 + sign(i2 - i1)), off(j1, j2)), p(off(i2, i1), off(j2, j1)))
	function ac(i1, j1, i2, j2, m)
		a(i1, j1, i2, j2)
		c(m, i2, j2)
	end
	Random.seed!(0)
	v = rand(-3:3, 2^n)
	@draw begin
		fontsize(24)
		for k in 0:n
			if good
				stride = 2^(n-k)
			else
				stride = 2^k
			end
			for i in (good ? (1:stride) : (1:stride:2^n))
				if k > 0
					a(i, k - 1, i, k-1/2)
					j = i + (good ? stride : div(stride, 2))
					a(j, k - 1, i, k-1/2)
					v[i] += v[j]
					c(string(i - 1), i, k - 1 / 2, hue = thread_hue)
					a(i, k - 1/2, i, k)
				end
				c(string(v[i]), i, k)
			end
		end
		c("Value", 2^n - 4, n - 1)
		c("Work item", 2^n - 4, n - 1/2, hue = thread_hue)
		setdash("dash")
		sethue("blue")
		line(p(32, -1), p(32, n+1), action = :stroke)
		fontsize(42)
		text("Warp", p(33, n))
	end (2^n + 1) * x_scale (n + 1/2) * y_scale
end;

# ╔═╡ 1a101883-a9ab-480b-8716-e75ac1b9db38
reduce(6, false)

# ╔═╡ 5ca18dbb-c10b-4953-a46c-41a46c0d80a3
reduce(6, true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SimpleClang = "d80a2e99-53a4-4f81-9fa2-fda2140d535e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
pocl_jll = "627d6b7a-bbe6-5189-83e7-98cc0a5aeadd"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
Luxor = "~4.4.1"
OpenCL = "~0.10.9"
PlotlyBase = "~0.8.23"
PlutoPlotly = "~0.6.5"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.79"
SimpleClang = "~0.1.0"
StaticArrays = "~1.9.17"
pocl_jll = "~7.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "8c064f985a86e9aaa26f889295d5034aed1526af"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

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

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

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

[[deps.Clang_unified_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML", "Zlib_jll", "Zstd_jll", "libLLVM_jll"]
git-tree-sha1 = "d4c7793146d5119ed42f7a16c9d0eee3b2bb188f"
uuid = "ffc816e1-ba66-5fa9-9ecc-bcc5cb19bea1"
version = "0.1.3+0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

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

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

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

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "6487601563e4a1d1dab796e88b4548bf5544209e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.4.1"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "966946d226e8b676ca6409454718accb18c34c54"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.8.2"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "9e9186b09a13b7f094f87d1a9bb266d8780e1b1c"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.0.0"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "157e2e5838984449e44af851a52fe374d56b9ada"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.13.0+0"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

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

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "fb14a863240d62fbf5922bf9f8803d7df6c62dc8"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.40"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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

[[deps.LLD_unified_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML", "Zlib_jll", "Zstd_jll", "libLLVM_jll"]
git-tree-sha1 = "3150954b8798ebe852d2fefc8806afadb3bc1026"
uuid = "fbc507ec-cd81-588a-baa9-9847e80d13e9"
version = "0.1.2+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "69e4739502b7ab5176117e97e1664ed181c35036"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.6"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
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

[[deps.OpenCL]]
deps = ["Adapt", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LinearAlgebra", "OpenCL_jll", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SPIRVIntrinsics", "SPIRV_LLVM_Backend_jll", "SPIRV_Tools_jll", "StaticArrays"]
git-tree-sha1 = "f45be8b00dee9375b1c8f16bf6c9e25c16bb9d9d"
uuid = "08131aa3-fb12-5dee-8b74-c09406e224a2"
version = "0.10.9"

[[deps.OpenCL_Headers_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8ed58a53e30fc5485c6a205eaed02ee26d5e00c8"
uuid = "a7aa756b-2b7f-562a-9e9d-e94076c5c8ee"
version = "2025.6.13+0"

[[deps.OpenCL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4980a4b2679f1b9cdd65a21ccb57d1a89c0d68b9"
uuid = "6cb37087-e8b6-5417-8430-1f242f1e46e4"
version = "2024.10.24+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

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

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

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

[[deps.SPIRVIntrinsics]]
deps = ["ExprTools", "GPUToolbox", "LLVM", "SpecialFunctions"]
git-tree-sha1 = "8a6e9e16c467f97db7a62fbef26fef151a8b4333"
uuid = "71d1d633-e7e8-4a92-83a1-de8814b09ba8"
version = "0.5.7"

    [deps.SPIRVIntrinsics.extensions]
    SPIRVIntrinsicsSIMDExt = "SIMD"

    [deps.SPIRVIntrinsics.weakdeps]
    SIMD = "fdea26ae-647d-5447-a871-4b548cad5224"

[[deps.SPIRV_LLVM_Backend_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "aa98f686cb4836afb2dd665c5217558e5786cae2"
uuid = "4376b9bf-cff8-51b6-bb48-39421dff0d0c"
version = "20.1.5+3"

[[deps.SPIRV_LLVM_Translator_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zstd_jll"]
git-tree-sha1 = "3b58c95bfdad8be11ed0c8df569904e3e1c9f648"
uuid = "4a5d46fc-d8cf-5151-a261-86b458210efb"
version = "20.1.0+6"

[[deps.SPIRV_Tools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "783c358eff59056d7e703a644f5179270c43226a"
uuid = "6ac6d60f-d740-5983-97d7-a4482c0689f4"
version = "2025.4.0+0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

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

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

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

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

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

[[deps.pocl_jll]]
deps = ["Artifacts", "Clang_unified_jll", "Hwloc_jll", "JLLWrappers", "LLD_unified_jll", "Libdl", "OpenCL_Headers_jll", "OpenCL_jll", "SPIRV_LLVM_Translator_jll", "SPIRV_Tools_jll", "Zstd_jll"]
git-tree-sha1 = "b6e907ea95787e895818714e3449b9856ac3554c"
uuid = "627d6b7a-bbe6-5189-83e7-98cc0a5aeadd"
version = "7.1.0+1"

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
# ╟─2861935c-c989-434b-996f-f2c99d785315
# ╟─6a09a11c-6ddd-4302-b371-7a947f339b52
# ╟─093f3598-0fbc-4236-af12-d02d361bde1b
# ╟─2b7036fe-2cd6-45bb-8124-b805b85fd0ba
# ╟─d235c08c-5508-4da1-9863-dcc75775b28d
# ╟─a3f31283-1054-4abe-9ec3-1e753905b83f
# ╟─ed8768e0-4b3c-4a13-8533-2219cbd1d1a1
# ╟─277130d7-1e4f-44e7-bcf5-3a70baa45f36
# ╟─2eba97cf-56c2-457c-b07d-1ec5678476b1
# ╟─426b14a2-218a-4639-a36a-0188e8f8328a
# ╟─a48fb960-3a78-435f-9167-78d831667252
# ╟─2cfe65d7-669f-426e-af8a-473bc5f36318
# ╟─adf26494-b700-4867-8c74-f8d520bbd29d
# ╟─1d6d90e1-c720-49c2-9eb0-e8a3b81b32ef
# ╟─11444947-ce05-47c2-8f84-8ed3af3d8665
# ╟─f161cf4d-f516-4db8-a54f-c757f50d4d83
# ╟─2e0ffb06-536b-402c-9ee8-8980c6f08d37
# ╟─269eadc2-77ea-4329-ae77-a2df4d2af8cb
# ╠═7e29d33b-9956-4663-9985-b89923fbf1f8
# ╟─05372b0b-f03c-4b50-99c2-51559da18137
# ╟─b8f2ae64-8e2a-4ac3-9635-4761077cb834
# ╟─7c6a4307-610b-461e-b63a-e1b10fade204
# ╟─ff473748-ed4a-4cef-9681-10ba978a3525
# ╟─6e8e7d28-f788-4fd7-80f9-1594d0502ad0
# ╟─0e932c41-691c-4a0a-b2e7-d2e2972de5b8
# ╟─c7ba2764-0921-4426-96be-6d7cf323684b
# ╟─5932765a-f69c-4281-80a0-dab181492b98
# ╟─7f24b243-c4d0-4ff7-9289-74eafcd6b617
# ╟─e1435446-a7ea-4a51-b7cd-60a526f3b0ef
# ╟─c9832cda-cb4a-4ffd-b093-ea440e85de20
# ╟─48943ec0-f596-4e82-a161-5062a2852a1d
# ╟─4c6dce77-890a-4cf2-a7e1-f5ac2507f679
# ╟─74ada0d5-8f5e-4958-a012-2ce507778b32
# ╟─e176f74e-b1c7-42fd-b150-966ef2c59835
# ╟─8bcfca40-b4b6-4ef6-94a9-dbdba8b6ca7b
# ╟─ee9ca02c-d431-4194-ba96-67a855d0f7b1
# ╟─b4bb6be6-fbe9-4500-8c0e-d5adbbcda20e
# ╟─3e0f2c68-c766-4277-8e3b-8ada91050aa3
# ╠═c902f1de-5659-4518-b3ac-534844e9a93c
# ╠═02a4d1b9-b8ec-4fd5-84fa-4cf67d947419
# ╟─5cb87ab9-5ce8-4ca7-9779-f9092fef31b2
# ╟─c034c5e1-ff03-4e8d-a519-cda42e52d59f
# ╟─81e9d99a-c6ce-48ff-9caa-9b1869b36c2a
# ╟─3f0383c1-f5e7-4f84-8b86-f5823c37e5eb
# ╟─0c3de497-aa34-441c-9e8d-8007809c05e4
# ╟─322b070d-4a1e-4e8b-80fe-85b1f69c451e
# ╠═6144d563-10c6-449b-a20e-92c2b11da4e6
# ╟─b525aeff-5d9f-49bf-b948-dc8de3f23c5d
# ╟─c3db554a-a910-404d-b54c-5d24c20b9800
# ╟─4eee8256-c989-47f4-94b8-9ad1b3f89357
# ╟─1fc9096b-52f9-4a4b-a3aa-388fd1e427dc
# ╟─64359922-c9ce-48a3-9f93-1626251e3d2d
# ╟─948a2fe6-1dfc-4d8a-a754-cff40756fe9d
# ╟─964e125c-5d09-49c0-bd24-1c25568eb661
# ╟─e6e73837-a474-45ce-8fee-14479a0ebe7f
# ╠═df052a84-5065-40fe-9828-dfa60b936482
# ╟─38f12061-5308-4e57-9064-af19f97e1bae
# ╟─8d5446c4-d283-4774-833f-338b5361fa7e
# ╟─ec2536b2-7198-4986-acd2-8ffd300a9ace
# ╟─124c0aa7-7e82-461e-a000-a47f387ddfd4
# ╟─a88cc545-7780-47b2-9eb8-a5a39d5d8f0e
# ╟─c61c2407-b9c7-4eb6-a056-54b69ec01540
# ╟─236e17f9-5c4a-471d-97d0-e8e57abb6c10
# ╠═8f88521c-4793-4b50-8bbc-8d799336a5ec
# ╟─c8794d8d-c3da-4761-8360-e8e8b71d1b06
# ╟─19869f7f-cc98-45d5-aec4-64faa40e5ede
# ╟─e5232de1-fb2f-492e-bee2-1911a662eabe
# ╟─7fcce948-ccd0-4276-bb8f-f4fd27fbf1e8
# ╟─154a0565-13ad-4fe1-8f3e-9c8c0ed83ca4
# ╟─fb8cfe2a-7e0d-4258-bd4a-ae7193dacfdd
# ╟─8181ffb4-57db-494f-b749-dd937608800b
# ╟─b13fdb24-1593-438a-a282-600750a5731c
# ╟─ed441d0c-7f33-4c61-846c-a60195a77f97
# ╠═a8f39218-e414-4d0e-a577-5d2a01b13c0c
# ╟─9195adff-cc5d-4504-9a31-ba19b18639a0
# ╟─9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
# ╟─15418031-5e3d-419a-aa92-8f2b69593c69
# ╟─5a9e881e-479c-4b5a-af0a-8f543bf981f3
# ╟─4293e21c-ffd1-4bf8-8797-23b0dec5a0c3
# ╟─15bd7314-9ce8-4042-aea8-1c6a736d12a7
# ╟─cefe3234-28ef-4591-87ad-a4b3468610d7
# ╟─d2de3aca-47e3-48be-8e37-5dd55338b4ce
# ╠═b151cf64-7297-44a1-ad7e-a6c9505ff7df
# ╟─040af2e8-fc93-40e6-a0f1-70c96d864609
# ╟─b275155c-c876-4ec0-b2e4-2c87f248562f
# ╟─0855eaeb-c6e4-40f9-80d2-930c960bbd3c
# ╟─901cb94a-1cf1-4193-805c-b04d4feb51d2
# ╟─1aa810e8-6017-4ed8-af33-5ea58f9393f3
# ╟─93453907-4072-4ae9-9fb9-38c859bd21a3
# ╟─11fb0663-b61d-41fc-9688-b31ff283df23
# ╟─328db68d-aa1e-456b-9fed-65c4527e7f37
# ╟─d1c5c1e6-ab41-45b7-9983-e36a444105ee
# ╟─8e9911a9-337e-49ab-a6ef-5cbffea8b227
# ╟─9ed8f1ba-8c9b-4d9d-b73c-66b327dc13a5
# ╟─d8943644-3795-4761-8021-8dafe7c358a9
# ╟─fca83c6f-bb3b-4b30-9050-fc365be9f3ec
# ╟─4d7e430d-8106-4f69-8c36-608754f223e5
# ╟─c9f81594-93f9-431d-812e-c30d51c74002
# ╟─8bc85b9b-a74e-4c6a-a8e1-0cfc57856ab5
# ╟─501b4fff-4905-4663-9cf5-d094761a27d2
# ╟─48ae3222-cf66-487f-9d9e-09ae601d425b
# ╟─fff22d71-2732-4206-8bed-26dee00d6c48
# ╟─1a101883-a9ab-480b-8716-e75ac1b9db38
# ╟─1933ee88-c6cc-460b-938b-9046e6fc8f67
# ╟─954d5f75-e7cd-4d8c-b07a-018b1f5a8b70
# ╟─da8ffc39-d7ca-46da-81c8-f172c853427c
# ╟─5ca18dbb-c10b-4953-a46c-41a46c0d80a3
# ╟─aa51b708-525e-467f-a20f-45e860c206cc
# ╟─8a999999-0312-4d38-bdc4-2e4b569165a4
# ╟─69bb37db-054e-48b9-9a7a-307ded792b2b
# ╟─1028e0b0-8357-4e92-86fc-7357114aba8e
# ╟─d7147373-238f-48c4-9dbd-6be3d6290fab
# ╟─42f69447-d0f0-44ca-a862-350d3c3dad73
# ╟─fd21958c-8e56-48dd-9327-f79b51860785
# ╟─8f7c528e-7a1e-4f78-9e2d-39b0522d3287
# ╟─bc42547f-5b8a-4b18-8f04-04fcd67bd61b
# ╟─dede8676-17d8-4cb4-9673-dcb00df7c9e7
# ╟─124251a9-4052-4e4b-a0b4-1476fd19731d
# ╠═c6d5d3b9-c293-4f40-8d2a-11cadf9c50e2
# ╟─1d84c896-5208-4d50-9fdd-b82a2233f834
# ╟─e18d0f97-4339-4388-b9fb-fe58ed701845
# ╟─335ab0f7-91ac-407d-a5c8-2455ee38bf5a
# ╟─0ae62b2b-25e7-41e7-bafc-6c2efb4a5c27
# ╟─ce06cd23-4d7d-43c9-80a4-119ac9ab2c30
# ╟─27e92271-9c5a-42a9-a522-946d1ad5b676
# ╟─b099820a-3be5-4bc0-a0da-8564558c61dc
# ╟─d97d7a7e-c4f5-4675-a40e-05289a55927c
# ╟─0e3a4f79-9d07-46cb-8368-a693304334c1
# ╟─243570f7-eaa5-4c08-8ac2-004274271e2c
# ╠═a553fdca-8a38-47a7-a1c4-fafa4d8e1939
# ╟─2ab1717e-afc2-4b1a-86e6-e64143546a94
# ╟─cf439f0a-5e14-45d6-8611-1b84e7574437
# ╟─79935f80-8c05-4849-8489-ea5c279a6e05
# ╟─8603d859-9f64-4932-a02f-32ba4258174c
# ╟─c99f5342-4536-4d89-828c-2f70934c4b0c
# ╟─c73b5b7b-b017-4692-860f-a98ac7a3cd5e
# ╟─09f6479a-bc27-436c-a3b3-12b84e084a86
# ╠═e1741ba3-cc15-4c0b-96ac-2b8621be2fa6
# ╠═e4f9813d-e171-4d04-870a-3802e0ee1728
# ╟─1c9f78c8-e71d-4bfa-a879-c1ea58d95886
# ╠═4034621b-b836-43f6-99ec-2f7ac88cf4e3
# ╠═584dcbdd-cfed-4e19-9b7c-0e5256d051fa
# ╠═3ce993a9-8354-47a5-8c63-ff0b0b70caa5
# ╟─c5a0a524-64cd-40fa-9d26-91e744083286
# ╟─d59192da-22c6-4e0f-8e0a-8d9dc70e1ffa
# ╟─d05b1a35-c49e-4e62-a166-a64f4b480290
# ╟─46fd1e65-a256-49ac-af02-93a46a8ce20b
# ╟─126d2bb5-b740-4212-8ccf-088169db5938
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
