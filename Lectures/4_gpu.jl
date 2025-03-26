### A Pluto.jl notebook ###
# v0.20.5

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

# ╔═╡ 7f00bb10-fe5b-11ef-0aeb-dd2bd85aac10
import Pkg

# ╔═╡ 8dcb5cf0-d579-42ba-ba4d-41c599587975
Pkg.activate(@__DIR__)

# ╔═╡ 4034621b-b836-43f6-99ec-2f7ac88cf4e3
using OpenCL, pocl_jll # `pocl_jll` provides the POCL OpenCL platform for CPU devices

# ╔═╡ 584dcbdd-cfed-4e19-9b7c-0e5256d051fa
using MyUtils, PlutoUI, PlutoUI.ExperimentalLayout, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools

# ╔═╡ 2861935c-c989-434b-996f-f2c99d785315
header("LINMA2710 - Scientific Computing
Graphics processing unit (GPU)", "P.-A. Absil and B. Legat")

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
section("Introduction")

# ╔═╡ 2b7036fe-2cd6-45bb-8124-b805b85fd0ba
frametitle("Context")

# ╔═╡ d235c08c-5508-4da1-9863-dcc75775b28d
hbox([
	md"""
* Most *dedicated* GPUs produced by $(img("https://upload.wikimedia.org/wikipedia/commons/a/a4/NVIDIA_logo.svg", :height => "15pt")) and $(img("https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg", :height => "15pt"))
* *Integrated* GPUs by $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "15pt")) used in laptops to reduce power consumption
* Designed for 3D rendering through ones of the APIs : $(img("https://upload.wikimedia.org/wikipedia/commons/7/7f/Microsoft-DirectX-Logo-wordmark.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/21/OpenGL_logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/25/WebGL_Logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/2/2f/WebGPU_logo.svg", :height => "25pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/f/fe/Vulkan_logo.svg", :height => "20pt")) or Apple's Metal $(img("https://upload.wikimedia.org/wikipedia/commons/8/8d/Metal_3_Logo.png", :height => "20pt"))
* Illustration on the right is from [Charge's film](https://studio.blender.org/blog/charge-poster/?utm_medium=homepage), it show how 3D modeling work.
""",
	img("https://ddz4ak4pa3d19.cloudfront.net/cache/d3/a3/d3a36ce594d73649a043288f18a8896b.jpg", :width => "120"),
])

# ╔═╡ a3f31283-1054-4abe-9ec3-1e753905b83f
frametitle("General-Purpose computing on GPU (GPGPU)")

# ╔═╡ ed8768e0-4b3c-4a13-8533-2219cbd1d1a1
hbox([
	md"""
Also known as *compute shader* as they abuses the programmable shading of GPUs by treating the data as texture maps.
""",
	img("https://upload.wikimedia.org/wikipedia/commons/3/3d/Phong-shading-sample_%28cropped%29.jpg", :height => "100"),
])

# ╔═╡ 277130d7-1e4f-44e7-bcf5-3a70baa45f36
grid([
	md"Hardware-specific" img("https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg", :height => "70pt") img("https://upload.wikimedia.org/wikipedia/commons/7/7b/ROCm_logo.png", :height => "60pt") md"""` ` $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "30pt")) $(img("https://upload.wikimedia.org/wikipedia/en/f/fa/OneAPI-rgb-3000.png", :height => "60pt"))"""
	md"Common interface" img("https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenCL_logo.svg") img("https://upload.wikimedia.org/wikipedia/commons/1/12/SYCL_logo.svg") img("https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/designworks/blog1/OpenACC-logo.png", :height => "50pt")
])

# ╔═╡ 2eba97cf-56c2-457c-b07d-1ec5678476b1
frametitle("Standard Portable Intermediate Representation (SPIR)")

# ╔═╡ 426b14a2-218a-4639-a36a-0188e8f8328a
md"""
Similar to LLVM IR : Intermediate representation for accelerated computation.
"""

# ╔═╡ a48fb960-3a78-435f-9167-78d831667252
img("https://www.khronos.org/assets/uploads/apis/2024-spirv-language-ecosystem.jpg")

# ╔═╡ 2cfe65d7-669f-426e-af8a-473bc5f36318
frametitle("Hierarchy")

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

# ╔═╡ 1d6d90e1-c720-49c2-9eb0-e8a3b81b32ef
md"""
| compute device    | compute unit     | processing element |
|-------------------|------------------|--------------------|
| `get_global_id`   | `get_group_id`   | `get_local_id`     |
| `get_global_size` | `get_num_groups` | `get_local_size`   |
"""

# ╔═╡ 11444947-ce05-47c2-8f84-8ed3af3d8665
frametitle("Memory")

# ╔═╡ f161cf4d-f516-4db8-a54f-c757f50d4d83
img("https://upload.wikimedia.org/wikipedia/de/d/d1/OpenCL_Memory_model.svg")

# ╔═╡ 2e0ffb06-536b-402c-9ee8-8980c6f08d37
frametitle("OpenCL Platforms and Devices")

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
frametitle("Important stats")

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
section("Examples")

# ╔═╡ 7f24b243-c4d0-4ff7-9289-74eafcd6b617
frametitle("Vectorized sum")

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

# ╔═╡ e176f74e-b1c7-42fd-b150-966ef2c59835
vadd_source = code(Example("OpenCL/vadd/vadd.cl"));

# ╔═╡ e1435446-a7ea-4a51-b7cd-60a526f3b0ef
codesnippet(vadd_source)

# ╔═╡ ee9ca02c-d431-4194-ba96-67a855d0f7b1
frametitle("Mandelbrot")

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

# ╔═╡ 0c3de497-aa34-441c-9e8d-8007809c05e4
mandel_source = code(Example("OpenCL/mandelbrot/mandel.cl"));

# ╔═╡ b4bb6be6-fbe9-4500-8c0e-d5adbbcda20e
codesnippet(mandel_source)

# ╔═╡ 322b070d-4a1e-4e8b-80fe-85b1f69c451e
frametitle("Compute π")

# ╔═╡ c3db554a-a910-404d-b54c-5d24c20b9800
aside((@bind π_platform Select([p => p.name for p in cl.platforms()])), v_offset = -200)

# ╔═╡ 4eee8256-c989-47f4-94b8-9ad1b3f89357
aside((@bind π_device Select([d => d.name for d in cl.devices(π_platform)])), v_offset = -200)

# ╔═╡ 1fc9096b-52f9-4a4b-a3aa-388fd1e427dc
π_code = code(Example("OpenCL/pi/pi_ocl.cl"));

# ╔═╡ b525aeff-5d9f-49bf-b948-dc8de3f23c5d
Foldable(md"How to compute π with a kernel ?", codesnippet(π_code))

# ╔═╡ 948a2fe6-1dfc-4d8a-a754-cff40756fe9d
frametitle("First element")

# ╔═╡ 964e125c-5d09-49c0-bd24-1c25568eb661
md"Let's write a simple kernel that returns the first element of a vector in global memory."

# ╔═╡ 8d5446c4-d283-4774-833f-338b5361fa7e
aside((@bind first_el_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ ec2536b2-7198-4986-acd2-8ffd300a9ace
aside((@bind first_el_device Select([d => d.name for d in cl.devices(first_el_platform)])), v_offset = -400)

# ╔═╡ 124c0aa7-7e82-461e-a000-a47f387ddfd4
aside(md"`first_el_len` = $(@bind first_el_len Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ a88cc545-7780-47b2-9eb8-a5a39d5d8f0e
first_el_code = code(Example("OpenCL/sum/first_el.cl"));

# ╔═╡ e6e73837-a474-45ce-8fee-14479a0ebe7f
codesnippet(first_el_code)

# ╔═╡ c61c2407-b9c7-4eb6-a056-54b69ec01540
frametitle("Copy to local memory")

# ╔═╡ 19869f7f-cc98-45d5-aec4-64faa40e5ede
aside((@bind copy_to_local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -300)

# ╔═╡ e5232de1-fb2f-492e-bee2-1911a662eabe
aside((@bind copy_to_local_device Select([d => d.name for d in cl.devices(copy_to_local_platform)])), v_offset = -300)

# ╔═╡ f1e52b53-f0c7-4f51-a09f-4284830bce40
aside(md"`copy_to_local_len` = $(@bind copy_to_local_len Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -300)

# ╔═╡ fb8cfe2a-7e0d-4258-bd4a-ae7193dacfdd
copy_to_local_code = code(Example("OpenCL/sum/copy_to_local.cl"));

# ╔═╡ 236e17f9-5c4a-471d-97d0-e8e57abb6c10
codesnippet(copy_to_local_code)

# ╔═╡ 8181ffb4-57db-494f-b749-dd937608800b
section("Reduction on GPU")

# ╔═╡ b13fdb24-1593-438a-a282-600750a5731c
md"""
Many operations can be framed in terms of a [MapReduce](https://en.wikipedia.org/wiki/MapReduce) operation.
* Given a vector of data
* It first map each elements through a given function
* It then reduces the results into a single element

The mapping part is easily embarassingly parallel but the reduction is harder to parallelize. Let's see how this reduction step can be achieved using arguably the simplest example of `mapreduce`, the sum (corresponding to an identity map and a reduction with `+`).
"""

# ╔═╡ ed441d0c-7f33-4c61-846c-a60195a77f97
frametitle("Sum")

# ╔═╡ 15418031-5e3d-419a-aa92-8f2b69593c69
aside((@bind local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 5a9e881e-479c-4b5a-af0a-8f543bf981f3
aside((@bind local_device Select([d => d.name for d in cl.devices(local_platform)])), v_offset = -400)

# ╔═╡ 15bd7314-9ce8-4042-aea8-1c6a736d12a7
aside(md"`local_len` = $(@bind local_len Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ 9cb2ba52-3602-4a01-9b47-2db2552ad4c5
vec = rand(Float32, local_len)

# ╔═╡ 162d84a4-1782-4fe0-8829-0b2f0aab1c4a
@btime sum(vec)

# ╔═╡ cefe3234-28ef-4591-87ad-a4b3468610d7
local_sum_code = code(Example("OpenCL/sum/local_sum.cl"));

# ╔═╡ 9195adff-cc5d-4504-9a31-ba19b18639a0
Foldable(
	md"How to compute the sum an array in **local** memory with a kernel ?",
	codesnippet(local_sum_code),
)

# ╔═╡ d2de3aca-47e3-48be-8e37-5dd55338b4ce
frametitle("Blocked sum")

# ╔═╡ 040af2e8-fc93-40e6-a0f1-70c96d864609
Foldable(
	md"How to reduce the amount of `barrier` synchronizations ?",
	codesnippet(local_sum_code),
)

# ╔═╡ 901cb94a-1cf1-4193-805c-b04d4feb51d2
aside((@bind block_local_platform Select([p => p.name for p in cl.platforms()])), v_offset = -400)

# ╔═╡ 1aa810e8-6017-4ed8-af33-5ea58f9393f3
aside((@bind block_local_device Select([d => d.name for d in cl.devices(block_local_platform)])), v_offset = -400)

# ╔═╡ 609e5894-db5b-48f9-bc4a-9224f40012c2
aside(md"`block_local_len` = $(@bind block_local_len Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ d945e9c5-5965-4859-9efb-0a356763ee6f
block_vec = rand(Float32, block_local_len)

# ╔═╡ 328db68d-aa1e-456b-9fed-65c4527e7f37
aside(md"`factor` = $(@bind factor Slider((2).^(1:9), default = 16, show_value = true))", v_offset = -400)

# ╔═╡ d1c5c1e6-ab41-45b7-9983-e36a444105ee
block_local_sum_code = code(Example("OpenCL/sum/block_local_sum.cl"));

# ╔═╡ 8e9911a9-337e-49ab-a6ef-5cbffea8b227
frametitle("Back to SIMD")

# ╔═╡ 9ed8f1ba-8c9b-4d9d-b73c-66b327dc13a5
md"""
* Also called Single Instruction Multiple Threads (SIMT)
* [CUDA Warp](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) : width of 32 threads
* AMD wavefront : width of 64 threads
* In general : [`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelWorkGroupInfo.html)
* Consecutive `get_local_id()` starting from 0
  - So the thread of local id from 0 to 31 are in the same CUDA warp.
* Threads execute the **same instruction** at the same time.
"""

# ╔═╡ d8943644-3795-4761-8021-8dafe7c358a9
frametitle("Warp divergence")

# ╔═╡ fca83c6f-bb3b-4b30-9050-fc365be9f3ec
md"Suppose a kernel is executed on a nvidia GPU with `global_size` threads. How much time will it take to execute it ?"

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

# ╔═╡ c9f81594-93f9-431d-812e-c30d51c74002
Foldable(
	md"How much time will it take to execute it if `global_size` is 32 and `n` is 16 ?",
	md"""
There are 32 threads so they are on the same warp. However, they are not executing the same instructions so `do_task_A` and `do_task_B` cannot be run in parallel. The total time will then be `a + b ns`.
"""
)

# ╔═╡ 8bc85b9b-a74e-4c6a-a8e1-0cfc57856ab5
Foldable(
	md"How much time will it take to execute it if `global_size` is 32 and `n` is 16 ?",
	md"""
There are 64 threads so they are on two different warps. All the threads of the first warp satisfy `item < 32` so they all execute `do_task_A`. All the threads of the second one go to the `else` clause so they all execute `do_task_B`. So all the threads in each task execute the same instructions and the two warps can execute in parallel. The total time will then be `max(a, b) ns`.
"""
)

# ╔═╡ 09f6479a-bc27-436c-a3b3-12b84e084a86
frametitle("Utils")

# ╔═╡ e1741ba3-cc15-4c0b-96ac-2b8621be2fa6
_pretty_time(x) = BenchmarkTools.prettytime(minimum(x))

# ╔═╡ 1c9f78c8-e71d-4bfa-a879-c1ea58d95886
md"`num_runs` = $(@bind num_runs Slider(1:100, default=10, show_value = true))"

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

# ╔═╡ 3f0383c1-f5e7-4f84-8b86-f5823c37e5eb
function mandel(q::Array{ComplexF32}, maxiter::Int64, device; kws...)
	cl.device!(device)
    q = CLArray(q)
    o = CLArray{Cushort}(undef, size(q))

    prg = cl.Program(; source = mandel_source.code) |> cl.build!
    k = cl.Kernel(prg, "mandelbrot")

    timed_clcall(k, Tuple{Ptr{ComplexF32}, Ptr{Cushort}, Cushort},
           q, o, maxiter; kws...)

    return Array(o)
end

# ╔═╡ 02a4d1b9-b8ec-4fd5-84fa-4cf67d947419
mandel_image = mandel(q, maxiter, mandel_device; global_size=length(q));

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
    timed_clcall(pi_kernel, Tuple{Int32, Float32, cl.LocalMem{Float32}, Ptr{Float32}},
    niters, step_size, localmem, d_partial_sums; global_size, local_size)
	cl.copy!(h_psum, d_partial_sums)

	return sum(h_psum) * step_size
end

# ╔═╡ 6144d563-10c6-449b-a20e-92c2b11da4e6
mypi()

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

# ╔═╡ c8794d8d-c3da-4761-8360-e8e8b71d1b06
function copy_to_local(x::Vector{T}) where {T}
	cl.device!(first_el_device)
	local_x = cl.LocalMem(T, length(x))

    prg = cl.Program(; source = copy_to_local_code.code) |> cl.build!
    k = cl.Kernel(prg, "copy_to_local")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}}, CLArray(x), local_x; global_size=length(x))
end

# ╔═╡ 8f88521c-4793-4b50-8bbc-8d799336a5ec
copy_to_local(rand(Float32, copy_to_local_len))

# ╔═╡ 9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
function local_sum(x::Vector{T}) where {T}
	cl.device!(local_device)
    global_x = CLArray(x)
	local_x = cl.LocalMem(T, length(global_x))
    result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = local_sum_code.code) |> cl.build!
    k = cl.Kernel(prg, "sum")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}, CLPtr{T}}, global_x, local_x, result; global_size=length(global_x))

    return Array(result)[]
end

# ╔═╡ a8f39218-e414-4d0e-a577-5d2a01b13c0c
local_sum(vec)

# ╔═╡ 0855eaeb-c6e4-40f9-80d2-930c960bbd3c
function block_local_sum(x::Vector{T}, factor) where {T}
	cl.device!(block_local_device)
    global_x = CLArray(x)
	local_x = cl.LocalMem(T, length(global_x))
    result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = block_local_sum_code.code) |> cl.build!
    k = cl.Kernel(prg, "sum")

    timed_clcall(k, Tuple{CLPtr{T}, CLPtr{T}, CLPtr{T}, Cint}, global_x, local_x, result, factor; global_size=length(global_x))

    return Array(result)[]
end

# ╔═╡ b151cf64-7297-44a1-ad7e-a6c9505ff7df
block_local_sum(block_vec, factor)

# ╔═╡ a4db4017-9ecd-4b03-9127-2c75e5d2c537
Pkg.instantiate()

# ╔═╡ 3ce993a9-8354-47a5-8c63-ff0b0b70caa5
import CairoMakie # not `using`  as `Slider` collides with PlutoUI

# ╔═╡ 81e9d99a-c6ce-48ff-9caa-9b1869b36c2a
aside(CairoMakie.image(CairoMakie.rotr90(mandel_image)), v_offset = -400)

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
# ╠═3f0383c1-f5e7-4f84-8b86-f5823c37e5eb
# ╠═0c3de497-aa34-441c-9e8d-8007809c05e4
# ╟─322b070d-4a1e-4e8b-80fe-85b1f69c451e
# ╠═6144d563-10c6-449b-a20e-92c2b11da4e6
# ╟─b525aeff-5d9f-49bf-b948-dc8de3f23c5d
# ╠═c3db554a-a910-404d-b54c-5d24c20b9800
# ╠═4eee8256-c989-47f4-94b8-9ad1b3f89357
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
# ╟─8f88521c-4793-4b50-8bbc-8d799336a5ec
# ╟─c8794d8d-c3da-4761-8360-e8e8b71d1b06
# ╟─19869f7f-cc98-45d5-aec4-64faa40e5ede
# ╟─e5232de1-fb2f-492e-bee2-1911a662eabe
# ╟─f1e52b53-f0c7-4f51-a09f-4284830bce40
# ╟─fb8cfe2a-7e0d-4258-bd4a-ae7193dacfdd
# ╟─8181ffb4-57db-494f-b749-dd937608800b
# ╟─b13fdb24-1593-438a-a282-600750a5731c
# ╟─ed441d0c-7f33-4c61-846c-a60195a77f97
# ╠═9cb2ba52-3602-4a01-9b47-2db2552ad4c5
# ╠═162d84a4-1782-4fe0-8829-0b2f0aab1c4a
# ╠═a8f39218-e414-4d0e-a577-5d2a01b13c0c
# ╟─9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
# ╟─9195adff-cc5d-4504-9a31-ba19b18639a0
# ╟─15418031-5e3d-419a-aa92-8f2b69593c69
# ╟─5a9e881e-479c-4b5a-af0a-8f543bf981f3
# ╟─15bd7314-9ce8-4042-aea8-1c6a736d12a7
# ╟─cefe3234-28ef-4591-87ad-a4b3468610d7
# ╟─d2de3aca-47e3-48be-8e37-5dd55338b4ce
# ╠═d945e9c5-5965-4859-9efb-0a356763ee6f
# ╠═b151cf64-7297-44a1-ad7e-a6c9505ff7df
# ╟─040af2e8-fc93-40e6-a0f1-70c96d864609
# ╟─0855eaeb-c6e4-40f9-80d2-930c960bbd3c
# ╟─901cb94a-1cf1-4193-805c-b04d4feb51d2
# ╟─1aa810e8-6017-4ed8-af33-5ea58f9393f3
# ╟─609e5894-db5b-48f9-bc4a-9224f40012c2
# ╟─328db68d-aa1e-456b-9fed-65c4527e7f37
# ╟─d1c5c1e6-ab41-45b7-9983-e36a444105ee
# ╟─8e9911a9-337e-49ab-a6ef-5cbffea8b227
# ╟─9ed8f1ba-8c9b-4d9d-b73c-66b327dc13a5
# ╟─d8943644-3795-4761-8021-8dafe7c358a9
# ╟─fca83c6f-bb3b-4b30-9050-fc365be9f3ec
# ╟─4d7e430d-8106-4f69-8c36-608754f223e5
# ╟─c9f81594-93f9-431d-812e-c30d51c74002
# ╟─8bc85b9b-a74e-4c6a-a8e1-0cfc57856ab5
# ╟─09f6479a-bc27-436c-a3b3-12b84e084a86
# ╠═e1741ba3-cc15-4c0b-96ac-2b8621be2fa6
# ╠═e4f9813d-e171-4d04-870a-3802e0ee1728
# ╟─1c9f78c8-e71d-4bfa-a879-c1ea58d95886
# ╟─7f00bb10-fe5b-11ef-0aeb-dd2bd85aac10
# ╟─8dcb5cf0-d579-42ba-ba4d-41c599587975
# ╟─a4db4017-9ecd-4b03-9127-2c75e5d2c537
# ╠═4034621b-b836-43f6-99ec-2f7ac88cf4e3
# ╟─584dcbdd-cfed-4e19-9b7c-0e5256d051fa
# ╟─3ce993a9-8354-47a5-8c63-ff0b0b70caa5
