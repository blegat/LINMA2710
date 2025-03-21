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

# ╔═╡ 7f00bb10-fe5b-11ef-0aeb-dd2bd85aac10
import Pkg

# ╔═╡ 8dcb5cf0-d579-42ba-ba4d-41c599587975
Pkg.activate(".")

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
"""

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
* ICD allows to change platform at runtime.
"""

# ╔═╡ 7e29d33b-9956-4663-9985-b89923fbf1f8
OpenCL.versioninfo()

# ╔═╡ 05372b0b-f03c-4b50-99c2-51559da18137
md"See also `clinfo` command line tool and `examples/OpenCL/common/device_info.c`."

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

# ╔═╡ 229994de-6ea5-47f4-9da7-72f4827ab4ba
local_sum_code = code(Example("OpenCL/sum/local_sum.cl"));

# ╔═╡ 9195adff-cc5d-4504-9a31-ba19b18639a0
Foldable(md"How to compute the sum an array in **local** memory with a kernel ?", codesnippet(local_sum_code))

# ╔═╡ 09f6479a-bc27-436c-a3b3-12b84e084a86
frametitle("Utils")

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
	cl.queue!(:profile) do
        evt = clcall(kernel, args...; kws...)
        wait(evt)
	
		# See https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html
        t = evt.profile_duration
		println("Send command from host to device  | $(BenchmarkTools.prettytime(evt.profile_submit - evt.profile_queued))")
		println("Including data transfer           | $(BenchmarkTools.prettytime(evt.profile_start - evt.profile_submit))")
        println("Execution of kernel               | $(BenchmarkTools.prettytime(evt.profile_end - evt.profile_start))")
	end
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
function mandel(q::Array{ComplexF32}, maxiter::Int64, device)
	cl.device!(device)
    q = CLArray(q)
    o = CLArray{Cushort}(undef, size(q))

    prg = cl.Program(; source = mandel_source.code) |> cl.build!
    k = cl.Kernel(prg, "mandelbrot")

    timed_clcall(k, Tuple{Ptr{ComplexF32}, Ptr{Cushort}, Cushort},
           q, o, maxiter; global_size=length(q))

    return Array(o)
end

# ╔═╡ 02a4d1b9-b8ec-4fd5-84fa-4cf67d947419
mandel_image = mandel(q, maxiter, mandel_device);

# ╔═╡ 64359922-c9ce-48a3-9f93-1626251e3d2d
function mypi(; niters = 10, in_nsteps = 10)
	cl.device!(π_device)

    prg = cl.Program(; source = π_code.code) |> cl.build!
    pi_kernel = cl.Kernel(prg, "pi")
	work_group_size = cl.device().max_work_group_size
	nwork_groups = in_nsteps ÷ (work_group_size * niters)
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

# ╔═╡ 9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
function local_sum(x::Vector{T}) where {T}
	cl.device!(local_device)
    global_x = CLArray(x)
	local_x = cl.LocalMem(T, length(global_x))
    result = CLArray(zeros(T, 1))

    prg = cl.Program(; source = local_sum_code.code) |> cl.build!
    k = cl.Kernel(prg, "local_sum")

    timed_clcall(k, Tuple{Ptr{T}, cl.LocalMem{T}, Ptr{T}}, global_x, local_x, result; global_size=length(global_x))

    return Array(result)[]
end

# ╔═╡ a8f39218-e414-4d0e-a577-5d2a01b13c0c
local_sum(vec)

# ╔═╡ a4db4017-9ecd-4b03-9127-2c75e5d2c537
Pkg.instantiate()

# ╔═╡ 3ce993a9-8354-47a5-8c63-ff0b0b70caa5
import CairoMakie # not `using`  as `Slider` collides with PlutoUI

# ╔═╡ 81e9d99a-c6ce-48ff-9caa-9b1869b36c2a
aside(CairoMakie.image(CairoMakie.rotr90(mandel_image)), v_offset = -400)

# ╔═╡ Cell order:
# ╟─2861935c-c989-434b-996f-f2c99d785315
# ╟─6a09a11c-6ddd-4302-b371-7a947f339b52
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
# ╟─7c6a4307-610b-461e-b63a-e1b10fade204
# ╟─ff473748-ed4a-4cef-9681-10ba978a3525
# ╟─6e8e7d28-f788-4fd7-80f9-1594d0502ad0
# ╟─0e932c41-691c-4a0a-b2e7-d2e2972de5b8
# ╟─c7ba2764-0921-4426-96be-6d7cf323684b
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
# ╟─0c3de497-aa34-441c-9e8d-8007809c05e4
# ╟─322b070d-4a1e-4e8b-80fe-85b1f69c451e
# ╠═6144d563-10c6-449b-a20e-92c2b11da4e6
# ╠═64359922-c9ce-48a3-9f93-1626251e3d2d
# ╟─b525aeff-5d9f-49bf-b948-dc8de3f23c5d
# ╟─c3db554a-a910-404d-b54c-5d24c20b9800
# ╟─4eee8256-c989-47f4-94b8-9ad1b3f89357
# ╟─1fc9096b-52f9-4a4b-a3aa-388fd1e427dc
# ╟─ed441d0c-7f33-4c61-846c-a60195a77f97
# ╠═9cb2ba52-3602-4a01-9b47-2db2552ad4c5
# ╠═162d84a4-1782-4fe0-8829-0b2f0aab1c4a
# ╠═a8f39218-e414-4d0e-a577-5d2a01b13c0c
# ╠═9fc9e122-a49b-4ead-b0e0-4f7a42a1123d
# ╟─9195adff-cc5d-4504-9a31-ba19b18639a0
# ╟─15418031-5e3d-419a-aa92-8f2b69593c69
# ╟─5a9e881e-479c-4b5a-af0a-8f543bf981f3
# ╟─15bd7314-9ce8-4042-aea8-1c6a736d12a7
# ╟─229994de-6ea5-47f4-9da7-72f4827ab4ba
# ╟─09f6479a-bc27-436c-a3b3-12b84e084a86
# ╠═e4f9813d-e171-4d04-870a-3802e0ee1728
# ╟─7f00bb10-fe5b-11ef-0aeb-dd2bd85aac10
# ╟─8dcb5cf0-d579-42ba-ba4d-41c599587975
# ╟─a4db4017-9ecd-4b03-9127-2c75e5d2c537
# ╠═4034621b-b836-43f6-99ec-2f7ac88cf4e3
# ╟─584dcbdd-cfed-4e19-9b7c-0e5256d051fa
# ╟─3ce993a9-8354-47a5-8c63-ff0b0b70caa5
