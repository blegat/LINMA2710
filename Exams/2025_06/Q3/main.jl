using OpenCL, pocl_jll, Plots
n = 32
a = rand(Float32, n)
d = CLArray(a)
loc = cl.LocalMem(Float32, n); # Semicolons as workaround for https://github.com/JuliaGPU/OpenCL.jl/issues/322
source = read(joinpath(@__DIR__, "ode.cl"), String)
prog = cl.Program(; source) |> cl.build!
kernel = cl.Kernel(prog, "diffuse")
p = plot(eachindex(a), Vector(d), label = "0")
for i in 1:10
    clcall(kernel, Tuple{CLPtr{Float32}, CLPtr{Float32}, Float32, Cint}, d, loc, Float32(0.001), 100; global_size=size(d))
    plot!(eachindex(a), Vector(d), label = string(i))
end
p
