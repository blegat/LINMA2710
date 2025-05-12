### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 773b4012-6425-40b1-91bf-9e5a482ff142
import Pkg

# ╔═╡ e1ec6f88-564b-419c-9bc9-8da79f952185
Pkg.activate(@__DIR__)

# ╔═╡ 4d557e55-abaa-4526-9a80-1d32fd656633
using MyUtils, PlutoUI, PlutoUI.ExperimentalLayout, Luxor, BenchmarkTools, PlutoTeachingTools, DataFrames, CSV, Downloads, Unitful

# ╔═╡ 3136760a-2be1-11f0-082c-73210e038b56
header("LINMA2710 - Scientific Computing
Power Consumption", "P.-A. Absil and B. Legat")

# ╔═╡ f1482d26-4aaf-44a9-b2cc-c672581bea36
section("Energy consumption")

# ╔═╡ 9ee51ff3-73c7-45eb-a4f2-29b191abbf3a
frametitle("Primary energy consumption")

# ╔═╡ 483e6ccc-114f-4be2-8606-0b6a35b8e513
html"""<iframe src="https://ourworldindata.org/grapher/global-energy-substitution?tab=chart" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>"""

# ╔═╡ 68b25c38-7c43-42fd-89ec-a6aede2e467a
frametitle("The emisson from the power sector not negligible")

# ╔═╡ 417c20b5-9ec3-48d7-96c6-6d86401dab86
md"[Total U.S. Greenhouse Gas Emissions by Economic Sector in 2022](https://www.epa.gov/ghgemissions/electric-power-sector-emissions)"

# ╔═╡ de0ac9cc-4664-4068-ba5f-fc4d36d479e5
img("https://www.epa.gov/system/files/styles/large/private/images/2024-04/electricity-ghg-2024-chart.png?itok=Is4AR6wp")

# ╔═╡ a23fc565-360b-4528-91eb-588a218bedb0
frametitle("Electricity generation per country")

# ╔═╡ 04e552e5-e1b4-4dc4-b6f0-634ccab49039
html"""<iframe src="https://ourworldindata.org/grapher/electricity-generation?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>"""

# ╔═╡ 362631aa-fc47-4f12-b1c4-b687575738e8
frametitle("Electricity generation per capita")

# ╔═╡ eb1bdc8e-54c4-4bb4-ac32-d3d3c5a8fb03
html"""<iframe src="https://ourworldindata.org/grapher/per-capita-electricity-generation?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>"""

# ╔═╡ 5d31609d-0b7c-41ec-ba83-67a7b1ce5e63
section("Carbon intensity")

# ╔═╡ 6ebb3eae-c7c6-469c-9710-a98d18b349dc
frametitle("Intensity per source")

# ╔═╡ e31d6a8c-942e-4a65-a073-96ef250163fb
md"""
| Energy Source Carbon | Intensity (kg/MWh) |
|----------------------|--------------------|
| Coal                 | 995                |
| Petroleum            | 816                |
| Natural Gas          | 743                |
| Solar                | 48                 |
| Geothermal           | 38                 |
| Nuclear              | 29                 |
| Hydroelectricity     | 26                 |
| Wind                 | 26                 |
"""

# ╔═╡ d76e68a9-2925-4602-86ed-5fe9d1d82ce0
aside(md"[Source](https://mlco2.github.io/codecarbon/methodology.html#carbon-intensity)", v_offset = -130)

# ╔═╡ 91a1971d-1e70-489d-9026-d9312207c112
frametitle("Share of production by sources")

# ╔═╡ af7cb311-594a-4f87-b75d-f982bf81da61
html"""<iframe src="https://ourworldindata.org/grapher/share-elec-by-source?tab=chart" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>"""

# ╔═╡ 1732b3e0-e2de-4b27-b29f-fb62ff38ba56
frametitle("Share production by group")

# ╔═╡ e01ddf29-9dd4-426c-aa44-65253dfc17bc
html"""<iframe src="https://ourworldindata.org/grapher/electricity-fossil-renewables-nuclear-line?country=~BEL&tab=chart" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>"""

# ╔═╡ e1867b63-647e-4cd3-be2b-3af5bb18f709
frametitle("Carbon Intensity per country")

# ╔═╡ 127a5424-c2ad-4157-baf3-ceacc53fcc63
html"""
<iframe src="https://ourworldindata.org/grapher/carbon-intensity-electricity?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>
"""

# ╔═╡ 86c3ecfb-cffc-492b-8f15-3105608a9203
section("Power consumption of computing")

# ╔═╡ 78220d6a-012c-400b-9a02-34772264dc31
frametitle("RAM consumption")

# ╔═╡ 2ee9c7b0-b647-4f8a-be9f-92ab421176e0
md"""
* 2-5 W per slot when the computer is on
* Drops to around 1 W when in sleep mode
* This is the reason your computer still consumes while sleeping.
* To turn the computer off, you need to loose all data in the RAM.
* Hibernation consists on copying the RAM to disk so that the RAM can be completely shut-off and copy it back when you turn it on.
"""

# ╔═╡ 1d05bb8f-466f-4b48-9c31-d92605282663
frametitle("CPU consumption : Thermal Design Power")

# ╔═╡ 1e7dc6ea-1b80-4bf2-8a33-c8719bd2c88c
md"""
Thermal Design Power (TDP) : Maximum amount of heat a CPU is designed to generate (important for cooling).
"""

# ╔═╡ 14bf7b19-2cf7-4f41-960b-de5bafa398b9
md"""
Power of CPU is approx. proportional to its utilization/load until below 10%
```math
\texttt{power} = \texttt{TDP} \times \max(0.1, \text{load})
```
[Source](https://github.com/mlco2/codecarbon/blob/018cc95937d1ffea07f03d3711d145321bab5266/codecarbon/external/hardware.py#L187-L188)
"""

# ╔═╡ e626343e-44ec-48cc-8a4a-2ada0fe05cf6
cpu = let
    f = Downloads.download("https://raw.githubusercontent.com/mlco2/codecarbon/refs/heads/master/codecarbon/data/hardware/cpu_power.csv")
	df = CSV.read(f, DataFrame)
	# It contains a weird `Intel Celeron 1000A,27.29.5` so I need
	# `tryparse` instead of `parse`
	df[!, :TDP] = tryparse.(Float64, df[:, :TDP])
	df = df[(!isnothing).(df[:, :TDP]), :]
	sort!(df, :TDP, rev = true)
	df
end;

# ╔═╡ 43d03970-8f6e-4ec5-81b4-e2d2ce332a7b
cpu

# ╔═╡ 09382a24-7e35-45ac-bfbc-46ea15a40590
frametitle("Power consumption of GPUs")

# ╔═╡ 98063156-9304-45e7-ad9c-af6dd39187b8
gpu = let
    f = Downloads.download("https://raw.githubusercontent.com/mlco2/impact/refs/heads/master/data/gpus.csv")
	df = CSV.read(f, DataFrame)
	src = "https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#Tegra_GPU"
	push!(df, ["Tesla H100-PCIE", "gpu", 350, missing, missing, missing, missing, 80, src])
	push!(df, ["Tesla H100-PCIE-80GB", "gpu", 700, missing, missing, missing, missing, 80, src])
	push!(df, ["Tesla H200-PCIE", "gpu", 600, missing, missing, missing, missing, 141, src])
	push!(df, ["Tesla H200-SXM", "gpu", 700, missing, missing, missing, missing, 141, src])
	sort!(df, :tdp_watts, rev = true)
	df
end;

# ╔═╡ 6ee903bb-66ab-496c-baf9-0c39f5dadda8
gpu

# ╔═╡ a48b8239-ff4c-4b14-8f6e-88d240bd29ac
frametitle("Power consumption of a cluster")

# ╔═╡ 64c9a463-7109-4ef0-9789-4fc885e98be9
md"""
xAI Colossus will be the largest cluster being built (not counting mega-clusters made of several ones connected by optic fibers).
Made of 100k H100 GPUs and 100k H200 GPUs so a total of 150 MW.
This will need [a whole new Gas turbine just to power it](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musks-xai-allegedly-powers-colossus-supercomputer-facility-using-illegal-generators):
"""

# ╔═╡ 04ec5e72-71b8-4a63-b73f-b1851664445c
let 
	gen = DataFrame(
		Source = ["Nuclear plant", "Gas turbine", "Wind turbine", "1000 PV panels"],
		Power = uconvert.(u"MW", [1.0u"GW", 150u"MW", 3u"MW", 320u"kW"]),
	)
	gen[:, "xAI need"] = round.(150u"MW" ./ gen[:, :Power], sigdigits=4)
	gen
end

# ╔═╡ 0a08cba4-cfb8-46a5-b85e-a826e4637af8
aside(md"[Source](https://www.energy.gov/ne/articles/infographic-how-much-power-does-nuclear-reactor-produce)", v_offset = -120)

# ╔═╡ eb328609-d0a0-4984-aba5-00ab7e90c763
section("Reducing power consumption")

# ╔═╡ 584168f4-facb-4ab8-88af-c2e6e46f981e
frametitle("Break down")

# ╔═╡ 0adbb198-25a5-42ef-8fe0-9d725b671c3a
frametitle("Dynamic voltage and frequency scaling (DVFS)")

# ╔═╡ d95d4506-0517-4460-a555-3a7767fb5c71
Foldable(md"If the clock frequency is decreased, does the time performance always get worse ?",
md"It gets worse if the program is compute-bound but not if it is bandwidth-bound (cfr. *roofline model* of part 2)",
)

# ╔═╡ 7bbf3989-cd27-4fc4-9b3c-246e2ae8f7d4
frametitle("Gating")

# ╔═╡ fc2b703d-d692-470c-a503-335426b50f1a
frametitle("Reducing the power consumption of your code")

# ╔═╡ 3fbdadde-05a3-4737-899b-3f4e352be7ee
Foldable(md"""
DVFS and gating are automatically handled. So what actions can be taken in the design of your code or GPU kernels to allow these optimizations ?
""",
md"""
* Don't use more threads than necessary. If it is bandwidth-bound, you may be using too many threads.
* Use of shared memory in kernels to reduce global memory accesses, improving both time and power efficiency.
* Decrease the number of active warps in GPU kernels (cfr. part 4).
* Decrease interleaving count (cfr. part 1) which needs high frequency but may render the program bandwidth-bound hence the gain in time efficiency is not worth it for high interleave.
"""
)

# ╔═╡ 8cbe61d0-e02b-4022-bd6b-3fcf9d0b2be6
Pkg.instantiate()

# ╔═╡ 7471cb12-234e-4b4e-83bc-8ebca7493647
import CairoMakie

# ╔═╡ ab04cd81-8dab-4dca-a77d-e1da369bce1d
let
	f = CairoMakie.Figure()
	ax = CairoMakie.Axis(f[1, 1], xlabel = "TDP [W]",)
	CairoMakie.hist!(ax, cpu[:, :TDP],
     label_formatter=x-> round(x, digits=2), label_size = 15,
     strokewidth = 0.5, strokecolor = (:black, 0.5), color = :values)
	f
end

# ╔═╡ 23007175-ffd5-4cb3-9b8c-344eb5d7cce1
let
	f = CairoMakie.Figure()
	ax = CairoMakie.Axis(f[1, 1], xlabel = "TDP [W]",)
	CairoMakie.hist!(ax, gpu[:, :tdp_watts],
     label_formatter=x-> round(x, digits=2), label_size = 15,
     strokewidth = 0.5, strokecolor = (:black, 0.5), color = :values)
	f
end

# ╔═╡ d52ac95e-5c5c-47f7-a758-3534d31dff8c
biblio = load_biblio!()

# ╔═╡ 02303063-676b-4a82-b15a-676120838a21
md"""
The power consumption of a chip is the sum of two sources:
* *Static power* : primarily due to leakage currents, which become more important as the transistor size decreases.
* *Dynamic power* : Switching power given by ``CV^2Af`` where
  - ``C`` : Capacitance being switched
  - ``V`` : Voltage
  - ``A`` : *Activity factor*, i.e., number of switches of transistors per clock cycle.
  - ``f`` : Clock frequency

The higher the voltage is, the higher are the leakage currents hence the power consumption but the voltage cannot be lowered without lowering the frequency hence the two are often done together → DVFS. Example of application in $(bibcite(biblio, "you2023Zeus")).
"""

# ╔═╡ e7ae324e-4a0d-4199-8673-6bdb23ff4ae9
md"""
The higher the voltage is, the higher are the leakage currents hence the power consumption but the voltage cannot be lowered without lowering the frequency hence the two are often done together → DVFS. Example of application in $(bibcite(biblio, "you2023Zeus")).
"""

# ╔═╡ 7ba23aa6-22a8-4474-a6fd-6949e4d6e201
bibrefs(biblio, "you2023Zeus")

# ╔═╡ 23999906-85bc-4c5f-9b42-2c9f4a63b9b8
md"""
When a core is idle, it may first be
* *clock-gated* : part of the circuit stops switching. [Corresponds to states C1-C3 in intel CPUs](https://www.intel.com/content/www/us/en/support/articles/000006619/processors/intel-core-processors.html).

If it continues being idle, it may then

* Reduces the voltage as the clock frequency is now zero. This will reduce the leakage currents. [Corresponds to state C4 in intel CPUs](https://www.intel.com/content/www/us/en/support/articles/000006619/processors/intel-core-processors.html).

If it continues being idle, it may even be

* *power-gated* : turn off circuit blocks. Which eliminates the leakage currents. It has a fixed power cost but is worth it if it is unused for a long time $(bibcite(biblio, "wang2011Power")). Corresponds to states C6-C10 in intel CPUs.

You can inspect the states of your cores on your laptop with
* Intel Power Gadget (Windows/macOS)
* [powertop](https://github.com/fenrus75/powertop) (Linux)
* [HWInfo](https://www.hwinfo.com/) or [Throttlestop](https://throttlestop.net/) (Windows)
"""

# ╔═╡ 7de807fb-0c75-4b3b-85d0-a349736448ce
bibrefs(biblio, "wang2011Power")

# ╔═╡ Cell order:
# ╟─3136760a-2be1-11f0-082c-73210e038b56
# ╟─f1482d26-4aaf-44a9-b2cc-c672581bea36
# ╟─9ee51ff3-73c7-45eb-a4f2-29b191abbf3a
# ╟─483e6ccc-114f-4be2-8606-0b6a35b8e513
# ╟─68b25c38-7c43-42fd-89ec-a6aede2e467a
# ╟─417c20b5-9ec3-48d7-96c6-6d86401dab86
# ╟─de0ac9cc-4664-4068-ba5f-fc4d36d479e5
# ╟─a23fc565-360b-4528-91eb-588a218bedb0
# ╟─04e552e5-e1b4-4dc4-b6f0-634ccab49039
# ╟─362631aa-fc47-4f12-b1c4-b687575738e8
# ╟─eb1bdc8e-54c4-4bb4-ac32-d3d3c5a8fb03
# ╟─5d31609d-0b7c-41ec-ba83-67a7b1ce5e63
# ╟─6ebb3eae-c7c6-469c-9710-a98d18b349dc
# ╟─e31d6a8c-942e-4a65-a073-96ef250163fb
# ╟─d76e68a9-2925-4602-86ed-5fe9d1d82ce0
# ╟─91a1971d-1e70-489d-9026-d9312207c112
# ╟─af7cb311-594a-4f87-b75d-f982bf81da61
# ╟─1732b3e0-e2de-4b27-b29f-fb62ff38ba56
# ╟─e01ddf29-9dd4-426c-aa44-65253dfc17bc
# ╟─e1867b63-647e-4cd3-be2b-3af5bb18f709
# ╟─127a5424-c2ad-4157-baf3-ceacc53fcc63
# ╟─86c3ecfb-cffc-492b-8f15-3105608a9203
# ╟─78220d6a-012c-400b-9a02-34772264dc31
# ╟─2ee9c7b0-b647-4f8a-be9f-92ab421176e0
# ╟─1d05bb8f-466f-4b48-9c31-d92605282663
# ╟─1e7dc6ea-1b80-4bf2-8a33-c8719bd2c88c
# ╟─14bf7b19-2cf7-4f41-960b-de5bafa398b9
# ╟─ab04cd81-8dab-4dca-a77d-e1da369bce1d
# ╟─43d03970-8f6e-4ec5-81b4-e2d2ce332a7b
# ╟─e626343e-44ec-48cc-8a4a-2ada0fe05cf6
# ╟─09382a24-7e35-45ac-bfbc-46ea15a40590
# ╟─23007175-ffd5-4cb3-9b8c-344eb5d7cce1
# ╟─6ee903bb-66ab-496c-baf9-0c39f5dadda8
# ╟─98063156-9304-45e7-ad9c-af6dd39187b8
# ╟─a48b8239-ff4c-4b14-8f6e-88d240bd29ac
# ╟─64c9a463-7109-4ef0-9789-4fc885e98be9
# ╟─04ec5e72-71b8-4a63-b73f-b1851664445c
# ╟─0a08cba4-cfb8-46a5-b85e-a826e4637af8
# ╟─eb328609-d0a0-4984-aba5-00ab7e90c763
# ╟─584168f4-facb-4ab8-88af-c2e6e46f981e
# ╟─02303063-676b-4a82-b15a-676120838a21
# ╟─0adbb198-25a5-42ef-8fe0-9d725b671c3a
# ╟─e7ae324e-4a0d-4199-8673-6bdb23ff4ae9
# ╟─d95d4506-0517-4460-a555-3a7767fb5c71
# ╟─7ba23aa6-22a8-4474-a6fd-6949e4d6e201
# ╟─7bbf3989-cd27-4fc4-9b3c-246e2ae8f7d4
# ╟─23999906-85bc-4c5f-9b42-2c9f4a63b9b8
# ╟─7de807fb-0c75-4b3b-85d0-a349736448ce
# ╟─fc2b703d-d692-470c-a503-335426b50f1a
# ╟─3fbdadde-05a3-4737-899b-3f4e352be7ee
# ╟─773b4012-6425-40b1-91bf-9e5a482ff142
# ╟─e1ec6f88-564b-419c-9bc9-8da79f952185
# ╟─8cbe61d0-e02b-4022-bd6b-3fcf9d0b2be6
# ╟─4d557e55-abaa-4526-9a80-1d32fd656633
# ╟─7471cb12-234e-4b4e-83bc-8ebca7493647
# ╟─d52ac95e-5c5c-47f7-a758-3534d31dff8c
