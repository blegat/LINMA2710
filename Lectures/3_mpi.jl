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

# ╔═╡ 82e1ea5e-f8e0-11ef-0f93-49a66050feaf
import Pkg

# ╔═╡ a55c5090-1455-4fb9-bdb9-a0b9b340b154
Pkg.activate(@__DIR__)

# ╔═╡ 8df4ff2f-d176-4b4e-a525-665b5d07ea52
using MyUtils, PlutoUI, PlutoUI.ExperimentalLayout, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools, Markdown

# ╔═╡ 58e12afd-6eb0-4731-bd57-d9ae7ab4e164
header("LINMA2710 - Scientific Computing
Distributed Computing with MPI", "P.-A. Absil and B. Legat")

# ╔═╡ 5a566137-fbd1-45b2-9a55-e4aded366bb3
section("Single Program Multiple Data (SPMD)")

# ╔═╡ a6c337c4-0c81-4463-ad4f-9a4528d953ab
frametitle("Message Passing Interface (MPI)")

# ╔═╡ c04bcc96-e5fe-4d6e-a12e-40dcde58c62e
md"""
* MPI $(img("https://avatars.githubusercontent.com/u/14836989", name = "MPI.png", :height => "20pt")) is an open standard for distributed computing
* [Many implementations](https://www.mpi-forum.org/implementation-status/):
  - MPICH, from $(img("https://upload.wikimedia.org/wikipedia/commons/6/65/ArgonneLaboratoryLogo.png", :height => "20pt")) and $(img("https://upload.wikimedia.org/wikipedia/commons/6/69/Mississippi_State_University_logo.svg", :height => "20pt"))
  - Open MPI $(img("https://upload.wikimedia.org/wikipedia/commons/6/6f/Open_MPI_logo.png", :height => "20pt")) (not to be confused with $(img("https://upload.wikimedia.org/wikipedia/commons/e/eb/OpenMP_logo.png", :width => "45pt")))
  - commercial implementations from $(img("https://upload.wikimedia.org/wikipedia/commons/4/46/Hewlett_Packard_Enterprise_logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "15pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg", :height => "15pt")), and $(img("https://upload.wikimedia.org/wikipedia/commons/9/96/NEC_logo.svg", :height => "15pt"))
"""

# ╔═╡ cf799c26-1cea-4b38-9a15-8497813bd668
frametitle("MPI basics")

# ╔═╡ 6d2b3dbc-0686-49f0-904a-56c3ce63b4dd
hbox([
	Div(md"Initializes MPI, remove `mpiexec`, etc... from `argc` and `argv`."; style = Dict("flex-grow" => "1")),
	c"""
MPI_Init(&argc, &argv)
""",
])

# ╔═╡ b5a3e471-af4a-466f-bbae-96306bcc7563
vbox([
	Div(md"Get the number of processes. `nprocs` is the **same** on all processes."; style = Dict("flex-grow" => "1")),
	c"""
int nprocs;
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
""",
])

# ╔═╡ d722a86d-6d51-4d91-ac22-53af94c91497
vbox([
	Div(md"Get the id of processes. `procid` is the **different** for **different** processes."; style = Dict("flex-grow" => "1")),
	c"""
int procid;
MPI_Comm_rank(MPI_COMM_WORLD, &procid);
""",
])

# ╔═╡ c3590376-06ed-45a4-af0b-2d46f1a387c8
hbox([
	Div(md"""
Free up memory.
"""; style = Dict("flex-grow" => "1")),
	c"""
MPI_Finalize();
""",
])

# ╔═╡ 52d428d5-cb33-4f2a-89eb-3a8ce3f5bb81
Foldable(
	md"Each process runs the **same** executable. So how can we make them do different things ?",
	md"Even if the code is the same, `MPI_Comm_rank` will give different `procid` so the part of the program depending on the value of `procid` will differ.",
)

# ╔═╡ 273ad3a6-cb32-49bb-8702-fdaf8597e812
frametitle("Different processes may be on the same node")

# ╔═╡ 4e32f7fb-cd5a-4190-9c92-ba4029313475
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/mpi-node2.png")

# ╔═╡ 82230d6c-25ce-4d12-8842-e0651fc4b143
frametitle("Processor name identifies the node")

# ╔═╡ 7d9ac5f9-39bf-4052-ad8a-ac0fec15c64a
md"""
Processes that are on the same node share the same `processor_name` (the `hostname`).
"""

# ╔═╡ a103c5af-42fe-4f8c-b78c-6946895105d7
md"`num_processes` = $(@bind procname_num_processes Slider(2:8, default = 2, show_value = true))"

# ╔═╡ b0ca0392-71b8-4f44-8c6c-0978a02a0e6c
compile_and_run(Example("procname.c"); mpi = true, verbose = 1, show_run_command = true, num_processes = procname_num_processes)

# ╔═╡ 21b6133f-db59-4885-9b3d-331c3d6ef306
frametitle("Compiling")

# ╔═╡ 35ba1eea-56ae-4b74-af96-21ec5a93c455
md"""
You could simply add `lmpi` bu using `mpicc` and `mpic++` is easier.
"""

# ╔═╡ 8981b5e2-2497-478e-ab28-a14b62f6f916
run(`mpicc -show`)

# ╔═╡ 5441e428-b320-433c-acde-15fe6bf58537
run(`mpic++ -show`)

# ╔═╡ 40606ee3-38cc-4123-9b86-b774bf89e499
section("Collectives")

# ╔═╡ b94cd399-0370-49e9-a522-056f3af22955
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/collectives.jpg")

# ╔═╡ 9b4cae31-c319-444e-98c8-2c0bfc6dfa0c
frametitle("Broadcast")

# ╔═╡ 8b83570a-6982-47e5-a167-a6d6afee0f7d
hbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x`` |   |   |   |
""",
	Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x`` | ``x`` | ``x`` | ``x`` |
""",
])

# ╔═╡ 5d72bf87-7f3a-4229-9d7a-2e63c115087d
Foldable(
	md"Lower bound complexity for ``n`` bytes with ``p`` processes ?",
	md"""Lower bound : ``\log_2(p) (\alpha + \beta n)`` using *spanning tree* algorithm:

After first communication (1 → 3):

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x`` |   | ``x``  |   |

After second communication (1 → 2 and 3 → 4 at the same time):

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x`` | ``x`` | ``x`` | ``x`` |
	"""
)

# ╔═╡ 7b1d26c6-9499-4e44-84c8-c272737a175e
frametitle("Gather")

# ╔═╡ fc43b343-79cd-4342-8d80-8ea72cf34942
hbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` |   |   |   |
|    |   | ``x_2`` |   |   |
|    |   |   | ``x_3`` |   |
|    |   |   |   | ``x_4`` |
""",
	Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` |   |   |   |
|    | ``x_2`` |   |   |   |
|    | ``x_3`` |   |   |   |
|    | ``x_4`` |   |   |   |
""",
])

# ╔═╡ 233c13ff-f008-40b0-a6c5-c5395b2215ec
Foldable(
	md"Lower bound complexity with ``p`` processes if ``x_i`` has length ``n/p`` bytes ?",
	md"""
Lower bound : ``\log_2(p) \alpha`` using *spanning tree* algorithm and ``\beta n`` as all message need to sent at least once. *spanning tree* is advantageous if ``\alpha`` is larger than ``\beta`` and direct to `1` if otherwise. In practice, you want a mix of both.

First send ``x_2`` from 2 to 1 and simultaneously send ``x_4`` from 4 to 3.
Complexity is ``\alpha + \beta n/4``

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` |   |   |   |
|    | ``x_2`` | ``x_2`` |   |   |
|    |   |   | ``x_3`` |   |
|    |   |   | ``x_4`` | ``x_4`` |

Then send ``(x_3, x_4)`` from 3 to 1.
Complexity is ``\alpha + 2\beta n/4``

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` |   |   |   |
|    | ``x_2`` | ``x_2`` |   |   |
|    | ``x_3`` |   | ``x_3`` |   |
|    | ``x_4`` |   | ``x_4`` | ``x_4`` |

It total, it is ``2\alpha + 3\beta n/4``. In general, we have
```math
\log_2(p)\alpha + \beta n(1 + 2 + 4 + \cdots + p/2)/p = \log_2(p)\alpha + \beta n(p - 1)/p \approx \log_2(p)\alpha + \beta n
```
"""
)

# ╔═╡ ad3559d1-6180-4eaa-b97d-3c1f10f036b9
frametitle("Reduce")

# ╔═╡ c420ad25-6af1-4fb4-823a-b6bbd4e10f7f
hbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` | ``x_2`` | ``x_3`` | ``x_4`` |
""",
	Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1 + x_2 + x_3 + x_4`` |  |  |  |
""",
])

# ╔═╡ db16e939-b490-497b-a03f-80ce2e8485af
Foldable(
	md"Lower bound complexity for ``n`` bytes with ``p`` processes and arithmetic complexity ``\gamma`` ?",
	md"""Lower bound : ``\log_2(p) (\alpha + \beta n) + \log_2(p) \gamma n`` using *spanning tree* algorithm:

First communication (2 → 1 and 4 → 3 at the same time):

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1 + x_2`` |   | ``x_3 + x_4``  |   |

Then second communication (3 → 1)
	"""
)

# ╔═╡ 4fdb4cd6-a794-4b14-84b0-72f484c6ea86
frametitle("All gather")

# ╔═╡ a258eec9-f4f6-49bd-8470-8541836f5f6b
hbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` |   |   |   |
|    |   | ``x_2`` |   |   |
|    |   |   | ``x_3`` |   |
|    |   |   |   | ``x_4`` |
""",
	Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After `MPI_Allgather`

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` | ``x_1`` | ``x_1`` | ``x_1`` |
|    | ``x_2`` | ``x_2`` | ``x_2`` | ``x_2`` |
|    | ``x_3`` | ``x_3`` | ``x_3`` | ``x_3`` |
|    | ``x_4`` | ``x_4`` | ``x_4`` | ``x_4`` |
""",
])

# ╔═╡ 6fc34de1-469b-41a9-9677-ff3182f7a498
Foldable(md"Can `MPI_Allgather` be implemented by combining existing collectives ?", md"`MPI_Allgather` can be implemented by `MPI_Gather` followed by `MPI_Bcast`")

# ╔═╡ de20bf96-7d33-4a78-8147-f0b7f8488e46
Foldable(
	md"""
Would it be more efficient to have a specialized implementation instead of combining existing collectives ?
""",
	md"""
Let the size of `x_i` be ``n/p``. `MPI_Gather` has complexity ``\log_2(p)\alpha + \beta n`` and `MPI_Bcast` has complexity ``\log_2(p) (\alpha + \beta n)``
so in total ``\log_2(p) (\alpha + \beta n)``. Can we do better ?

Start exchanging between 1 and 2 and simultaneously exchanging between 3 and 4.
The complexity is ``\alpha + \beta n/4``.

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` | ``x_1`` |   |   |
|    | ``x_2`` | ``x_2`` |   |   |
|    |   |   | ``x_3`` | ``x_3`` |
|    |   |   | ``x_4`` | ``x_4`` |

Next, we exchange between 1 and 3 and simultaneously between 2 and 4.
The complexity is ``\alpha + 2\beta n/4``.
In total, we have complexity
```math
\begin{align}
\log_2(p) \alpha + \beta n(1 + 2 + 4 + \cdots + p/2)/p
& =
\log_2(p) \alpha + \beta n(p-1)/p\\
& \approx \log_2(p) \alpha + \beta n.
\end{align}
```
""",
)

# ╔═╡ e119c2d3-1e24-464f-b812-62f28c00a913
frametitle("Reduce scatter")

# ╔═╡ dbc19cbb-1349-4904-b655-2452aa7e2452
vbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_{1,1}`` | ``x_{1,2}`` | ``x_{1,3}`` | ``x_{1,4}`` |
|    | ``x_{2,1}`` | ``x_{2,2}`` | ``x_{2,3}`` | ``x_{2,4}`` |
|    | ``x_{3,1}`` | ``x_{3,2}`` | ``x_{3,3}`` | ``x_{3,4}`` |
|    | ``x_{4,1}`` | ``x_{4,2}`` | ``x_{4,3}`` | ``x_{4,4}`` |
""",
	#Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After `MPI_Reduce_scatter`

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_{1,1} + \cdots + x_{1,4}`` |  |  |  |
|    |  | ``x_{2,1} + \cdots + x_{2,4}`` |  |  |
|    |  |  | ``x_{3,1} + \cdots + x_{3,4}`` |  |
|    |  |  |  | ``x_{4,1} + \cdots + x_{4,4}`` |
""",
])

# ╔═╡ 2ff573a3-4a84-4497-9305-2d97e35e5e3d
Foldable(md"Can `MPI_Reduce_scatter` be implemented by combining existing collectives ?", md"`MPI_Reduce_scatter` can be implemented by `MPI_Reduce` followed by `MPI_Scatter`")

# ╔═╡ 6be49c46-4900-4457-81b4-0704cd7da0af
Foldable(
	md"""
Would it be more efficient to have a specialized implementation instead of combining existing collectives ?
""",
	md"""
Let the size of ``x_i`` be ``n/p``. `MPI_Reduce` has complexity ``\log_2(p)(\alpha + \beta n + \gamma n)`` and `MPI_Scatter` has complexity ``\log_2(p) \alpha + \beta n``
so in total ``\log_2(p) (\alpha + \beta n)``. Can we do better ?

Start exchanging between 1 and 2 and simultaneously exchanging between 3 and 4.
The complexity is ``\alpha + 2(\beta + \gamma) n/4``.

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_{1,1} + x_{1,2}`` |  | ``x_{1,3} + x_{1,4}`` |  |
|    |  | ``x_{2,1} + x_{2,2}`` |  | ``x_{2,3} + x_{2,4}`` |
|    | ``x_{3,1} + x_{3,2}`` |  | ``x_{3,3} + x_{3,4}`` |  |
|    |  | ``x_{4,1} + x_{4,2}`` |  | ``x_{4,3} + x_{4,4}`` |

Next, we exchange between 1 and 3 and simultaneously between 2 and 4.
The complexity is ``\alpha + (\beta + \gamma) n/4``.
In total, we have complexity
```math
\begin{align}
  \log_2(p) \alpha + (\beta + \gamma) n(p/2 + \cdots + 4 + 2 + 1)/p
  & =
  \log_2(p) \alpha + (\beta + \gamma) n(p-1)/p\\
  & \approx
  \log_2(p) \alpha + (\beta + \gamma) n.
\end{align}
```
""",
)

# ╔═╡ 60bc118f-6795-43f9-97a2-865fd1704895
frametitle("Allreduce")

# ╔═╡ 0d69e94b-492a-4acc-adba-a2126b871724
vbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` | ``x_2`` | ``x_3`` | ``x_4`` |
""",
	#Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After `MPI_Reduce_scatter`

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1 + \cdots + x_4`` |``x_1 + \cdots + x_4`` | ``x_1 + \cdots + x_4`` | ``x_1 + \cdots + x_4`` |
""",
])

# ╔═╡ b9a9e335-1328-4c63-a213-ce21263bc201
Foldable(
	md"Can `MPI_Allreduce` be implemented by combining existing collectives ?",
	md"""
`MPI_Allreduce` can be implemented either by combining `MPI_Reduce` followed by `MPI_Bcast` or `MPI_Reduce_scatter` followed by `MPI_Allgather`.
The first choice would lead to a complexity of ``\log_2(p)(\alpha + \beta n)``
The second would lead to a complexity of ``\log_2(p)\alpha + \beta n``.
""",
)

# ╔═╡ a1b2d090-d498-4d5d-90a0-8cdc648dc833
section("Distributed sum")

# ╔═╡ a771f33f-7ed1-41aa-bee0-c215729a8c8d
frametitle("Distributed vector")

# ╔═╡ 370f0f20-e373-4028-bca1-83e93678cbcb
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/mpi-array.png")

# ╔═╡ 141d162c-c817-498f-be16-f1cd35d82487
Foldable(md"How to collect the partial sums ?", md"`MPI_Reduce`")

# ╔═╡ 7cf59087-efca-4f03-90dc-f2acefdcbc8a
frametitle("Let's try it")

# ╔═╡ 4788d8b4-2efa-4489-80c3-71f405513644
md"`num_processes` = $(@bind sum_num_processes Slider(2:8, default = 2, show_value = true))"

# ╔═╡ 35aa1295-642f-4525-bf19-df2a42ff39d6
compile_and_run(Example("mpi_sum.c"), mpi = true, num_processes = sum_num_processes, verbose = 1)

# ╔═╡ e832ce25-94e2-4743-854d-02b52cc7b56d
aside(Foldable(md"Why is it the first process that gets the sum ?", md"We gave 0 to the 6th argument of `MPI_Reduce`, this decides which node gets the sum."), v_offset = -100)

# ╔═╡ 79b405a5-54b5-4727-a0cd-b79522ad109f
section("Point-to-point")

# ╔═╡ d2104fbd-ba22-4501-b03a-8809271d598b
frametitle(md"Blocking communication")

# ╔═╡ 0e640e07-82c7-4dab-a8f1-2f634bbebdea
hbox([
	img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-ideal.png", :height => "150pt"),
	img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-blocking.png", :height => "160pt"),
])

# ╔═╡ 4569aa05-9963-4976-ac63-caf3f3979e83
md"""
Blocking send/received with `MPI_Send` and `MPI_Recv`.

The network cannot buffer the whole message (unless it is short). The sender need to wait for the receiver to be ready and then transfer its copy of the data.
"""

# ╔═╡ 34a10003-2c32-4332-b3e6-ce70eec0cbbe
frametitle("Example")

# ╔═╡ ce7bf747-7116-4e76-9004-f234317046c3
compile_and_run(Example("mpi_bench1.c"), mpi = true, num_processes = 2)

# ╔═╡ d7e31ced-4eb2-4221-b83f-462e8f32fe89
aside(Foldable(md"Is this timing bandwith accurately ?",
md"No, the time also includes the time that process 0 has to wait until process 1 is ready to start receiving. If the message is too small, it will just buffer the message and `MPI_Send` could return before the other process even reached `MPI_Recv`, see next slide."
), v_offset = -500)

# ╔═╡ c3c848ff-526a-450d-9b1c-5d9d3ccccf28
frametitle("Eager vs rendezvous protocol")

# ╔═╡ 67dee339-98b4-4714-88b2-8098a13235f2
md"""
There are two protocols:
* Rendezvous protocol
  1. the sender sends a header;
  2. the receiver returns a ‘ready-to-send’ message;
  3. the sender sends the actual data.
* Eager protocol the message is buffered so `MPI_Send` can return eagerly, before the receiver is even ready

Eager protocol is used if the data size is smaller than the *eager limit*.
To force the rendezvous protocol, use `MPI_Ssend`.
"""

# ╔═╡ 32f740e7-9338-4c42-8eaf-ce8022412c50
frametitle("Nonblocking communication")

# ╔═╡ 8a527c17-bf2b-4e6b-937f-ef3a269c5112
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-nonblocking.jpeg", :height => "200pt")

# ╔═╡ 93f0c63c-b597-4f89-809c-7af0476f319a
md"""
`MPI_Isend` and `MPI_Irecv` where `I` stands for `immediate` or `incomplete`.
`MPI_Wait` can be used to wait for the send and receive to finish.
"""

# ╔═╡ 568057f5-b0b8-4225-8e4b-5eec911a52ef
frametitle("Example")

# ╔═╡ 26aa369f-e5c7-4fe5-8b6b-903f4f4e91ba
compile_and_run(Example("mpi_bench2.c"), mpi = true, num_processes = 2)

# ╔═╡ a79c410a-bebf-434c-9730-568e0ff4f4c7
section("Consortium des Équipements de Calcul Intensif (CÉCI)")

# ╔═╡ 39f48c25-6efb-4ff2-aedc-9d3e722dad24
md"""
* [Follow README instructions to create an account and setup your computer](https://github.com/blegat/LINMA2710?tab=readme-ov-file#ceci-cluster)
  - Don't wait the last minute, if you get into trouble it's easier to get this setup before you actually need it
* Select $(img("https://www.ceci-hpc.be/assets/img/new_ceci_logo.png", :height => "15pt")) cluster from [the list](https://www.ceci-hpc.be/clusters.html) + `manneback` for GPU. You only have access to Tier-2 clusters. This sadly leaves out:
  - Tier-1 clusters such as Lucia
  - Tier-0 cluster such as $(img("https://www.lumi-supercomputer.eu/content/uploads/2020/02/lumi_logo.png", :height => "15pt")) from $(img("https://upload.wikimedia.org/wikipedia/commons/8/8f/HPC_JU_logo_RGB.svg", :height => "20pt"))
* Connect with SSH using `ssh lemaitre4` or `ssh manneback`.
"""

# ╔═╡ 55e96151-2aa1-4ea0-b672-2038c57d911e
aside(img("https://upload.wikimedia.org/wikipedia/en/3/3e/The_LUMI_supercomputer.jpg", :height => "100pt"), v_offset = -140)

# ╔═╡ be0e3ba0-18cc-4b9a-a56d-2566f5148fae
frametitle(md"""$(img("https://github.com/TACC/Lmod/raw/main/logos/2x/Lmod-4color%402x.png", :height => "30px"))""")

# ╔═╡ c1285653-38ba-418b-bdf5-cda99440998d
aside(tip(Foldable(md"Use `module spider` to see which version are available",
md"""
```
[blegat@lm4-f001 ~]$ module spider gompi

----------------------------
  gompi:
----------------------------
    Description:
      GNU Compiler Collection (GCC) based compiler toolchain, including OpenMPI for MPI support.

     Versions:
        gompi/2021b
        gompi/2022b
        gompi/2023a
        gompi/2023b

----------------------------
  For detailed information about a specific "gompi" package (including how to load the modules) use the module's full name.
  Note that names that have a trailing (E) are extensions provided by other modules.
  For example:

     $ module spider gompi/2023b
----------------------------
```
""")), v_offset = -300)

# ╔═╡ 88f33f35-d922-4d98-af4a-ebb79d9b7dc6
mpicc_cmd = md"""
```sh
[blegat@lm4-f001 ~]$ mpicc
-bash: mpicc: command not found

[blegat@lm4-f001 ~]$ module load gompi/2023a

[blegat@lm4-f001 ~]$ mpicc
gcc: fatal error: no input files
compilation terminated.
```
""";

# ╔═╡ e3474aea-ee14-4c78-ae46-5badc66a543a
list_1 = Foldable(md"`[blegat@lm4-f001 ~]$ module list`", md"""
```
Currently Loaded Modules:
  1) tis/2018.01 (S)   2) releases/2023a (S)   3) StdEnv

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```
""");

# ╔═╡ 6c1984f6-4e36-4637-b0da-c7dd8b0f9ff0
list_2 = Foldable(md"`[blegat@lm4-f001 ~]$ module list`", md"""
```
Currently Loaded Modules:
  1) tis/2018.01                   (S)  11) libpciaccess/0.17-GCCcore-12.3.0
  2) releases/2023a                (S)  12) hwloc/2.9.1-GCCcore-12.3.0
  3) StdEnv                             13) OpenSSL/1.1
  4) GCCcore/12.3.0                     14) libevent/2.1.12-GCCcore-12.3.0
  5) zlib/1.2.13-GCCcore-12.3.0         15) UCX/1.14.1-GCCcore-12.3.0
  6) binutils/2.40-GCCcore-12.3.0       16) libfabric/1.18.0-GCCcore-12.3.0
  7) GCC/12.3.0                         17) PMIx/4.2.4-GCCcore-12.3.0
  8) numactl/2.0.16-GCCcore-12.3.0      18) UCC/1.2.0-GCCcore-12.3.0
  9) XZ/5.4.2-GCCcore-12.3.0            19) OpenMPI/4.1.5-GCC-12.3.0
 10) libxml2/2.11.4-GCCcore-12.3.0      20) gompi/2023a

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```
""");

# ╔═╡ c0daf219-cb87-4203-b835-49ab7eb955be
md"""
```
[local computer]$ ssh lemaitre4
```

$list_1

$mpicc_cmd

$list_2
"""

# ╔═╡ 51d70f9a-cd67-44b9-8fd1-5ab70b526c7a
frametitle("Launching a job")

# ╔═╡ 944d827e-bc6a-4de8-b959-5fde8790bedc
md"""
```sh
[local computer]$ ssh lemaitre4
[blegat@lm4-f001 ~]$ cd LINMA2710/examples
[blegat@lm4-f001 examples]$ mpicc procname.c
-bash: mpicc: command not found
```
"""

# ╔═╡ 3a2bfd4e-0ce6-4a79-a578-fc1b4ef563c5
Foldable(md"How to fix it ?", md"""
We should load `gompi` or at least `OpenMPI`:
```sh
[blegat@lm4-f001 examples]$ module load OpenMPI
[blegat@lm4-f001 examples]$ mpicc procname.c
[blegat@lm4-f001 examples]$ mpiexec -n 4 a.out
Process 1/4 is running on node <<lm4-f001>>
Process 3/4 is running on node <<lm4-f001>>
Process 0/4 is running on node <<lm4-f001>>
Process 2/4 is running on node <<lm4-f001>>
```
$(Foldable(md"Why are they all on same node ?", md"We are on the *login node*, we need to run jobs on the *compute nodes* using Slurm !"))
""")

# ╔═╡ beee4908-d519-413a-964f-149bb82cdbb8
frametitle("Slurm")

# ╔═╡ d8bb1d43-bf42-4a09-bdeb-5db406ef1ccd
hbox([Div(md"""
* `srun` : Synchronous (blocked) job
```
[blegat@lm4-f001 ~]$ srun --time=1 pwd
srun: job 3491072 queued and waiting for resources
srun: job 3491072 has been allocated resources
/home/users/b/l/blegat
```
* `$ sbatch submit.sh` : Asynchronous job, get status with
* `$ squeue --me`
""", style = Dict("flex-grow" => "1", "margin-right" => "30px")),
md"""
$(img("https://upload.wikimedia.org/wikipedia/commons/3/3a/Slurm_logo.svg", :width => "160px", :height => "160px"))
See [CÉCI documentation](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html)
""",
])

# ╔═╡ 9a100ccf-1ad3-4d2c-bbe0-e297969eb69e
section("Topology")

# ╔═╡ 921b5a18-0733-4032-a543-9d60e254b1b2
md"""
Topology awareness is important, specified in [Slurm's `topology.conf` file](https://slurm.schedmd.com/topology.conf.html).
"""

# ╔═╡ 10a1b3a7-21c7-4f97-93e1-006ad3aea40d
frametitle("Butterfly")

# ╔═╡ 3ec3c058-a94d-4717-b99f-66373f2fa31d
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/butterflys.jpeg")

# ╔═╡ 21d507f6-02f8-4f8b-84f1-bcb84731df66
frametitle("Fat-tree")

# ╔═╡ 4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/fattree5.jpg")

# ╔═╡ b53ec488-ff25-4647-ab00-fbf90963a795
md"""
*blocking factor* : Ratio between upper links and lower links. Ratio is 1 for fat-tree to prevent bottlenecks if all nodes start communicating.
"""

# ╔═╡ 972b8af7-5e4d-4236-8875-016d1ed5b535
Pkg.instantiate()

# ╔═╡ 9d0c7847-a76f-42a0-b73c-0aed1c58d87a
biblio = load_biblio!();

# ╔═╡ 2504c31b-ea38-403f-931a-8ebb72e73af4
bibrefs(biblio, "eijkhout2017Parallel")

# ╔═╡ 3a50ca06-06e8-4a61-ade2-afbfc52ca655
aside(md"""See $(bibcite(biblio, "eijkhout2017Parallel", "Section 4.1.4.2"))""", v_offset = -100)

# ╔═╡ a59db59c-d34e-4abd-8865-9907607e06a8
aside(md"""From $(bibcite(biblio, "eijkhout2010Introduction", "Figure 2.27"))""", v_offset = -200)

# ╔═╡ f2417047-33fc-4489-8e89-115bc6b46c13
aside(md"""From $(bibcite(biblio, "eijkhout2010Introduction", "Figure 2.30"))""", v_offset = -200)

# ╔═╡ Cell order:
# ╟─58e12afd-6eb0-4731-bd57-d9ae7ab4e164
# ╟─2504c31b-ea38-403f-931a-8ebb72e73af4
# ╟─5a566137-fbd1-45b2-9a55-e4aded366bb3
# ╟─a6c337c4-0c81-4463-ad4f-9a4528d953ab
# ╟─c04bcc96-e5fe-4d6e-a12e-40dcde58c62e
# ╟─cf799c26-1cea-4b38-9a15-8497813bd668
# ╟─6d2b3dbc-0686-49f0-904a-56c3ce63b4dd
# ╟─b5a3e471-af4a-466f-bbae-96306bcc7563
# ╟─d722a86d-6d51-4d91-ac22-53af94c91497
# ╟─c3590376-06ed-45a4-af0b-2d46f1a387c8
# ╟─52d428d5-cb33-4f2a-89eb-3a8ce3f5bb81
# ╟─273ad3a6-cb32-49bb-8702-fdaf8597e812
# ╟─4e32f7fb-cd5a-4190-9c92-ba4029313475
# ╟─82230d6c-25ce-4d12-8842-e0651fc4b143
# ╟─7d9ac5f9-39bf-4052-ad8a-ac0fec15c64a
# ╟─b0ca0392-71b8-4f44-8c6c-0978a02a0e6c
# ╟─a103c5af-42fe-4f8c-b78c-6946895105d7
# ╟─21b6133f-db59-4885-9b3d-331c3d6ef306
# ╟─35ba1eea-56ae-4b74-af96-21ec5a93c455
# ╠═8981b5e2-2497-478e-ab28-a14b62f6f916
# ╠═5441e428-b320-433c-acde-15fe6bf58537
# ╟─40606ee3-38cc-4123-9b86-b774bf89e499
# ╟─b94cd399-0370-49e9-a522-056f3af22955
# ╟─9b4cae31-c319-444e-98c8-2c0bfc6dfa0c
# ╟─8b83570a-6982-47e5-a167-a6d6afee0f7d
# ╟─5d72bf87-7f3a-4229-9d7a-2e63c115087d
# ╟─7b1d26c6-9499-4e44-84c8-c272737a175e
# ╟─fc43b343-79cd-4342-8d80-8ea72cf34942
# ╟─233c13ff-f008-40b0-a6c5-c5395b2215ec
# ╟─ad3559d1-6180-4eaa-b97d-3c1f10f036b9
# ╟─c420ad25-6af1-4fb4-823a-b6bbd4e10f7f
# ╟─db16e939-b490-497b-a03f-80ce2e8485af
# ╟─4fdb4cd6-a794-4b14-84b0-72f484c6ea86
# ╟─a258eec9-f4f6-49bd-8470-8541836f5f6b
# ╟─6fc34de1-469b-41a9-9677-ff3182f7a498
# ╟─de20bf96-7d33-4a78-8147-f0b7f8488e46
# ╟─e119c2d3-1e24-464f-b812-62f28c00a913
# ╟─dbc19cbb-1349-4904-b655-2452aa7e2452
# ╟─2ff573a3-4a84-4497-9305-2d97e35e5e3d
# ╟─6be49c46-4900-4457-81b4-0704cd7da0af
# ╟─60bc118f-6795-43f9-97a2-865fd1704895
# ╟─0d69e94b-492a-4acc-adba-a2126b871724
# ╟─b9a9e335-1328-4c63-a213-ce21263bc201
# ╟─a1b2d090-d498-4d5d-90a0-8cdc648dc833
# ╟─a771f33f-7ed1-41aa-bee0-c215729a8c8d
# ╟─370f0f20-e373-4028-bca1-83e93678cbcb
# ╟─141d162c-c817-498f-be16-f1cd35d82487
# ╟─7cf59087-efca-4f03-90dc-f2acefdcbc8a
# ╟─35aa1295-642f-4525-bf19-df2a42ff39d6
# ╟─4788d8b4-2efa-4489-80c3-71f405513644
# ╟─e832ce25-94e2-4743-854d-02b52cc7b56d
# ╟─79b405a5-54b5-4727-a0cd-b79522ad109f
# ╟─d2104fbd-ba22-4501-b03a-8809271d598b
# ╟─0e640e07-82c7-4dab-a8f1-2f634bbebdea
# ╟─4569aa05-9963-4976-ac63-caf3f3979e83
# ╟─34a10003-2c32-4332-b3e6-ce70eec0cbbe
# ╟─ce7bf747-7116-4e76-9004-f234317046c3
# ╟─d7e31ced-4eb2-4221-b83f-462e8f32fe89
# ╟─c3c848ff-526a-450d-9b1c-5d9d3ccccf28
# ╟─67dee339-98b4-4714-88b2-8098a13235f2
# ╟─3a50ca06-06e8-4a61-ade2-afbfc52ca655
# ╟─32f740e7-9338-4c42-8eaf-ce8022412c50
# ╟─8a527c17-bf2b-4e6b-937f-ef3a269c5112
# ╟─93f0c63c-b597-4f89-809c-7af0476f319a
# ╟─568057f5-b0b8-4225-8e4b-5eec911a52ef
# ╟─26aa369f-e5c7-4fe5-8b6b-903f4f4e91ba
# ╟─a79c410a-bebf-434c-9730-568e0ff4f4c7
# ╟─39f48c25-6efb-4ff2-aedc-9d3e722dad24
# ╟─55e96151-2aa1-4ea0-b672-2038c57d911e
# ╟─be0e3ba0-18cc-4b9a-a56d-2566f5148fae
# ╟─c0daf219-cb87-4203-b835-49ab7eb955be
# ╟─c1285653-38ba-418b-bdf5-cda99440998d
# ╟─88f33f35-d922-4d98-af4a-ebb79d9b7dc6
# ╟─e3474aea-ee14-4c78-ae46-5badc66a543a
# ╟─6c1984f6-4e36-4637-b0da-c7dd8b0f9ff0
# ╟─51d70f9a-cd67-44b9-8fd1-5ab70b526c7a
# ╟─944d827e-bc6a-4de8-b959-5fde8790bedc
# ╟─3a2bfd4e-0ce6-4a79-a578-fc1b4ef563c5
# ╟─beee4908-d519-413a-964f-149bb82cdbb8
# ╟─d8bb1d43-bf42-4a09-bdeb-5db406ef1ccd
# ╟─9a100ccf-1ad3-4d2c-bbe0-e297969eb69e
# ╟─921b5a18-0733-4032-a543-9d60e254b1b2
# ╟─10a1b3a7-21c7-4f97-93e1-006ad3aea40d
# ╟─3ec3c058-a94d-4717-b99f-66373f2fa31d
# ╟─a59db59c-d34e-4abd-8865-9907607e06a8
# ╟─21d507f6-02f8-4f8b-84f1-bcb84731df66
# ╟─4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
# ╟─b53ec488-ff25-4647-ab00-fbf90963a795
# ╟─f2417047-33fc-4489-8e89-115bc6b46c13
# ╟─82e1ea5e-f8e0-11ef-0f93-49a66050feaf
# ╟─972b8af7-5e4d-4236-8875-016d1ed5b535
# ╟─8df4ff2f-d176-4b4e-a525-665b5d07ea52
# ╟─a55c5090-1455-4fb9-bdb9-a0b9b340b154
# ╟─9d0c7847-a76f-42a0-b73c-0aed1c58d87a
