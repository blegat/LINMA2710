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
[laptop]$ ssh lemaitre4
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
* More details on the [README](https://github.com/blegat/LINMA2710)
""", style = Dict("flex-grow" => "1", "margin-right" => "30px")),
md"""
$(img("https://upload.wikimedia.org/wikipedia/commons/3/3a/Slurm_logo.svg", :width => "160px", :height => "160px"))
See [CÉCI documentation](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html)
""",
])

# ╔═╡ b540d5e3-6686-479a-b2c7-c1f65b85b6ba
frametitle("Profiling with NVIDIA Nsight Systems")

# ╔═╡ 091dd042-580b-4fda-8086-e048663aed6c
md"""
* NVIDIA Nsight Systems $(img("https://developer.download.nvidia.com/images/nvidia-nsight-systems-icon-gbp-shaded-256.png", :width => "20pt")) can provile CUDA code but also MPI
* Available on `manneback` after loading `CUDA` with $(img("https://github.com/TACC/Lmod/raw/main/logos/2x/Lmod-4color%402x.png", :height => "20px"))

```sh
[laptop]$ ssh manneback
[blegat@mbackf1 ~]$ nsys
-bash: nsys: command not found
[blegat@mbackf1 ~]$ ml CUDA
[blegat@mbackf1 ~]$ nsys
```
"""

# ╔═╡ 9a100ccf-1ad3-4d2c-bbe0-e297969eb69e
section("Topology")

# ╔═╡ 9612a1ef-fd3a-4a58-87b0-b2255ac86331
frametitle("Graph diameter")

# ╔═╡ 98392c40-6542-4a26-8552-c0960bbaa6a6
md"""
* Consider graph ``G`` with nodes ``v`` corresponding to computer nodes or switches.
* There is an edge ``(u, v) \in E`` if there is an ethernet cable **directly** connecting ``u`` and ``b``.
*  ``e \in E`` are ethernet cables of bandwidth ``w_e``
* Distance (unweighted) from node ``i \in V`` to node ``j \in V`` is ``d(G, u, v)``
  - Does not depend on bandwidth ``w_e`` of edges of the path
"""

# ╔═╡ 49b596b8-891d-4f3f-a6a4-a62cc8237df3
definition("Graph diameter", md"*Graph diameter* is ``d(G) := \max_{u, v \in V} d(G, u, v)``")

# ╔═╡ c253bb24-ad76-4b58-8dfc-7dc2576e3db5
frametitle("Bisection bandwidth")

# ╔═╡ 1b617828-e2b2-4a94-a120-59fa533d3e11
md"""
Bandwidth ``\texttt{bw}(u, v)`` is the bandwidth of the cable if ``(u, v) \in E``
or 0 otherwise. Given ``S, T \subseteq V``,
```math
\begin{align}
\text{Width} &\qquad &  w(S, T) & = |\{ (u, v) \in E \mid u \in S, v \in T \}|\\
\text{Bandwidth} & & \texttt{bw}(S, T) & = \sum_{u\in S, v\not\in S} w(u,v)
\end{align}
```
"""

# ╔═╡ f2ebc6fb-e07c-4922-897d-9bbe0f5fa1d0
#definition("Bisection bandwidth", 
hbox([
	md"""
The *bisection width* is:
```math
\min_{S \subset V : \lfloor |V|/2 \rfloor \le |S| \le \lceil |V|/2 \rceil} \quad w(S, V \setminus S)
```
""", Div(html" ",  style = Dict("flex-grow" => "1")),
	md"""
The *bisection **band**width* is:
```math
\min_{S \subset V : \lfloor |V|/2 \rfloor \le |S| \le \lceil |V|/2 \rceil} \quad \texttt{bw}(S, V \setminus S)
```
"""])#)

# ╔═╡ 8da580fe-6b56-4d8f-ad43-aed7b728a06e
md"""
* Worst case pairwise communication of two groups ``S`` and ``V \setminus S`` of *almost* (``\pm 1``) equal size.
* NP-hard to compute for general graphs
"""

# ╔═╡ fa024a5d-52a6-459d-894d-13a60ec723d2
Foldable(md"What are the differences with Min-Cut ?",
md"""
In Min-Cut, we fix a node in ``S``, a node in ``V \setminus S``
and the cardinality of `S` is not constrained.
These differences allow Min-Cut to be solvable in polynomial time.
""")

# ╔═╡ 360091c4-d3a0-462d-abcf-b9bbb9480871
frametitle("Linear array")

# ╔═╡ 3dc860be-016d-49ee-8535-7d9457c70f85
Foldable(md"What is the graph diameter ?", md"``|V| - 1`` if ``u`` and ``v`` are extreme points of the array")

# ╔═╡ 7fc70992-973a-43c6-904a-dd1b622a5ed8
Foldable(md"What is the bisection width ?", md"""
The bisection width is 1 : $(img("https://upload.wikimedia.org/wikipedia/commons/7/79/Bisected_linear_array.jpg", :width => "300pt"))
""")

# ╔═╡ c55dcd4a-8438-4679-9c4a-78cceec6835d
function path(ring::Bool; s = 80, offset = 0.04)
	off(a, b) = a + sign(b - a) * offset
	p(i, j) = Point(i * s, j * s)
	c(m, i, j) = circle(p(i, j), 0.06s, action = :fill)
	a(i1, j1, i2, j2) = line(p(off(i1, i2), off(j1, j2)), p(off(i2, i1), off(j2, j1)), action = :stroke)
	function ac(i1, j1, i2, j2, m)
		a(i1, j1, i2, j2)
		c(m, i2, j2)
	end
	@draw begin
		c("1", -3, 0)
		ac(-3, 0, -2, 0, "2")
		ac(-2, 0, -1, 0, "3")
		ac(-1, 0, 0, 0, "4")
		ac(0, 0, 1, 0, "5")
		if ring
			move(p(off(1, 0), off(0, -1)))
			curve(p(off(1, 2), off(0, -1)), p(-1, -1), p(off(-3, -2), off(0, -1)))
			strokepath()
		end
	end 7.5s 1.7s
end;

# ╔═╡ e44b0038-d68f-4a49-9da2-67fbcbe098c3
path(false)

# ╔═╡ 7d37fbea-baa3-43ec-b003-a4707017a4cf
frametitle("Rings")

# ╔═╡ fc705b81-7310-44cc-ad9f-dc2cf8a9b645
path(true)

# ╔═╡ 86394e1c-0ff4-449a-8940-4b5906d8b6f0
Foldable(md"What is the graph diameter ?", md"``|V|/2``")

# ╔═╡ 23bfbe95-7ba2-41b9-bd8b-dc4baa3ad53a
Foldable(md"What is the bisection width ?", md"""
The bisection width is 2: 
$(img("https://upload.wikimedia.org/wikipedia/commons/5/51/Bisected_ring.jpg", :width => "300pt"))
""")

# ╔═╡ 2257220c-6f0e-4edf-9fea-7e388b84df9b
frametitle("Multidimensional array and torus")

# ╔═╡ 39b055f5-3dbf-403c-b21e-210e3813d8b0
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/torus.jpeg")

# ╔═╡ 2e4dc3f9-a132-444f-a35d-f583823a7dfd
Foldable(md"What is the graph diameter of a ``n \times n`` 2D array ?",
md"""
It is ``2(n-1)``, attained for opposite vertices of the square.
$(Foldable(md"What is the bisection width of a ``n^d`` ``d``D array ?",
md"It is ``d(n-1)``, attained for opposite vertices of the hypercube."))
""")

# ╔═╡ b68eb860-a5b4-4e9e-9fbf-6eb6ce43ae69
Foldable(md"What is the bisection width of a ``n \times n`` 2D array ?",
md"""
It is ``n = \sqrt{|V|}``:
$(img("https://upload.wikimedia.org/wikipedia/commons/2/2f/Bisected_mesh.jpg", :width => "300pt"))
$(Foldable(md"What is the bisection width of a ``n^d`` ``d``D array ?",
md"It is 1 for ``d = 1``, ``n`` for ``d = 2`` and ``n^2`` for ``d = 3``. In general, it is ``n^{-1} = |V|^{(d-1)/d}``"))
""")

# ╔═╡ 2c84bd84-b54d-4594-b9f8-35db2124d7e8
frametitle("Hypercube")

# ╔═╡ 4309dc43-aeb8-4ec7-94fe-0e320b784349
md"Special case of multidimensional array"

# ╔═╡ f6f9447c-9bc9-432d-bd80-2c39f9d842f8
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/hypercubes.jpg", :width => "400pt")

# ╔═╡ 1551122c-70ae-4e37-b3fb-4be91fcc4afb
Foldable(
md"""
How to order the nodes so that consecutive nodes in the order are adjacent in the graph ?
""",
md"""
Map nodes to binary number and use [Gray code](https://en.wikipedia.org/wiki/Gray_code).
$(img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/hypercubenumber.jpg", :width => "300pt"))
"""
)

# ╔═╡ e796b093-9c1d-4656-9acb-918de53f7e4d
frametitle("Crossbar")

# ╔═╡ 97d3cf3f-ddac-4850-8b05-bdc0c4741f61
Foldable(md"What are the number of switches, edges, graph diameter and bisection width for ``n`` computer nodes ?",
md"""
``2n(n-1)`` switches, ``|E| = 4n(n-1)``, diameter is 2 and bisection width is ``n/2``.
""")

# ╔═╡ 143dca7c-f9a4-472a-a4bc-4578e4e8413b
frametitle("Tree")

# ╔═╡ e4d1de1d-d57a-48ab-ad7a-c09b427daa03
Foldable(md"What is the diameter and bisection width of ``n`` computer nodes ?",
md"""
Diameter is ``2\log_2(n)`` and bisection width is 1.
$(img("https://upload.wikimedia.org/wikipedia/commons/d/da/Bisected_tree.jpg") )
""")

# ╔═╡ 954f1ab1-1e2f-458b-96d7-a1746631fac7
function tree(; s = 80, offset = 0.04)
	off(a, b) = a + sign(b - a) * offset
	p(i, j) = Point(i * s, j * s)
	c(i, j; kws...) = circle(p(i, j), 0.06s; kws...)
	a(i1, j1, i2, j2) = line(p(off(i1, i2), off(j1, j2)), p(off(i2, i1), off(j2, j1)), action = :stroke)
	function ac(i1, j1, i2, j2; kws...)
		a(i1, j1, i2, j2)
		c(i2, j2; kws...)
	end
	@draw begin
		c(0, -1, action = :stroke)
		ac(0, -1, -2, 0, action = :stroke)
		ac(0, -1, 2, 0, action = :stroke)
		ac(-2, 0, -3, 1, action = :fill)
		ac(-2, 0, -1, 1, action = :fill)
		ac(2, 0, 3, 1, action = :fill)
		ac(2, 0, 1, 1, action = :fill)
		c(2.8, -0.9, action = :stroke)
		c(2.8, -0.7, action = :fill)
		text("Switch", p(3, -0.9), valign = :middle)
		text("Computer node", p(3, -0.7), valign = :middle)
	end 8s 3s
end;

# ╔═╡ 1bac238f-79c8-4f9f-a187-bacb288de3b0
tree()

# ╔═╡ 21d507f6-02f8-4f8b-84f1-bcb84731df66
frametitle("Fat-tree")

# ╔═╡ 4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/fattree5.jpg", :width => "500pt")

# ╔═╡ b53ec488-ff25-4647-ab00-fbf90963a795
md"""
*blocking factor* : Ratio between upper links and lower links. Ratio is 1 for fat-tree to prevent bottlenecks if all nodes start communicating.
"""

# ╔═╡ de72d596-0daf-4629-bbb5-20bb8a67cbed
Foldable(md"What is the number of edges ? What is the bisection width ?",
md"""
Number of edges is ``n\log_2(n)`` and bisection width is ``n/2``.
""")

# ╔═╡ 10a1b3a7-21c7-4f97-93e1-006ad3aea40d
frametitle("Butterfly")

# ╔═╡ f7f097cb-d7bd-49eb-a030-ac26f8f61a67
md"Fat-tree need large switches, alternative is butterfly network:"

# ╔═╡ 3ec3c058-a94d-4717-b99f-66373f2fa31d
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/butterflys.jpeg")

# ╔═╡ 6041a909-d26c-4ab1-836b-29953c578759
Foldable(md"What is the number of edges ? What is the bisection width ?",
md"""
Same as fat-tree.
""")

# ╔═╡ 972b8af7-5e4d-4236-8875-016d1ed5b535
Pkg.instantiate()

# ╔═╡ 9d0c7847-a76f-42a0-b73c-0aed1c58d87a
biblio = load_biblio!();

# ╔═╡ 2504c31b-ea38-403f-931a-8ebb72e73af4
bibrefs(biblio, "eijkhout2017Parallel")

# ╔═╡ 3a50ca06-06e8-4a61-ade2-afbfc52ca655
aside(md"""See $(bibcite(biblio, "eijkhout2017Parallel", "Section 4.1.4.2"))""", v_offset = -100)

# ╔═╡ 921b5a18-0733-4032-a543-9d60e254b1b2
md"""
* Specializing on topology is important for communication libraries like MPI/NCCL. For instance, Deepseek-V3 by-passed NCCL and used PTX directly to hardcode how ther hardware should be used.
* Specified in [Slurm's `topology.conf` file](https://slurm.schedmd.com/topology.conf.html).
* Source : $(bibcite(biblio, "eijkhout2010Introduction", "Section 2.7"))
"""

# ╔═╡ a59db59c-d34e-4abd-8865-9907607e06a8
aside(md"""From $(bibcite(biblio, "eijkhout2010Introduction", "Figure 2.27"))""", v_offset = -200)

# ╔═╡ f2417047-33fc-4489-8e89-115bc6b46c13
aside(md"""From $(bibcite(biblio, "eijkhout2010Introduction", "Figure 2.30"))""", v_offset = -200)

# ╔═╡ 1152dec8-3810-42b1-bb2a-8755dcaef56c
img1(f, args...) = img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/$f", args...)

# ╔═╡ d04b9af5-f004-4ca4-b1c9-2c86d46cb37d
hbox([
Div(md"""
* Each dot is a node
* Each intersection is a switch
$(Foldable("What is the underlying graph between nodes", "Complete directed graph"))
""", style = Dict("flex-grow" => "1")),
img1("crossbar.jpg"),
])

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
# ╟─b540d5e3-6686-479a-b2c7-c1f65b85b6ba
# ╟─091dd042-580b-4fda-8086-e048663aed6c
# ╟─9a100ccf-1ad3-4d2c-bbe0-e297969eb69e
# ╟─921b5a18-0733-4032-a543-9d60e254b1b2
# ╟─9612a1ef-fd3a-4a58-87b0-b2255ac86331
# ╟─98392c40-6542-4a26-8552-c0960bbaa6a6
# ╟─49b596b8-891d-4f3f-a6a4-a62cc8237df3
# ╟─c253bb24-ad76-4b58-8dfc-7dc2576e3db5
# ╟─1b617828-e2b2-4a94-a120-59fa533d3e11
# ╟─f2ebc6fb-e07c-4922-897d-9bbe0f5fa1d0
# ╟─8da580fe-6b56-4d8f-ad43-aed7b728a06e
# ╟─fa024a5d-52a6-459d-894d-13a60ec723d2
# ╟─360091c4-d3a0-462d-abcf-b9bbb9480871
# ╟─e44b0038-d68f-4a49-9da2-67fbcbe098c3
# ╟─3dc860be-016d-49ee-8535-7d9457c70f85
# ╟─7fc70992-973a-43c6-904a-dd1b622a5ed8
# ╟─c55dcd4a-8438-4679-9c4a-78cceec6835d
# ╟─7d37fbea-baa3-43ec-b003-a4707017a4cf
# ╟─fc705b81-7310-44cc-ad9f-dc2cf8a9b645
# ╟─86394e1c-0ff4-449a-8940-4b5906d8b6f0
# ╟─23bfbe95-7ba2-41b9-bd8b-dc4baa3ad53a
# ╟─2257220c-6f0e-4edf-9fea-7e388b84df9b
# ╟─39b055f5-3dbf-403c-b21e-210e3813d8b0
# ╟─2e4dc3f9-a132-444f-a35d-f583823a7dfd
# ╟─b68eb860-a5b4-4e9e-9fbf-6eb6ce43ae69
# ╟─2c84bd84-b54d-4594-b9f8-35db2124d7e8
# ╟─4309dc43-aeb8-4ec7-94fe-0e320b784349
# ╟─f6f9447c-9bc9-432d-bd80-2c39f9d842f8
# ╟─1551122c-70ae-4e37-b3fb-4be91fcc4afb
# ╟─e796b093-9c1d-4656-9acb-918de53f7e4d
# ╟─d04b9af5-f004-4ca4-b1c9-2c86d46cb37d
# ╟─97d3cf3f-ddac-4850-8b05-bdc0c4741f61
# ╟─143dca7c-f9a4-472a-a4bc-4578e4e8413b
# ╟─1bac238f-79c8-4f9f-a187-bacb288de3b0
# ╟─e4d1de1d-d57a-48ab-ad7a-c09b427daa03
# ╟─954f1ab1-1e2f-458b-96d7-a1746631fac7
# ╟─21d507f6-02f8-4f8b-84f1-bcb84731df66
# ╟─4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
# ╟─b53ec488-ff25-4647-ab00-fbf90963a795
# ╟─de72d596-0daf-4629-bbb5-20bb8a67cbed
# ╟─10a1b3a7-21c7-4f97-93e1-006ad3aea40d
# ╟─f7f097cb-d7bd-49eb-a030-ac26f8f61a67
# ╟─3ec3c058-a94d-4717-b99f-66373f2fa31d
# ╟─6041a909-d26c-4ab1-836b-29953c578759
# ╟─a59db59c-d34e-4abd-8865-9907607e06a8
# ╟─f2417047-33fc-4489-8e89-115bc6b46c13
# ╟─82e1ea5e-f8e0-11ef-0f93-49a66050feaf
# ╟─972b8af7-5e4d-4236-8875-016d1ed5b535
# ╟─8df4ff2f-d176-4b4e-a525-665b5d07ea52
# ╟─a55c5090-1455-4fb9-bdb9-a0b9b340b154
# ╟─9d0c7847-a76f-42a0-b73c-0aed1c58d87a
# ╟─1152dec8-3810-42b1-bb2a-8755dcaef56c
