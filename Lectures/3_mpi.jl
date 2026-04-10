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

# ╔═╡ 8df4ff2f-d176-4b4e-a525-665b5d07ea52
using SimpleClang, PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, Luxor, StaticArrays, BenchmarkTools, PlutoTeachingTools, Markdown

# ╔═╡ 58e12afd-6eb0-4731-bd57-d9ae7ab4e164
@htl("""
<p align=center style=\"font-size: 40px;\">LINMA2710 - Scientific Computing
Distributed Computing with MPI</p><p align=right><i>P.-A. Absil and B. Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ bfab5c2d-61c3-468b-9ddf-4aaa49cb7785
md"""
* [Eij10] V. Eijkhout. [Introduction to High Performance Scientific Computing](https://theartofhpc.com/istc.html). 3 Edition, Vol. 1 (Lulu.com, 2010).
* [Eij17] V. Eijkhout. [Parallel Programming in MPI and OpenMP](https://theartofhpc.com/pcse.html). 2 Edition, Vol. 2 (Lulu.com, 2017).
"""

# ╔═╡ 063f0acc-c023-46d0-9ed9-fbd7fbdcfa3b
citeintro(what) = "[Eij10; " * what * "]";

# ╔═╡ 3e98c0ca-1b47-4631-83d7-cd0c8c0a431d
citepara(what) = "[Eij17; " * what * "]";

# ╔═╡ 5a566137-fbd1-45b2-9a55-e4aded366bb3
md"# Single Program Multiple Data (SPMD)"

# ╔═╡ a6c337c4-0c81-4463-ad4f-9a4528d953ab
md"## Message Passing Interface (MPI)"

# ╔═╡ cf799c26-1cea-4b38-9a15-8497813bd668
md"## MPI basics"

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
	Div(md"Get the id of processes. `procid` is **different** for **different** processes."; style = Dict("flex-grow" => "1")),
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
md"## Different processes may be on the same node"

# ╔═╡ 82230d6c-25ce-4d12-8842-e0651fc4b143
md"## Processor name identifies the node"

# ╔═╡ 7d9ac5f9-39bf-4052-ad8a-ac0fec15c64a
md"""
Processes that are on the same node share the same `processor_name` (the `hostname`).
"""

# ╔═╡ a103c5af-42fe-4f8c-b78c-6946895105d7
md"`num_processes` = $(@bind procname_num_processes Slider(2:8, default = 2, show_value = true))"

# ╔═╡ 21b6133f-db59-4885-9b3d-331c3d6ef306
md"## Compiling"

# ╔═╡ 35ba1eea-56ae-4b74-af96-21ec5a93c455
md"""
You could simply add `lmpi` but using `mpicc` and `mpic++` is easier.
"""

# ╔═╡ 8981b5e2-2497-478e-ab28-a14b62f6f916
run(`mpicc -show`)

# ╔═╡ 5441e428-b320-433c-acde-15fe6bf58537
run(`mpic++ -show`)

# ╔═╡ 40606ee3-38cc-4123-9b86-b774bf89e499
md"# Collectives"

# ╔═╡ 9b4cae31-c319-444e-98c8-2c0bfc6dfa0c
md"## Broadcast"

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
	md"Lower bound complexity with ``p`` processes if ``x`` has  length ``n`` bytes ?",
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
md"## Gather"

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
	md"Lower bound complexity with ``p`` processes if each ``x_i`` has length ``n/p`` bytes ?",
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

In total, it is ``2\alpha + 3\beta n/4``. In general, we have
```math
\log_2(p)\alpha + \beta n(1 + 2 + 4 + \cdots + p/2)/p = \log_2(p)\alpha + \beta n(p - 1)/p \approx \log_2(p)\alpha + \beta n
```
"""
)

# ╔═╡ ad3559d1-6180-4eaa-b97d-3c1f10f036b9
md"## Reduce"

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
	md"Lower bound complexity with ``p`` processes if each ``x_i`` has length $n$ bytes and the arithmetic complexity is ``\gamma`` ?",
	md"""Lower bound : ``\log_2(p) (\alpha + \beta n) + \log_2(p) \gamma n`` using *spanning tree* algorithm:

First communication (2 → 1 and 4 → 3 at the same time):

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1 + x_2`` |   | ``x_3 + x_4``  |   |

Then second communication (3 → 1)
	"""
)

# ╔═╡ 4fdb4cd6-a794-4b14-84b0-72f484c6ea86
md"## All gather"

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
Let the size of ``x_i`` be ``n/p`` bytes.

1. `MPI_Gather` has complexity ``\log_2(p)\alpha + \beta n``
2. `MPI_Bcast` acts on the concatenation ``x_:`` which has length ``n`` bytes so the complexity is  ``\log_2(p) (\alpha + \beta n)``

In total, we have the complexity ``\log_2(p) (\alpha + \beta n)``. Can we do better ?

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
md"## Reduce scatter"

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
Let the size of each ``x_{i,j}`` be ``n/p`` bytes.

1. `MPI_Reduce` acts on the concatenation ``x_{:,j}`` which has length ``n`` bytes hence the complexity is ``\log_2(p)(\alpha + \beta n + \gamma n)``
2. `MPI_Scatter` has the same complexity as `MPI_Gather` (since it's the same but backwards in time) : ``\log_2(p) \alpha + \beta n``

In total, we have the complexity ``\log_2(p) (\alpha + \beta n + \gamma n)``. Can we do better ?

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
This is better than the approaches combining existing collectives above since we removed the ``\log_2(p)`` in front of ``\beta`` and ``\gamma``.
""",
)

# ╔═╡ 60bc118f-6795-43f9-97a2-865fd1704895
md"## Allreduce"

# ╔═╡ 0d69e94b-492a-4acc-adba-a2126b871724
vbox([
	md"""Before

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1`` | ``x_2`` | ``x_3`` | ``x_4`` |
""",
	#Div(md"` `", style = Dict("margin" => "50pt")),
	md"""After `MPI_Allreduce`

| `procid` | 1 | 2 | 3 | 4 |
|----------|---|---|---|---|
|    | ``x_1 + \cdots + x_4`` |``x_1 + \cdots + x_4`` | ``x_1 + \cdots + x_4`` | ``x_1 + \cdots + x_4`` |
""",
])

# ╔═╡ b9a9e335-1328-4c63-a213-ce21263bc201
Foldable(
	md"Can `MPI_Allreduce` be implemented by combining existing collectives ?",
	md"""
Let the size of each ``x_i`` be ``n`` bytes. `MPI_Allreduce` can be implemented either by combining `MPI_Reduce` followed by `MPI_Bcast` or `MPI_Reduce_scatter` followed by `MPI_Allgather`.
The first choice would lead to a complexity of ``\log_2(p)(\alpha + \beta n + \gamma n )``.
The second would lead to a complexity of ``\log_2(p)\alpha + \beta n + \gamma n``.
This second approach is faster for large ``p`` since we removed ``\log_2(p)`` in front of ``\beta`` and ``\gamma``.
""",
)

# ╔═╡ a1b2d090-d498-4d5d-90a0-8cdc648dc833
md"# Distributed sum"

# ╔═╡ a771f33f-7ed1-41aa-bee0-c215729a8c8d
md"## Distributed vector"

# ╔═╡ 141d162c-c817-498f-be16-f1cd35d82487
Foldable(md"How to collect the partial sums ?", md"`MPI_Reduce`")

# ╔═╡ 7cf59087-efca-4f03-90dc-f2acefdcbc8a
md"## Let's try it"

# ╔═╡ 4788d8b4-2efa-4489-80c3-71f405513644
md"`num_processes` = $(@bind sum_num_processes Slider(2:8, default = 2, show_value = true))"

# ╔═╡ e832ce25-94e2-4743-854d-02b52cc7b56d
aside(Foldable(md"Why is it the first process that gets the sum ?", md"We gave 0 to the 6th argument of `MPI_Reduce`, this decides which node gets the sum."), v_offset = -100)

# ╔═╡ 79b405a5-54b5-4727-a0cd-b79522ad109f
md"# Point-to-point"

# ╔═╡ d2104fbd-ba22-4501-b03a-8809271d598b
md"## Blocking communication"

# ╔═╡ 4569aa05-9963-4976-ac63-caf3f3979e83
md"""
Blocking send/received with `MPI_Send` and `MPI_Recv`.

The network cannot buffer the whole message (unless it is short). The sender need to wait for the receiver to be ready and then transfer its copy of the data.
"""

# ╔═╡ 34a10003-2c32-4332-b3e6-ce70eec0cbbe
md"## Example"

# ╔═╡ d7e31ced-4eb2-4221-b83f-462e8f32fe89
aside(Foldable(md"Is this timing bandwith accurately ?",
md"No, the time also includes the time that process 0 has to wait until process 1 is ready to start receiving. If the message is too small, it will just buffer the message and `MPI_Send` could return before the other process even reached `MPI_Recv`, see next slide."
), v_offset = -500)

# ╔═╡ c3c848ff-526a-450d-9b1c-5d9d3ccccf28
md"## Eager vs rendezvous protocol"

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

# ╔═╡ 3a50ca06-06e8-4a61-ade2-afbfc52ca655
aside(md"""See $(citepara("Section 4.1.4.2"))""", v_offset = -100)

# ╔═╡ 32f740e7-9338-4c42-8eaf-ce8022412c50
md"## Nonblocking communication"

# ╔═╡ 93f0c63c-b597-4f89-809c-7af0476f319a
md"""
`MPI_Isend` and `MPI_Irecv` where `I` stands for `immediate` or `incomplete`.
`MPI_Wait` can be used to wait for the send and receive to finish.
"""

# ╔═╡ 568057f5-b0b8-4225-8e4b-5eec911a52ef
md"## Example"

# ╔═╡ a79c410a-bebf-434c-9730-568e0ff4f4c7
md"# Consortium des Équipements de Calcul Intensif (CÉCI)"

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
md"## Launching a job"

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
md"## Slurm"

# ╔═╡ b540d5e3-6686-479a-b2c7-c1f65b85b6ba
md"## Profiling with NVIDIA Nsight Systems"

# ╔═╡ 9a100ccf-1ad3-4d2c-bbe0-e297969eb69e
md"# Topology"

# ╔═╡ 921b5a18-0733-4032-a543-9d60e254b1b2
md"""
* Specializing on topology is important for communication libraries like MPI/NCCL. For instance, Deepseek-V3 by-passed NCCL and used PTX directly to hardcode how their hardware should be used.
* Specified in [Slurm's `topology.conf` file](https://slurm.schedmd.com/topology.conf.html).
* Source : $(citeintro("Section 2.7"))
"""

# ╔═╡ 9612a1ef-fd3a-4a58-87b0-b2255ac86331
md"## Graph diameter"

# ╔═╡ 98392c40-6542-4a26-8552-c0960bbaa6a6
md"""
* Consider graph ``G`` with nodes ``v`` corresponding to computer nodes or switches.
* There is an edge ``(u, v) \in E`` if there is an ethernet cable **directly** connecting ``u`` and ``v``.
*  ``e \in E`` are ethernet cables of bandwidth ``w_e``
* Distance (unweighted) from node ``u \in V`` to node ``v \in V`` is ``d(G, u, v)``
  - Does not depend on bandwidth ``w_e`` of edges of the path
"""

# ╔═╡ c253bb24-ad76-4b58-8dfc-7dc2576e3db5
md"## Bisection bandwidth"

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
md"## Linear array"

# ╔═╡ 3dc860be-016d-49ee-8535-7d9457c70f85
Foldable(md"What is the graph diameter ?", md"``|V| - 1`` if ``u`` and ``v`` are extreme points of the array")

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
md"## Rings"

# ╔═╡ fc705b81-7310-44cc-ad9f-dc2cf8a9b645
path(true)

# ╔═╡ 86394e1c-0ff4-449a-8940-4b5906d8b6f0
Foldable(md"What is the graph diameter ?", md"``|V|/2``")

# ╔═╡ d7117a24-aba6-4479-a40e-5005310a6b38
aside(citeintro("Section 2.7.3"), v_offset = -150)

# ╔═╡ 2257220c-6f0e-4edf-9fea-7e388b84df9b
md"## Multidimensional array and torus"

# ╔═╡ 2e4dc3f9-a132-444f-a35d-f583823a7dfd
Foldable(md"What is the graph diameter of a ``n \times n`` 2D array ?",
md"""
It is ``2(n-1)``, attained for opposite vertices of the square.
$(Foldable(md"What is the graph diameter of a ``n^d`` ``d``D array ?",
md"It is ``d(n-1)``, attained for opposite vertices of the hypercube."))
""")

# ╔═╡ 8f46daf1-9ca2-4a08-99aa-4ed68af218b8
aside(citeintro("Section 2.7.4"), v_offset = -150)

# ╔═╡ 2c84bd84-b54d-4594-b9f8-35db2124d7e8
md"## Hypercube"

# ╔═╡ 4309dc43-aeb8-4ec7-94fe-0e320b784349
md"Special case of multidimensional array"

# ╔═╡ a0566fdb-a08d-4bcf-9b2f-ed211c9f111f
aside(citeintro("Section 2.7.5"), v_offset = -150)

# ╔═╡ e796b093-9c1d-4656-9acb-918de53f7e4d
md"## Crossbar"

# ╔═╡ d04b9af5-f004-4ca4-b1c9-2c86d46cb37d
md"""
* Each node input is a row and each node output is a column; [source of figure below](https://www.sciencedirect.com/topics/computer-science/crossbar-network).
* Each intersection is a switch. The cases (a) and (c) represent conflicting cases where two inputs want to simultaneously communicate with the same output.
"""

# ╔═╡ 97d3cf3f-ddac-4850-8b05-bdc0c4741f61
Foldable(md"What are the number of switches, edges, graph diameter and bisection width for ``n`` computer nodes ?",
md"""
* There are ``n^2`` switches one per intersection. This makes this architecture only suitable for small ``n``.
* The number of edges is : ``|E| = 2n^2`` which consists of ``n`` connections from an input to a switch, ``n`` connections from a switch to an output and ``2n(n-1)`` connections between switches.
* The diameter 2 if we don't count the in-between switches or ``2n`` if we coun't them.
* The bisection width is ``n/2``.
""")

# ╔═╡ 61af27f1-9f83-42f1-a419-06d12ea62133
aside(citeintro("Section 2.7.6.1"), v_offset = -200)

# ╔═╡ 143dca7c-f9a4-472a-a4bc-4578e4e8413b
md"## Tree"

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
md"## Fat-tree"

# ╔═╡ b53ec488-ff25-4647-ab00-fbf90963a795
md"""
*blocking factor* : Ratio between upper links and lower links. Ratio is 1 for fat-tree to prevent bottlenecks if all nodes start communicating.
"""

# ╔═╡ de72d596-0daf-4629-bbb5-20bb8a67cbed
Foldable(md"What is the number of edges ? What is the bisection width ?",
md"""
Number of edges is ``n\log_2(n)`` and bisection width is ``n/2``.
""")

# ╔═╡ 488b0c17-4f0f-43bf-a16c-b9faa7ae0595
aside(citeintro("Section 2.7.6.3"), v_offset = -150)

# ╔═╡ 10a1b3a7-21c7-4f97-93e1-006ad3aea40d
md"## Butterfly"

# ╔═╡ f7f097cb-d7bd-49eb-a030-ac26f8f61a67
md"Fat-tree need large switches, alternative is butterfly network:"

# ╔═╡ 6041a909-d26c-4ab1-836b-29953c578759
Foldable(md"What is the number of edges ? What is the bisection width ?",
md"""
Same as fat-tree.
""")

# ╔═╡ 16f8d28b-f201-4fe5-8446-68d7d9ddfb3c
aside(citeintro("Section 2.7.6.2"), v_offset = -250)

# ╔═╡ a59db59c-d34e-4abd-8865-9907607e06a8
aside(md"""From $(citeintro("Figure 2.27"))""", v_offset = -200)

# ╔═╡ f2417047-33fc-4489-8e89-115bc6b46c13
aside(md"""From $(citeintro("Figure 2.30"))""", v_offset = -200)

# ╔═╡ 7565e3da-84ce-42b6-8d4b-3615576f33b7
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


# ╔═╡ c04bcc96-e5fe-4d6e-a12e-40dcde58c62e
md"""
* MPI $(img("https://avatars.githubusercontent.com/u/14836989", name = "MPI.png", :height => "20pt")) is an open standard for distributed computing
* [Many implementations](https://www.mpi-forum.org/implementation-status/):
  - MPICH, from $(img("https://upload.wikimedia.org/wikipedia/commons/6/65/ArgonneLaboratoryLogo.png", :height => "20pt")) and $(img("https://upload.wikimedia.org/wikipedia/commons/6/69/Mississippi_State_University_logo.svg", :height => "20pt"))
  - Open MPI $(img("https://upload.wikimedia.org/wikipedia/commons/6/6f/Open_MPI_logo.png", :height => "20pt")) (not to be confused with $(img("https://upload.wikimedia.org/wikipedia/commons/e/eb/OpenMP_logo.png", :width => "45pt")))
  - commercial implementations from $(img("https://upload.wikimedia.org/wikipedia/commons/4/46/Hewlett_Packard_Enterprise_logo.svg", :height => "20pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/6/6a/Intel_logo_%282020%2C_dark_blue%29.svg", :height => "15pt")), $(img("https://upload.wikimedia.org/wikipedia/commons/9/96/Microsoft_logo_%282012%29.svg", :height => "15pt")), and $(img("https://upload.wikimedia.org/wikipedia/commons/9/96/NEC_logo.svg", :height => "15pt"))
"""

# ╔═╡ 4e32f7fb-cd5a-4190-9c92-ba4029313475
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/mpi-node2.png")

# ╔═╡ b94cd399-0370-49e9-a522-056f3af22955
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/collectives.jpg")

# ╔═╡ 370f0f20-e373-4028-bca1-83e93678cbcb
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/mpi-array.png")

# ╔═╡ 0e640e07-82c7-4dab-a8f1-2f634bbebdea
hbox([
	img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-ideal.png", :height => "150pt"),
	img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-blocking.png", :height => "160pt"),
])

# ╔═╡ 8a527c17-bf2b-4e6b-937f-ef3a269c5112
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming/refs/heads/main/booksources/graphics/send-nonblocking.jpeg", :height => "200pt")

# ╔═╡ 39f48c25-6efb-4ff2-aedc-9d3e722dad24
md"""
* [Follow README instructions to create an account and setup your computer](https://github.com/blegat/LINMA2710?tab=readme-ov-file#ceci-cluster)
  - Don't wait the last minute, if you get into trouble it's easier to get this setup before you actually need it
* Select $(img("https://www.ceci-hpc.be/assets/img/new_ceci_logo.png", :height => "15pt")) cluster from [the list](https://www.ceci-hpc.be/clusters.html) + `manneback` for GPU. You only have access to Tier-2 clusters. This sadly leaves out:
  - Tier-1 clusters such as Lucia
  - Tier-0 cluster such as $(img("https://www.lumi-supercomputer.eu/wp-content/uploads/2020/02/lumi_logo.png", :height => "15pt")) from $(img("https://upload.wikimedia.org/wikipedia/commons/8/8f/HPC_JU_logo_RGB.svg", :height => "20pt"))
* Connect with SSH using `ssh lemaitre4` or `ssh manneback`.
"""

# ╔═╡ 55e96151-2aa1-4ea0-b672-2038c57d911e
aside(img("https://upload.wikimedia.org/wikipedia/en/3/3e/The_LUMI_supercomputer.jpg", :height => "100pt"), v_offset = -140)

# ╔═╡ be0e3ba0-18cc-4b9a-a56d-2566f5148fae
md"""## $(img("https://github.com/TACC/Lmod/raw/main/logos/2x/Lmod-4color%402x.png", :height => "30px"))"""

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

# ╔═╡ 091dd042-580b-4fda-8086-e048663aed6c
md"""
* NVIDIA Nsight Systems $(img("https://developer.download.nvidia.com/images/nvidia-nsight-systems-icon-gbp-shaded-256.png", :width => "20pt")) can profile CUDA code but also MPI
* Available on `manneback` after loading `CUDA` with $(img("https://github.com/TACC/Lmod/raw/main/logos/2x/Lmod-4color%402x.png", :height => "20px"))

```sh
[laptop]$ ssh manneback
[blegat@mbackf1 ~]$ nsys
-bash: nsys: command not found
[blegat@mbackf1 ~]$ ml CUDA
[blegat@mbackf1 ~]$ nsys
```
"""

# ╔═╡ 7fc70992-973a-43c6-904a-dd1b622a5ed8
Foldable(md"What is the bisection width ?", md"""
The bisection width is 1 : $(img("https://upload.wikimedia.org/wikipedia/commons/7/79/Bisected_linear_array.jpg", :width => "300pt"))
""")

# ╔═╡ 23bfbe95-7ba2-41b9-bd8b-dc4baa3ad53a
Foldable(md"What is the bisection width ?", md"""
The bisection width is 2:
$(img("https://upload.wikimedia.org/wikipedia/commons/5/51/Bisected_ring.jpg", :width => "300pt"))
""")

# ╔═╡ 39b055f5-3dbf-403c-b21e-210e3813d8b0
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/torus.jpeg")

# ╔═╡ b68eb860-a5b4-4e9e-9fbf-6eb6ce43ae69
Foldable(md"What is the bisection width of a ``n \times n`` 2D array ?",
md"""
It is ``n = \sqrt{|V|}``:
$(img("https://upload.wikimedia.org/wikipedia/commons/2/2f/Bisected_mesh.jpg", :width => "300pt"))
$(Foldable(md"What is the bisection width of a ``n^d`` ``d``D array ?",
md"It is 1 for ``d = 1``, ``n`` for ``d = 2`` and ``n^2`` for ``d = 3``. In general, it is ``n^{d-1} = |V|^{(d-1)/d}``"))
""")

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

# ╔═╡ 655e980d-b4e9-4f56-a5ae-380072242d27
hbox([
img("https://ars.els-cdn.com/content/image/3-s2.0-B9781558608528500043-f01-09-9781558608528.jpg"),
img("https://ars.els-cdn.com/content/image/3-s2.0-B9781558608528500043-f01-10-9781558608528.jpg"),
])

# ╔═╡ e4d1de1d-d57a-48ab-ad7a-c09b427daa03
Foldable(md"What is the diameter and bisection width of ``n`` computer nodes ?",
md"""
Diameter is ``2\log_2(n)`` and bisection width is 1.
$(img("https://upload.wikimedia.org/wikipedia/commons/d/da/Bisected_tree.jpg") )
""")

# ╔═╡ 4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/fattree5.jpg", :width => "500pt")

# ╔═╡ 3ec3c058-a94d-4717-b99f-66373f2fa31d
img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/butterflys.jpeg")

# ╔═╡ 1152dec8-3810-42b1-bb2a-8755dcaef56c
img1(f, args...) = img("https://raw.githubusercontent.com/VictorEijkhout/TheArtOfHPC_vol1_scientificcomputing/refs/heads/main/booksources/graphics/$f", args...)

# ╔═╡ 133f4c7d-33e0-4e13-b716-f538125436ca
TwoColumnWideLeft(
md"""
There can be ``n`` simultaneous communication at the same time, provided that each input communicate with a different output.
The figure on the right provides an example of such non-conflicting communication with the black dots indicating that the input of that row communicates to the corresponding output (case (a) of above figure). The switch at row 1 and column 2 is just propagating the input data horizontally and output data vertically (case (b) of above figure). The switch at row 0 and column 5 is receiving no data.
""",
img1("crossbar.jpg"),
)

# ╔═╡ c45ff9b5-35d9-4a9d-a801-c762333a1f02
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

# ╔═╡ b0ca0392-71b8-4f44-8c6c-0978a02a0e6c
compile_and_run(Example("MPI/procname.c"); mpi = true, verbose = 1, show_run_command = true, num_processes = procname_num_processes)

# ╔═╡ 35aa1295-642f-4525-bf19-df2a42ff39d6
compile_and_run(Example("MPI/mpi_sum.c"), mpi = true, num_processes = sum_num_processes, verbose = 1)

# ╔═╡ ce7bf747-7116-4e76-9004-f234317046c3
compile_and_run(Example("MPI/mpi_bench1.c"), mpi = true, num_processes = 2)

# ╔═╡ 26aa369f-e5c7-4fe5-8b6b-903f4f4e91ba
compile_and_run(Example("MPI/mpi_bench2.c"), mpi = true, num_processes = 2)

# ╔═╡ e172f5c5-8b96-4efd-9cf3-805c58d1a6a3
function definition(name, content)
    return Markdown.MD(Markdown.Admonition("key-concept", "Def: $name", [content]))
end

# ╔═╡ 49b596b8-891d-4f3f-a6a4-a62cc8237df3
definition("Graph diameter", md"*Graph diameter* is ``d(G) := \max_{u, v \in V} d(G, u, v)``")

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
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

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

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

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

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

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

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
# ╟─58e12afd-6eb0-4731-bd57-d9ae7ab4e164
# ╟─bfab5c2d-61c3-468b-9ddf-4aaa49cb7785
# ╟─063f0acc-c023-46d0-9ed9-fbd7fbdcfa3b
# ╟─3e98c0ca-1b47-4631-83d7-cd0c8c0a431d
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
# ╟─d7117a24-aba6-4479-a40e-5005310a6b38
# ╟─2257220c-6f0e-4edf-9fea-7e388b84df9b
# ╟─39b055f5-3dbf-403c-b21e-210e3813d8b0
# ╟─2e4dc3f9-a132-444f-a35d-f583823a7dfd
# ╟─b68eb860-a5b4-4e9e-9fbf-6eb6ce43ae69
# ╟─8f46daf1-9ca2-4a08-99aa-4ed68af218b8
# ╟─2c84bd84-b54d-4594-b9f8-35db2124d7e8
# ╟─4309dc43-aeb8-4ec7-94fe-0e320b784349
# ╟─f6f9447c-9bc9-432d-bd80-2c39f9d842f8
# ╟─1551122c-70ae-4e37-b3fb-4be91fcc4afb
# ╟─a0566fdb-a08d-4bcf-9b2f-ed211c9f111f
# ╟─e796b093-9c1d-4656-9acb-918de53f7e4d
# ╟─d04b9af5-f004-4ca4-b1c9-2c86d46cb37d
# ╟─655e980d-b4e9-4f56-a5ae-380072242d27
# ╟─133f4c7d-33e0-4e13-b716-f538125436ca
# ╟─97d3cf3f-ddac-4850-8b05-bdc0c4741f61
# ╟─61af27f1-9f83-42f1-a419-06d12ea62133
# ╟─143dca7c-f9a4-472a-a4bc-4578e4e8413b
# ╟─1bac238f-79c8-4f9f-a187-bacb288de3b0
# ╟─e4d1de1d-d57a-48ab-ad7a-c09b427daa03
# ╟─954f1ab1-1e2f-458b-96d7-a1746631fac7
# ╟─21d507f6-02f8-4f8b-84f1-bcb84731df66
# ╟─4aac6ab5-053a-4f60-9e2e-e8d61ff0cecb
# ╟─b53ec488-ff25-4647-ab00-fbf90963a795
# ╟─de72d596-0daf-4629-bbb5-20bb8a67cbed
# ╟─488b0c17-4f0f-43bf-a16c-b9faa7ae0595
# ╟─10a1b3a7-21c7-4f97-93e1-006ad3aea40d
# ╟─f7f097cb-d7bd-49eb-a030-ac26f8f61a67
# ╟─3ec3c058-a94d-4717-b99f-66373f2fa31d
# ╟─6041a909-d26c-4ab1-836b-29953c578759
# ╟─16f8d28b-f201-4fe5-8446-68d7d9ddfb3c
# ╟─a59db59c-d34e-4abd-8865-9907607e06a8
# ╟─f2417047-33fc-4489-8e89-115bc6b46c13
# ╠═8df4ff2f-d176-4b4e-a525-665b5d07ea52
# ╟─1152dec8-3810-42b1-bb2a-8755dcaef56c
# ╟─7565e3da-84ce-42b6-8d4b-3615576f33b7
# ╟─c45ff9b5-35d9-4a9d-a801-c762333a1f02
# ╟─e172f5c5-8b96-4efd-9cf3-805c58d1a6a3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
