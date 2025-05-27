module MyUtils

using Markdown

include("biblio.jl")
include("clang.jl")
include("html.jl")
include("luxor.jl")
include("tutor.jl")
include("colors.jl")

# Taken from `JuMP/src/JuMP.jl`
const _EXCLUDE_SYMBOLS = [Symbol(@__MODULE__), :eval, :include]

for sym in names(@__MODULE__; all = true)
    sym_string = string(sym)
    if sym in _EXCLUDE_SYMBOLS ||
       startswith(sym_string, "_") ||
       startswith(sym_string, "@_")
        continue
    end
    if !(
        Base.isidentifier(sym) ||
        (startswith(sym_string, "@") && Base.isidentifier(sym_string[2:end]))
    )
        continue
    end
    @eval export $sym
end

end
