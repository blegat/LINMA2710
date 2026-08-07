# To sync this, clone `https://github.com/VictorEijkhout/TheArtOfHPC_vol2_parallelprogramming` the parent of the folder of this repository and then run `julia sync.jl`
function sync(book_dir)
    mpi_c = joinpath(book_dir, "examples/mpi/c")
    dest_dir = @__DIR__
    for example in [
        "procname.c",
    ]
        cp(joinpath(mpi_c, example), joinpath(dest_dir, example), force = true)
    end
end
sync(joinpath(dirname(dirname(@__DIR__)), "TheArtOfHPC_vol2_parallelprogramming"))
