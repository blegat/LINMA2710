import PlutoPDF

function generate_pdfs(dir)
    for filename in readdir(dir)
        if endswith(filename, ".jl")
            @info("Exporting pdf for $(joinpath(dir, filename))")
            output_path = PlutoPDF.pluto_to_pdf(joinpath(dir, filename))
            @info("Exported pdf at $(output_path) : $(isfile(output_path))")
        end
    end
end

generate_pdfs(joinpath(dirname(joinpath(@__DIR__)), "Lectures"))
