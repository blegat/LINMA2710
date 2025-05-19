import PlutoPDF

const enable_present = """`
	<script>
    window.present()
	</script>
`"""

function generate_pdfs(dir)
    tmp, _ = mktemp()
    for filename in readdir(dir)
        if endswith(filename, ".jl")
            PlutoPDF.pluto_to_pdf(joinpath(dir, filename); generate_html_options = (; preamble_html_js = enable_present, PlutoPDF.generate_html_default_options...))
            script = joinpath(@__DIR__, "remove_blank.sh")
            pdf = joinpath(dir, filename[1:end-2] * "pdf")
            mv(pdf, tmp)
            run(`sh $script $tmp $pdf`)
        end
    end
end

generate_pdfs(joinpath(dirname(joinpath(@__DIR__)), "Lectures"))
