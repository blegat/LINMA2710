import DocumenterCitations

# /!\ Important: use this Zotero config with Better BibTeX:
# https://github.com/JuliaDocs/DocumenterCitations.jl/issues/85#issuecomment-2479025454
function load_biblio!(file = joinpath(@__DIR__, "biblio.bib"); style = DocumenterCitations.AlphaStyle())
    @info("Loading bibliography from `$file`...")
    biblio = DocumenterCitations.CitationBibliography(file; style)
    DocumenterCitations.init_bibliography!(style, biblio)
    @info("Loading completed.")
    return biblio
end
citation_label(biblio, key::String) = DocumenterCitations.citation_label(biblio.style, biblio.entries[key], biblio.citations)
function bibcite(biblio, keys::Vector{String})
    return "[" * join(citation_label.(Ref(biblio), keys), ", ") * "]"
end
bibcite(biblio, key::String) = bibcite(biblio, [key])
function bibcite(biblio, key::String, what)
    return "[" * citation_label(biblio, key) * "; " * what * "]"
end
function citation_reference(biblio, key::String)
    # `DocumenterCitations` writes a `+` in the label after 3 authors so we use
    # `et_al = 3` for consistency
    DocumenterCitations.format_labeled_bibliography_reference(biblio.style, biblio.entries[key], et_al = 3)
end
# Markdown creates a `<p>` surrounding it but we don't want that in some cases
_inline_markdown(m::Markdown.MD) = sprint(Markdown.htmlinline, m.content[].content)
function _print_entry(io, biblio, key; links = false, kws...)
    print(io, '[')
    print(io, citation_label(biblio, key))
    print(io, "] ")
    println(io, _inline_markdown(Markdown.parse(citation_reference(biblio, key))))
end
function bibrefs(biblio, key::String; kws...)
    io = IOBuffer()
    println(io, "<p style=\"font-size:12px\">")
    _print_entry(io, biblio, key; kws...)
    println(io, "</p>")
    return HTML(String(take!(io)))
end
function bibrefs(biblio, keys::Vector{String}; kws...)
    io = IOBuffer()
    println(io, "<p style=\"font-size:12px\">")
    for key in keys
        _print_entry(io, biblio, key; kws...)
        println(io, "<br/>")
    end
    println(io, "</p>")
    return HTML(String(take!(io)))
end
