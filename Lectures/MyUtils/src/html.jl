import PlutoUI

function reset_width(width)
    return HTML("""
<style>
    main {
        margin: 0 auto;
        max-width: $(width)px;
        padding-left: max(160px, 10%);
        padding-right: max(160px, 10%);
    }
</style>
""")
end

function imgpath(file)
    if !('.' in file)
        file = file * ".png"
    end
    return joinpath(joinpath(dirname(dirname(@__DIR__)), "images", file))
end
function img(file, args...)
    PlutoUI.LocalResource(imgpath(file), args...)
end

function header(title, authors)
    return HTML("<p align=center style=\"font-size: 40px;\">$title</p><p align=right><i>$authors<i></p>")
end
section(t) = md"# $t"
# with `##`, it's not centered but it works better with TableOfContents
frametitle(t) = md"## $t"

endofslides() = html"<p align=center style=\"font-size: 20px; margin-bottom: 5cm; margin-top: 5cm;\">The End</p>"

struct Join
    list
    Join(a) = new(a)
    Join(a, b, args...) = Join(tuple(a, b, args...))
end
function Base.show(io::IO, mime::MIME"text/html", d::Join)
    for el in d.list
        show(io, mime, el)
    end
end

struct HTMLTag
    tag::String
    parent
end
function Base.show(io::IO, mime::MIME"text/html", d::HTMLTag)
    write(io, "<", d.tag, ">")
    show(io, mime, d.parent)
    write(io, "</", d.tag, ">")
end

function qa(question, answer)
    return HTMLTag("details", Join(HTMLTag("summary", question), answer))
end

function _inline_html(m::Markdown.Paragraph)
    return sprint(Markdown.htmlinline, m.content)
end

function _inline_html(code::Markdown.Code)
    # `html(m)` adds an annoying `<pre>` so this is taken from the implementation of
    # `html(::IO, ::Markdown.Code)` where `withtag(io, :pre)` is removed
    maybe_lang = !isempty(code.language) ? Any[:class=>"language-$(code.language)"] : []
    return sprint(code) do io, code
        Markdown.withtag(io, :code, maybe_lang...) do
            Markdown.htmlesc(io, code.code)
        end
    end
end

function qa(question::Markdown.MD, answer)
    # `html(question)` will create `<p>` if `question.content[]` is `Markdown.Paragraph`
    # This will print the question on a new line and we don't want that:
    h = HTML(_inline_html(question.content[]))
    return qa(h, answer)
end

function wooclap(link)
    return HTML("""<img alt="Wooclap Logo" src="https://www.wooclap.com/images/wooclap-logo.svg"> <a style="margin-left: 80px;" href="https://app.wooclap.com/$link"><tt>https://app.wooclap.com/JAPRXX</tt></a>""")
end
