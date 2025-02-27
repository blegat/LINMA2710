import Luxor, PlutoTeachingTools

function CenteredBoundedBox(str)
    xbearing, ybearing, width, height, xadvance, yadvance =
        Luxor.textextents(str)
    lcorner = Luxor.Point(xbearing - width/2, ybearing)
    ocorner = Luxor.Point(lcorner.x + width, lcorner.y + height)
    return Luxor.BoundingBox(lcorner, ocorner)
end

function boxed(str::AbstractString, p)
    Luxor.translate(p)
    Luxor.sethue("lightgrey")
    Luxor.poly(CenteredBoundedBox(str) + 5, action = :stroke, close=true)
    Luxor.sethue("black")
    Luxor.text(str, Luxor.Point(0, 0), halign=:center)
    #settext("<span font='26'>$str</span>", halign="center", markup=true)
    Luxor.origin()
end

function image_from_url(url, html_attributes...; name = split(url, '/')[end], kws...)
    path = joinpath("cache", name)
    return PlutoTeachingTools.RobustLocalResource(url, path, html_attributes...)
end

function placeimage_from_url(url, pos; scale = 1.0, centered = true, kws...)
    r = image_from_url(url; kws...)
    if r.mime isa MIME"image/svg+xml"
        img = Luxor.readsvg(path)
    else
        img = Luxor.readpng(path)
    end
    Luxor.gsave()
    Luxor.scale(scale)
    Luxor.placeimage(img, pos / scale; centered)
    Luxor.grestore() # undo `Luxor.scale`
    return
end
