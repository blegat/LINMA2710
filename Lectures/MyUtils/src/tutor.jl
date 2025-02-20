import HTTP

# No need to go beyond `620` because pythontutor will just add an internal slider an not show full code
function tutor(code; width = 800, height = min(620, 260 + 20 * countlines(IOBuffer(code))), frameborder = 0, curInstr = 0)
	esc_code = HTTP.escapeuri(code)
	src = "https://pythontutor.com/iframe-embed.html#code=$esc_code&cumulative=false&py=c&curInstr=$curInstr"
	return HTML("""<iframe width="$width" height="$height" frameborder="$frameborder" src="$src"></iframe>""")
end
