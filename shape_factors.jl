### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 683bc4a4-9a25-11eb-0e73-dba2de566d11
begin
	using Images
end

# ╔═╡ f962e9e9-3602-4950-821e-457b10c38aa5
imgs = [
	load(joinpath("baza", filename))
	for filename in readdir("baza") if occursin(r"Segmentation", filename)
]

# ╔═╡ f011e3b5-fb06-4f5b-87e4-f0e6f616322e


# ╔═╡ Cell order:
# ╠═683bc4a4-9a25-11eb-0e73-dba2de566d11
# ╠═f962e9e9-3602-4950-821e-457b10c38aa5
# ╠═f011e3b5-fb06-4f5b-87e4-f0e6f616322e
