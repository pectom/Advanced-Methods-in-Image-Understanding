### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 2cebc58a-7e9b-4f67-9756-420252a347f4
begin
	using Random
	using TestImages
	using Images
	using ImageTransformations
	using Plots
	using DataStructures
end

# ╔═╡ 9eed97b4-c720-4852-9529-0572b112eb0f
md"# Convolution

Authors: Paweł Renc, Tomasz Pęcak"

# ╔═╡ 2f2b5dcf-3af7-4701-8b67-bc3b6a5bc97b
md"
Convolution is a simple mathematical operation which is fundamental to many common image processing operators. Convolution provides a way of `multiplying together' two arrays of numbers, generally of different sizes, but of the same dimensionality, to produce a third array of numbers of the same dimensionality.
"

# ╔═╡ 214a122d-2cae-427b-a7bb-ec0bd1e734f5
function conv(input, filter)
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)

    result = zeros(input_r-filter_r+1, input_c-filter_c+1)
    result_r, result_c = size(result)
  
    return [
		sum(input[i:i+filter_r-1, j:j+filter_c-1] .* filter)
		for i in 1:result_r, j in 1:result_c
	]
end;

# ╔═╡ 96a28f29-9644-410e-8a13-a2151c74b35c
t = rand((1:10), 5, 5)

# ╔═╡ 64ba6340-7aa3-4802-a93f-3721d9d2ccc5
filter = [
	1  0 -1;
	1  0 -1;
	1  0 -1;
]

# ╔═╡ c93ace2e-72d3-4a7b-8d6e-5122d7e06a39
el = t[1:3, 1:3]

# ╔═╡ 61214f50-6979-42ca-ba17-a865dc89228c
el .* filter

# ╔═╡ e141326b-de33-4da3-ae52-7d3e88ecd343
sum(el .* filter)

# ╔═╡ 00970981-efb2-4c19-b9b8-110682b7a413
conv(t, filter)

# ╔═╡ f131681d-fa1a-4151-88cf-51522489ade5
function padding_conv(input, filter)
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)


	pad_r = (filter_r - 1) ÷ 2
	pad_c = (filter_c - 1) ÷ 2

	input_padded = zeros(input_r+(2*pad_r), input_c+(2*pad_c))
	for i in 1:input_r, j in 1:input_c
		input_padded[i+pad_r, j+pad_c] = input[i, j]
	end
	input = input_padded
	input_r, input_c = size(input)

    result = zeros(input_r-filter_r+1, input_c-filter_c+1)
    result_r, result_c = size(result)

    for i in 1:result_r, j in 1:result_c, k in 1:filter_r, l in 1:filter_c 
		result[i,j] += input[i+k-1,j+l-1]*filter[k,l]
    end

    return result
end

# ╔═╡ b1748591-bc44-449a-8a11-c8d1d79922ae
padding_conv(t, filter)

# ╔═╡ d6a4655f-f400-48a7-a3ab-87c11d56b9e1
function stride_conv(input, filter; stride=2)
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)

    result = zeros((input_r-filter_r) ÷ stride + 1, (input_c-filter_c) ÷ stride + 1)
    result_r, result_c = size(result)

    ir = 1 
    ic = 1
    for i in 1:result_r
        for j in 1:result_c
            for k in 1:filter_r, l in 1:filter_c 
				result[i,j] += input[ir+k-1,ic+l-1]*filter[k,l]
            end
            ic += stride
        end
        ir += stride 
        ic = 1
    end

    return result
end;

# ╔═╡ 73404de2-6243-4fbb-af33-6772c282c10d
stride_conv(t, filter; stride=1)

# ╔═╡ e800c3f0-363a-458e-9353-4bec44504c55
stride_conv(t, filter; stride=2)

# ╔═╡ 65e865bd-8887-48c2-90ce-5e0e80e1b399
filters = OrderedDict(
	"edge_detection_1" => [
		 1  0 -1;
		 0  0  0;
		-1  0  1;
	],
	"edge_detection_2" => [
		 0 -1  0;
		-1  4 -1;
		 0 -1  0;
	],
	"edge_detection_3" => [
		-1  -1 -1;
		-1   8 -1;
		-1  -1 -1;
	],
	"sharpen" => [
		 0 -1  0;
		-1  5 -1;
		 0 -1  0;
	],
	"box_blur" => [
		 1  1  1;
		 1  1  1;
		 1  1  1;
	] / 9,
	"gaussian_blur_3x3" => [
		 1  2  1;
		 2  4  2;
		 1  2  1;
	] / 16,
	"gaussian_blur_5x5" => [
		 1  4  6  4 1;
		 4 16 24 16 4;
		 6 24 36 24 6;
		 4 16 24 16 4;
		 1  4  6  4 1;
	] / 256,
	"unsharp_masking_5x5" => [
		 1  4    6  4 1;
		 4 16   24 16 4;
		 6 24 -476 24 6;
		 4 16   24 16 4;
		 1  4    6  4 1;
	] / -256,
);

# ╔═╡ 3fec8c1b-9a0b-4ff2-8e5f-6b1c6641f1ce
function apply_filter(img, filter; stride=1)
	
	cv_img = channelview(img)
		
	new_img = cat(
		[stride_conv(cv_img[i, :, :], filter; stride=stride) for i in 1:3]...,
		dims=3
	)
	
	colorview(RGB, permuteddimsview(new_img, (3, 1, 2)))
end;

# ╔═╡ cd2e4894-8fc3-4554-8681-e6c8cd201173
function compare_filters(img, filter_substr)
	strides = [1, 2, 5]
	test_filters = [
		filter_name
		for filter_name in keys(filters)
		if occursin(filter_substr, filter_name)
	]
	
	imgs = [
		apply_filter(img, filters[filter_name]; stride=stride)
		for filter_name in test_filters, stride in strides
	]
	
	ylabels = ["stride=$i" for i in strides]
	layout_x = length(ylabels)
	layout_y = length(test_filters)
	
	plot(
		[
			plot(
				im, 
				xticks = nothing,
				yticks = nothing,
				title = i <= layout_y ? (test_filters[i]) : "",
				ylabel = i % layout_y == 1 ? ylabels[i ÷ layout_y + 1] : ""
			) for (i, im) in enumerate(imgs)
		]...,
		layout=(layout_x, layout_y)
	)
end;

# ╔═╡ 5332db06-5159-45b2-8d68-9b0f8a2b2680
md"## Results
Masks were taken from [wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Details:~:text=effects.-,Operation).
"

# ╔═╡ 038e838b-85f4-4f2b-a97a-1b81741c981a
compare_filters(testimage("lena"), r"edge")

# ╔═╡ 39e6027f-6b7f-4d70-abec-4a4bc2a1dde9
compare_filters(testimage("fabio"), r"blur")

# ╔═╡ d12cc2ba-2f0f-4d4e-a4a8-9a6c649bf5c2
compare_filters(testimage("fabio"), r"sharp")

# ╔═╡ Cell order:
# ╟─9eed97b4-c720-4852-9529-0572b112eb0f
# ╟─2cebc58a-7e9b-4f67-9756-420252a347f4
# ╟─2f2b5dcf-3af7-4701-8b67-bc3b6a5bc97b
# ╠═214a122d-2cae-427b-a7bb-ec0bd1e734f5
# ╟─96a28f29-9644-410e-8a13-a2151c74b35c
# ╟─64ba6340-7aa3-4802-a93f-3721d9d2ccc5
# ╠═c93ace2e-72d3-4a7b-8d6e-5122d7e06a39
# ╠═61214f50-6979-42ca-ba17-a865dc89228c
# ╠═e141326b-de33-4da3-ae52-7d3e88ecd343
# ╠═00970981-efb2-4c19-b9b8-110682b7a413
# ╠═f131681d-fa1a-4151-88cf-51522489ade5
# ╠═b1748591-bc44-449a-8a11-c8d1d79922ae
# ╠═d6a4655f-f400-48a7-a3ab-87c11d56b9e1
# ╠═73404de2-6243-4fbb-af33-6772c282c10d
# ╠═e800c3f0-363a-458e-9353-4bec44504c55
# ╠═65e865bd-8887-48c2-90ce-5e0e80e1b399
# ╠═3fec8c1b-9a0b-4ff2-8e5f-6b1c6641f1ce
# ╟─cd2e4894-8fc3-4554-8681-e6c8cd201173
# ╟─5332db06-5159-45b2-8d68-9b0f8a2b2680
# ╟─038e838b-85f4-4f2b-a97a-1b81741c981a
# ╟─39e6027f-6b7f-4d70-abec-4a4bc2a1dde9
# ╟─d12cc2ba-2f0f-4d4e-a4a8-9a6c649bf5c2
