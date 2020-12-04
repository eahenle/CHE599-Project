### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 0e09d780-35b9-11eb-34c7-a7ff2d86d982
using CSV, DataFrames

# ╔═╡ c61fe1e0-35d0-11eb-0121-7bf0c4660d01
using ScikitLearn

# ╔═╡ a0953640-35b8-11eb-1357-37314995bbd0
md"
# Annealed Steel
"

# ╔═╡ 5c246a30-35c2-11eb-3ee2-e70dfe2b95f9
md"
`anneal.names` is horribly formatted!  Extract column names:
"

# ╔═╡ e232f93e-35bb-11eb-30d2-8fb4b187ada5
begin
	namesfile = open("anneal.names")
	namesfiledata = readlines(namesfile)
	close(namesfile)
	# get ids of lines w/ names
	attr_idx = [occursin(r"    ?[0-9]?[0-9]. [a-z]", line) for line ∈ namesfiledata]
	names = Symbol[]
	# loop over attribute lines
	for line in namesfiledata[attr_idx]
		# split on whitespace, drop ""
		nowhitespace = filter(ss -> ss ≠ "", split(line, " "))
		# get id of token containing name
		name_id = [occursin(r"[a-z]*-*_*[a-z]*:", substr) for substr ∈ nowhitespace]
		# split name from rest of line
		name = split(String(nowhitespace[name_id][1]), ":")[1]
		# add name to array
		push!(names, Symbol(name))
	end
	names
end

# ╔═╡ e9ba3a90-35b9-11eb-3c82-8dc71624c603
classes = ["1", "2", "3", "4", "5", "U"]

# ╔═╡ 27455850-35b9-11eb-29cf-1f97fb339e66
df_train = CSV.read("anneal.data", DataFrame, header=0)

# ╔═╡ b77f525e-35cb-11eb-1bb8-8f2fabb48df0
names!(df_train, [names..., :class])

# ╔═╡ 415714e0-35b9-11eb-39db-338b9ad54699
df_test = CSV.read("anneal.test", DataFrame, header=0)

# ╔═╡ bf8a3e6e-35d0-11eb-3076-212d47a44b25
names!(df_test, [names..., :class])

# ╔═╡ d2a2b3c0-35d0-11eb-24f7-495a3459c779
@sk_import preprocessing : OneHotEncoder

# ╔═╡ d2b50340-35d0-11eb-3d85-2dc942f49a2b
one_hot_encoder = OneHotEncoder(drop="first", sparse=false)

# ╔═╡ 478fde60-35d1-11eb-0f84-3f8344536cc0
# peel off all String-type columns
str_cols = [names[i] for (i,col) in enumerate(eachcol(df_train)) if typeof(col[1]) == String && i < 39]

# ╔═╡ 0f419560-35d3-11eb-2771-c3ca6ea4c912
df_train[str_cols]

# ╔═╡ 7c0f04b0-35d4-11eb-0489-1146ca65eb13
str_mat = convert(Matrix, df_train[str_cols])

# ╔═╡ 64cc34ae-35d6-11eb-1898-313acd5364fd
unique(df_train[:,2])

# ╔═╡ d2c99cb0-35d0-11eb-0a13-9bb4cf692fe5
one_hot_encoded = one_hot_encoder.fit_transform(str_mat)

# ╔═╡ d2f03780-35d0-11eb-0d06-d9a99e641d5f
begin
	converted = convert(DataFrame, one_hot_encoded)
	names!(converted, [convert(Symbol, name) for name in one_hot_encoder.get_feature_names()])
end

# ╔═╡ d313c510-35d0-11eb-28e4-9de2f998c526
clean_training_data = hcat(df_train[[col for col ∈ names if !(col ∈ str_cols)]], converted)

# ╔═╡ d32689c0-35d0-11eb-3e99-0ba92d7041b5
@sk_import decomposition : PCA

# ╔═╡ d33a11c0-35d0-11eb-1aa8-37175b05bac1
pca = PCA()

# ╔═╡ b578e0c0-35db-11eb-36c5-ef3331dffbed
pca.fit(convert(Matrix, clean_training_data))

# ╔═╡ cc4f3680-35dd-11eb-279b-b13dbd9ab57b
col1 = pca.transform(convert(Matrix, clean_training_data))[:,2]

# ╔═╡ 3ae86c82-35dc-11eb-3f2d-4db915c25e8a
begin
	using PyPlot
	figure()
	hist(col1, bins=100)
	gcf()
end

# ╔═╡ d7f241a0-35db-11eb-3419-5b185359c3ac
pca.explained_variance_ / sum(pca.explained_variance_)

# ╔═╡ Cell order:
# ╟─a0953640-35b8-11eb-1357-37314995bbd0
# ╠═0e09d780-35b9-11eb-34c7-a7ff2d86d982
# ╟─5c246a30-35c2-11eb-3ee2-e70dfe2b95f9
# ╠═e232f93e-35bb-11eb-30d2-8fb4b187ada5
# ╠═e9ba3a90-35b9-11eb-3c82-8dc71624c603
# ╠═27455850-35b9-11eb-29cf-1f97fb339e66
# ╠═b77f525e-35cb-11eb-1bb8-8f2fabb48df0
# ╠═415714e0-35b9-11eb-39db-338b9ad54699
# ╠═bf8a3e6e-35d0-11eb-3076-212d47a44b25
# ╠═c61fe1e0-35d0-11eb-0121-7bf0c4660d01
# ╠═d2a2b3c0-35d0-11eb-24f7-495a3459c779
# ╠═d2b50340-35d0-11eb-3d85-2dc942f49a2b
# ╠═478fde60-35d1-11eb-0f84-3f8344536cc0
# ╠═0f419560-35d3-11eb-2771-c3ca6ea4c912
# ╠═7c0f04b0-35d4-11eb-0489-1146ca65eb13
# ╠═64cc34ae-35d6-11eb-1898-313acd5364fd
# ╠═d2c99cb0-35d0-11eb-0a13-9bb4cf692fe5
# ╠═d2f03780-35d0-11eb-0d06-d9a99e641d5f
# ╠═d313c510-35d0-11eb-28e4-9de2f998c526
# ╠═d32689c0-35d0-11eb-3e99-0ba92d7041b5
# ╠═d33a11c0-35d0-11eb-1aa8-37175b05bac1
# ╠═b578e0c0-35db-11eb-36c5-ef3331dffbed
# ╠═cc4f3680-35dd-11eb-279b-b13dbd9ab57b
# ╠═d7f241a0-35db-11eb-3419-5b185359c3ac
# ╠═3ae86c82-35dc-11eb-3f2d-4db915c25e8a
