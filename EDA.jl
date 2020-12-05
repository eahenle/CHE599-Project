### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 0e09d780-35b9-11eb-34c7-a7ff2d86d982
using CSV, DataFrames

# ╔═╡ c61fe1e0-35d0-11eb-0121-7bf0c4660d01
using ScikitLearn

# ╔═╡ 29b496a0-3695-11eb-0326-0ff056416f1a
using LinearAlgebra

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

# ╔═╡ d2c99cb0-35d0-11eb-0a13-9bb4cf692fe5
one_hot_encoded = one_hot_encoder.fit_transform(str_mat)

# ╔═╡ d2f03780-35d0-11eb-0d06-d9a99e641d5f
begin
	converted = convert(DataFrame, one_hot_encoded)
	names!(converted, [convert(Symbol, name) for name in one_hot_encoder.get_feature_names()])
end

# ╔═╡ d313c510-35d0-11eb-28e4-9de2f998c526
clean_training_data = hcat(df_train[[col for col ∈ names if !(col ∈ str_cols)]], converted)

# ╔═╡ 5aa46890-3680-11eb-30af-a7af3580042b
size(clean_training_data)

# ╔═╡ d32689c0-35d0-11eb-3e99-0ba92d7041b5
@sk_import decomposition : PCA

# ╔═╡ d33a11c0-35d0-11eb-1aa8-37175b05bac1
pca = PCA()

# ╔═╡ b578e0c0-35db-11eb-36c5-ef3331dffbed
pca.fit(convert(Matrix, clean_training_data))

# ╔═╡ cc4f3680-35dd-11eb-279b-b13dbd9ab57b
col1 = pca.transform(convert(Matrix, clean_training_data))[:,2]

# ╔═╡ d7f241a0-35db-11eb-3419-5b185359c3ac
pca.explained_variance_ / sum(pca.explained_variance_)

# ╔═╡ 4d98e940-3681-11eb-2034-259a284343f5
pca.components_[1,:]

# ╔═╡ d96aaa60-3697-11eb-34f4-7d6d847089a4
transformed_training_data = pca.transform(convert(Matrix, clean_training_data))

# ╔═╡ 6d522dd2-3683-11eb-206b-3f3052492622
begin
	using StatsBase
	nb_SVMs = 50
	bootstrap_samples = []
	bootstrap_sample_labels = []
	for _ ∈ 1:nb_SVMs
	    # get id's of our random sample from our sample.
		#   (i) with replacement
		#   (ii) same size as our initial sample
	    ids = StatsBase.sample(1:nrow(clean_training_data), nrow(clean_training_data), replace=true)
	    # store bootstrapped data
		push!(bootstrap_samples, transformed_training_data[ids, :])
		push!(bootstrap_sample_labels, df_train[ids, :class])
	end
end

# ╔═╡ 28cc25b0-3685-11eb-3f53-0b32f5135f2c
@sk_import svm: SVC

# ╔═╡ a81a53a0-3685-11eb-3169-7d7be58f9039
svc_ensemble = [SVC(decision_function_shape="ovr", break_ties=true, random_state=i, probability=true) for i ∈ 1:nb_SVMs]

# ╔═╡ d74b5200-3685-11eb-0d19-7d316177c7aa
for (i, svc) ∈ enumerate(svc_ensemble)
	svc.fit(convert(Matrix, bootstrap_samples[i]), bootstrap_sample_labels[i])
end

# ╔═╡ ac4ce350-3691-11eb-0a4a-3795c1e627d3
svc_ensemble[1].score(transformed_training_data, df_train.class)

# ╔═╡ 27de9e22-3695-11eb-17ee-1d263b5128e4
svm_scores = [svc_ensemble[i].score(transformed_training_data, df_train.class) for i ∈ 1:nb_SVMs]

# ╔═╡ 29c73440-3695-11eb-1de0-4992f59f8b30
normalize!(svm_scores)

# ╔═╡ 29d9d1e0-3695-11eb-3ab2-4ff888669c41
test_encoded = one_hot_encoder.transform(convert(Matrix, df_test[str_cols]))

# ╔═╡ 33cb0a20-369a-11eb-04f0-59404ead383c
begin
	test_converted = convert(DataFrame, test_encoded)
		names!(test_converted, [convert(Symbol, name) for name in one_hot_encoder.get_feature_names()])
end

# ╔═╡ 29ece4b0-3695-11eb-1367-1988b8a20517
clean_test_data = hcat(df_test[[col for col ∈ names if !(col ∈ str_cols)]], test_converted)

# ╔═╡ 2a229ab0-3695-11eb-1c5d-8d0e5801f76d
transformed_test_data = pca.transform(convert(Matrix, clean_test_data))

# ╔═╡ 2a349c10-3695-11eb-3c5f-6734facd0931
begin
	prob_mats = []
	# loop over SVMs
	for i ∈ 1:nb_SVMs
		probsᵢ = zeros(nrow(df_test), length(unique(df_train.class)))
		# loop over examples
		for j ∈ 1:nrow(df_test)
			# get weighted predictions from each SVM for each test example
			probsᵢ[j,:] = svc_ensemble[i].predict_proba(
				[convert(Array, transformed_test_data[j,:])])
		end
		push!(prob_mats, probsᵢ)
	end
end

# ╔═╡ 2a6a03f0-3695-11eb-3be8-2db07954660a
for i ∈ 1:nb_SVMs # loop over SVM probability matrices
	# multiply each value in matrix i in prob_mats by svm_score[i]
	prob_mats[i] .*= svm_scores[i]
end

# ╔═╡ f0eeda0e-369e-11eb-149b-9db412ff10bd
prob_mats

# ╔═╡ a11770a0-369f-11eb-183f-3559fde51376
begin
	sums = zeros(nrow(df_test), length(unique(df_train.class))) # one array of 5 score sums for each test example
	for i ∈ 1:nb_SVMs
		for j ∈ 1:nrow(df_test)
			for k ∈ 1:length(unique(df_train.class))
				sums[j, k] += prob_mats[i][j, k]
			end
		end
	end
end

# ╔═╡ a164f462-369f-11eb-1ae5-698510068e43
findmax.(eachrow(sums))

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
# ╠═d2c99cb0-35d0-11eb-0a13-9bb4cf692fe5
# ╠═d2f03780-35d0-11eb-0d06-d9a99e641d5f
# ╠═d313c510-35d0-11eb-28e4-9de2f998c526
# ╠═5aa46890-3680-11eb-30af-a7af3580042b
# ╠═d32689c0-35d0-11eb-3e99-0ba92d7041b5
# ╠═d33a11c0-35d0-11eb-1aa8-37175b05bac1
# ╠═b578e0c0-35db-11eb-36c5-ef3331dffbed
# ╠═cc4f3680-35dd-11eb-279b-b13dbd9ab57b
# ╠═d7f241a0-35db-11eb-3419-5b185359c3ac
# ╠═4d98e940-3681-11eb-2034-259a284343f5
# ╠═d96aaa60-3697-11eb-34f4-7d6d847089a4
# ╠═6d522dd2-3683-11eb-206b-3f3052492622
# ╠═28cc25b0-3685-11eb-3f53-0b32f5135f2c
# ╠═a81a53a0-3685-11eb-3169-7d7be58f9039
# ╠═d74b5200-3685-11eb-0d19-7d316177c7aa
# ╠═ac4ce350-3691-11eb-0a4a-3795c1e627d3
# ╠═27de9e22-3695-11eb-17ee-1d263b5128e4
# ╠═29b496a0-3695-11eb-0326-0ff056416f1a
# ╠═29c73440-3695-11eb-1de0-4992f59f8b30
# ╠═29d9d1e0-3695-11eb-3ab2-4ff888669c41
# ╠═33cb0a20-369a-11eb-04f0-59404ead383c
# ╠═29ece4b0-3695-11eb-1367-1988b8a20517
# ╠═2a229ab0-3695-11eb-1c5d-8d0e5801f76d
# ╠═2a349c10-3695-11eb-3c5f-6734facd0931
# ╠═2a6a03f0-3695-11eb-3be8-2db07954660a
# ╠═f0eeda0e-369e-11eb-149b-9db412ff10bd
# ╠═a11770a0-369f-11eb-183f-3559fde51376
# ╠═a164f462-369f-11eb-1ae5-698510068e43
