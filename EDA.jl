### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 0e09d780-35b9-11eb-34c7-a7ff2d86d982
using CSV, DataFrames, ScikitLearn, StatsBase, LinearAlgebra

# ╔═╡ a0953640-35b8-11eb-1357-37314995bbd0
md"
# Annealed Steel
"

# ╔═╡ 6075f110-3722-11eb-228e-67a4b6fd8e79
md"""
## Data

The UC Irvine ML dataset [Annealing](https://archive.ics.uci.edu/ml/datasets/Annealing) contains a total of 868 examples of characterization data from samples of steel annealed under various conditions and labeled by grade. 100 examples are reserved for testing the model. There are six grade classes (1-5 and "U") and there are 38 data features per example.

## Goal

We would like to train an AI on these data for rapid prediction of steel grade based on new example data, and we would like it to require the minimum number of input features.

## Approach

We decided to approach this problem by reducing the dimensionality of the input data via Principal Component Analysis (PCA), and training an ensemble of Support Vector Machines (SVMs) to perform multi-class classification (MSVC).

Here is the Data Flow Diagram describing the ML architechture.

![DFD](https://github.com/eahenle/CHE599-Project/raw/main/CHE599_DFD.png)
"""

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

# ╔═╡ 91cffb10-3750-11eb-00c7-b9cad9abcd96
begin
	using PyPlot
	clf()
	figure()
	hist(df_train[:family])
	gcf()
end

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

# ╔═╡ ecd57112-3729-11eb-37a5-61ae29db9853
begin
	num_cols = [col for col ∈ names if !(col ∈ str_cols)]
	numerics_train = convert(Matrix, df_train[:, num_cols])
	@sk_import preprocessing : StandardScaler
	stdscaler = StandardScaler()
	numerics_train = convert(DataFrame, stdscaler.fit_transform(numerics_train))
	names!(numerics_train, num_cols)
end

# ╔═╡ 83978260-372e-11eb-3034-71d93524b5f9
begin
	numerics_test = convert(Matrix, df_test[:, num_cols])
	numerics_test = convert(DataFrame, stdscaler.transform(numerics_test))
	names!(numerics_test, num_cols)
end

# ╔═╡ d313c510-35d0-11eb-28e4-9de2f998c526
clean_training_data = hcat(numerics_train, converted)

# ╔═╡ a1d25690-372b-11eb-17ad-b73016fa7b55
select!(clean_training_data, Not(:bore)) # bore

# ╔═╡ 8f22eb2e-374a-11eb-3d32-43a290d79ea8
begin
	select!(clean_training_data, Not(:len))
	select!(clean_training_data, Not(:width))
end

# ╔═╡ 14accaa0-3728-11eb-1a83-43ac80710b65
unique(clean_training_data[:strength])

# ╔═╡ 5aa46890-3680-11eb-30af-a7af3580042b
size(clean_training_data)

# ╔═╡ d32689c0-35d0-11eb-3e99-0ba92d7041b5
@sk_import decomposition : PCA

# ╔═╡ 81ce29b0-3752-11eb-3cce-f1fae2ecd0ee


# ╔═╡ 7a3c1a50-3751-11eb-1322-bda70c8dabb3


# ╔═╡ 70058230-372c-11eb-2951-bb6c9f96c587
#@sk_import linear_model : LogisticRegression

# ╔═╡ a63ff470-372c-11eb-154e-5bdf42ff336a
#logistic_regressor = LogisticRegression()#(multi_class="multinomial")

# ╔═╡ b710c3b0-372c-11eb-39ef-2b16c3b44e46
#logistic_regressor.fit(convert(Matrix, clean_training_data), df_train[:class])

# ╔═╡ 0f7b0ab0-372d-11eb-014d-cd94adcd949e
#logistic_regressor.predict(convert(Matrix, clean_test_data))

# ╔═╡ 20c69d4e-372f-11eb-27bc-b7a1ec0c23bc
#df_test[:class]

# ╔═╡ 28cc25b0-3685-11eb-3f53-0b32f5135f2c
@sk_import svm: SVC

# ╔═╡ 813ad410-3736-11eb-2b7f-0f26997e054a
# note: test-train sub-splits would probably be better than bootstrapping

# ╔═╡ 19c28e60-3734-11eb-0d8b-8f8ee1a84329
@sk_import metrics : roc_auc_score

# ╔═╡ 29d9d1e0-3695-11eb-3ab2-4ff888669c41
test_encoded = one_hot_encoder.transform(convert(Matrix, df_test[str_cols]))

# ╔═╡ 33cb0a20-369a-11eb-04f0-59404ead383c
begin
	test_converted = convert(DataFrame, test_encoded)
		names!(test_converted, [convert(Symbol, name) for name in one_hot_encoder.get_feature_names()])
end

# ╔═╡ 29ece4b0-3695-11eb-1367-1988b8a20517
clean_test_data = hcat(numerics_test, test_converted)

# ╔═╡ 0f01ff90-372c-11eb-3dfd-cf1d66874ad4
# drop irrelevant physical data (size of sample shouldn't matter!)
begin
	select!(clean_test_data, Not(:bore)) # bore
end

# ╔═╡ a60fc110-374a-11eb-0234-dba8605864b7
begin
	select!(clean_test_data, Not(:len))
	select!(clean_test_data, Not(:width))
end

# ╔═╡ ce6b3100-3743-11eb-3b08-cdcff67da52b
classes

# ╔═╡ ce828990-3743-11eb-19f0-916a3d16b085
unique(df_train[:class])

# ╔═╡ cebfb9a2-3743-11eb-06df-27ca3aecb1a1
begin
	# hyperparameters
	nb_SVMs = 20
	nb_components = 8
	auroc_delta = false
end;

# ╔═╡ d33a11c0-35d0-11eb-1aa8-37175b05bac1
pca = PCA(n_components=nb_components)

# ╔═╡ b578e0c0-35db-11eb-36c5-ef3331dffbed
pca.fit(convert(Matrix, clean_training_data))

# ╔═╡ cc4f3680-35dd-11eb-279b-b13dbd9ab57b
col1 = pca.transform(convert(Matrix, clean_training_data))[:,6]

# ╔═╡ 1f8942e0-3751-11eb-0b95-5bbbe03cc2bf
begin
	clf()
	figure()
	hist(col1, bins=20)
	gcf()
end

# ╔═╡ d7f241a0-35db-11eb-3419-5b185359c3ac
sum(pca.explained_variance_ / sum(pca.explained_variance_))

# ╔═╡ ce469830-3726-11eb-2483-9b23c88fa945
pca.explained_variance_

# ╔═╡ 4d98e940-3681-11eb-2034-259a284343f5
findmax(pca.components_[1,:])

# ╔═╡ d96aaa60-3697-11eb-34f4-7d6d847089a4
transformed_training_data = pca.transform(convert(Matrix, clean_training_data))

# ╔═╡ 2a229ab0-3695-11eb-1c5d-8d0e5801f76d
transformed_test_data = pca.transform(convert(Matrix, clean_test_data))

# ╔═╡ 6d522dd2-3683-11eb-206b-3f3052492622
begin
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

# ╔═╡ a81a53a0-3685-11eb-3169-7d7be58f9039
svc_ensemble = [SVC(decision_function_shape="ovr", break_ties=true, random_state=i, probability=true) for i ∈ 1:nb_SVMs]

# ╔═╡ d74b5200-3685-11eb-0d19-7d316177c7aa
for (i, svc) ∈ enumerate(svc_ensemble)
	svc.fit(convert(Matrix, bootstrap_samples[i]), bootstrap_sample_labels[i])
end

# ╔═╡ 0a925720-3731-11eb-327d-2fddc666ee11
svc_ensemble[1].predict_proba(transformed_training_data)

# ╔═╡ ac4ce350-3691-11eb-0a4a-3795c1e627d3
svc_ensemble[1].score(transformed_training_data, df_train.class)

# ╔═╡ ce983470-3743-11eb-28d6-57d29ddb834f
ordered_classes = svc_ensemble[1].classes_

# ╔═╡ db4af520-3731-11eb-30b5-c3ddde182cad
begin
	aurocs = []
	# run predict_proba on all examples for each SVM (50 x (n_samples, n_classes))
	svc_probs = [
		svc_ensemble[i].predict_proba(transformed_training_data) for i ∈ 1:nb_SVMs]
	# avg over SVMs by class for each example (n_samples, n_classes) 
	class_probabilities = zeros(nrow(df_train), length(unique(df_train[:class])))
	# loop over SVMs
	for mat ∈ svc_probs
		# one ROC per mat (SVM)
		push!(aurocs, roc_auc_score(df_train[:class], mat, multi_class="ovr"))
		# loop over classes
		for c ∈ 1:length(unique(df_train[:class]))
			# loop over examples
			for e ∈ 1:nrow(df_train)
				# accumulate probabilities
				class_probabilities[e, c] += mat[e, c]
			end
		end
	end
	class_probabilities ./= nb_SVMs
end

# ╔═╡ a2ed8890-3741-11eb-21cb-9735d0c30418
aurocs

# ╔═╡ fc5ce7b0-373a-11eb-3929-31a571cfb0c9
sum.(eachrow(class_probabilities))

# ╔═╡ 4c04a3b0-373c-11eb-007c-4ba2f9af318b
sum.(eachcol(class_probabilities))

# ╔═╡ 03a68a20-3741-11eb-1153-992c7a6d8bb7
class_probabilities

# ╔═╡ 2e81cb50-3733-11eb-066d-310024101c50
class_probabilities

# ╔═╡ 814ac480-3734-11eb-13ef-a7f000537101
roc_auc_score(df_train[:class], class_probabilities, multi_class="ovr")

# ╔═╡ 27de9e22-3695-11eb-17ee-1d263b5128e4
svm_scores = [svc_ensemble[i].score(transformed_training_data, df_train.class) for i ∈ 1:nb_SVMs]

# ╔═╡ 29c73440-3695-11eb-1de0-4992f59f8b30
normalize!(svm_scores)

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

# ╔═╡ f0eeda0e-369e-11eb-149b-9db412ff10bd
prob_mats

# ╔═╡ 2a6a03f0-3695-11eb-3be8-2db07954660a
for i ∈ 1:nb_SVMs # loop over SVM probability matrices
	# multiply each value in matrix i in prob_mats by svm_score[i]
	prob_mats[i] .*= aurocs[i] - 0.5*auroc_delta
end

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
unique(findmax.(eachrow(sums)))

# ╔═╡ ce549bc0-3743-11eb-01cb-bfeca5d4acbf
final_predictions = [ordered_classes[t[2]] for t ∈ findmax.(eachrow(sums))]

# ╔═╡ ceac0a90-3743-11eb-1cce-e7b4c8eb5a55
final_predictions[[df_test[i, :class] ≠ final_predictions[i] for i ∈ 1:nrow(df_test)]]

# ╔═╡ e99c6a80-3747-11eb-12d2-33d4c5322b52
length(final_predictions[[df_test[i, :class] ≠ final_predictions[i] for i ∈ 1:nrow(df_test)]])

# ╔═╡ ced3419e-3743-11eb-0766-93ae85a72895
findmax.(abs.(pca.components_[i,:]) for i ∈ 1:nb_components)

# ╔═╡ cee5b830-3743-11eb-1bea-55606f41ea57
pca.components_[1,:]

# ╔═╡ cefce9ae-3743-11eb-10b0-715d2ad4a2fd


# ╔═╡ cf0bddd0-3743-11eb-3799-db73c43b0aac


# ╔═╡ Cell order:
# ╟─a0953640-35b8-11eb-1357-37314995bbd0
# ╠═0e09d780-35b9-11eb-34c7-a7ff2d86d982
# ╟─6075f110-3722-11eb-228e-67a4b6fd8e79
# ╟─5c246a30-35c2-11eb-3ee2-e70dfe2b95f9
# ╠═e232f93e-35bb-11eb-30d2-8fb4b187ada5
# ╠═e9ba3a90-35b9-11eb-3c82-8dc71624c603
# ╠═27455850-35b9-11eb-29cf-1f97fb339e66
# ╠═b77f525e-35cb-11eb-1bb8-8f2fabb48df0
# ╠═91cffb10-3750-11eb-00c7-b9cad9abcd96
# ╠═415714e0-35b9-11eb-39db-338b9ad54699
# ╠═bf8a3e6e-35d0-11eb-3076-212d47a44b25
# ╠═d2a2b3c0-35d0-11eb-24f7-495a3459c779
# ╠═d2b50340-35d0-11eb-3d85-2dc942f49a2b
# ╠═478fde60-35d1-11eb-0f84-3f8344536cc0
# ╠═0f419560-35d3-11eb-2771-c3ca6ea4c912
# ╠═7c0f04b0-35d4-11eb-0489-1146ca65eb13
# ╠═d2c99cb0-35d0-11eb-0a13-9bb4cf692fe5
# ╠═d2f03780-35d0-11eb-0d06-d9a99e641d5f
# ╠═ecd57112-3729-11eb-37a5-61ae29db9853
# ╠═83978260-372e-11eb-3034-71d93524b5f9
# ╠═d313c510-35d0-11eb-28e4-9de2f998c526
# ╠═a1d25690-372b-11eb-17ad-b73016fa7b55
# ╠═8f22eb2e-374a-11eb-3d32-43a290d79ea8
# ╠═14accaa0-3728-11eb-1a83-43ac80710b65
# ╠═5aa46890-3680-11eb-30af-a7af3580042b
# ╠═d32689c0-35d0-11eb-3e99-0ba92d7041b5
# ╠═d33a11c0-35d0-11eb-1aa8-37175b05bac1
# ╠═b578e0c0-35db-11eb-36c5-ef3331dffbed
# ╠═cc4f3680-35dd-11eb-279b-b13dbd9ab57b
# ╠═81ce29b0-3752-11eb-3cce-f1fae2ecd0ee
# ╠═1f8942e0-3751-11eb-0b95-5bbbe03cc2bf
# ╠═7a3c1a50-3751-11eb-1322-bda70c8dabb3
# ╠═d7f241a0-35db-11eb-3419-5b185359c3ac
# ╠═ce469830-3726-11eb-2483-9b23c88fa945
# ╠═4d98e940-3681-11eb-2034-259a284343f5
# ╠═d96aaa60-3697-11eb-34f4-7d6d847089a4
# ╠═70058230-372c-11eb-2951-bb6c9f96c587
# ╠═a63ff470-372c-11eb-154e-5bdf42ff336a
# ╠═b710c3b0-372c-11eb-39ef-2b16c3b44e46
# ╠═0f7b0ab0-372d-11eb-014d-cd94adcd949e
# ╠═20c69d4e-372f-11eb-27bc-b7a1ec0c23bc
# ╠═6d522dd2-3683-11eb-206b-3f3052492622
# ╠═28cc25b0-3685-11eb-3f53-0b32f5135f2c
# ╠═a81a53a0-3685-11eb-3169-7d7be58f9039
# ╠═d74b5200-3685-11eb-0d19-7d316177c7aa
# ╠═0a925720-3731-11eb-327d-2fddc666ee11
# ╠═db4af520-3731-11eb-30b5-c3ddde182cad
# ╠═a2ed8890-3741-11eb-21cb-9735d0c30418
# ╠═fc5ce7b0-373a-11eb-3929-31a571cfb0c9
# ╠═4c04a3b0-373c-11eb-007c-4ba2f9af318b
# ╠═03a68a20-3741-11eb-1153-992c7a6d8bb7
# ╠═813ad410-3736-11eb-2b7f-0f26997e054a
# ╠═2e81cb50-3733-11eb-066d-310024101c50
# ╠═19c28e60-3734-11eb-0d8b-8f8ee1a84329
# ╠═814ac480-3734-11eb-13ef-a7f000537101
# ╠═ac4ce350-3691-11eb-0a4a-3795c1e627d3
# ╠═27de9e22-3695-11eb-17ee-1d263b5128e4
# ╠═29c73440-3695-11eb-1de0-4992f59f8b30
# ╠═29d9d1e0-3695-11eb-3ab2-4ff888669c41
# ╠═33cb0a20-369a-11eb-04f0-59404ead383c
# ╠═29ece4b0-3695-11eb-1367-1988b8a20517
# ╠═0f01ff90-372c-11eb-3dfd-cf1d66874ad4
# ╠═a60fc110-374a-11eb-0234-dba8605864b7
# ╠═2a229ab0-3695-11eb-1c5d-8d0e5801f76d
# ╠═2a349c10-3695-11eb-3c5f-6734facd0931
# ╠═2a6a03f0-3695-11eb-3be8-2db07954660a
# ╠═f0eeda0e-369e-11eb-149b-9db412ff10bd
# ╠═a11770a0-369f-11eb-183f-3559fde51376
# ╠═a164f462-369f-11eb-1ae5-698510068e43
# ╠═ce549bc0-3743-11eb-01cb-bfeca5d4acbf
# ╠═ce6b3100-3743-11eb-3b08-cdcff67da52b
# ╠═ce828990-3743-11eb-19f0-916a3d16b085
# ╠═ce983470-3743-11eb-28d6-57d29ddb834f
# ╠═ceac0a90-3743-11eb-1cce-e7b4c8eb5a55
# ╠═e99c6a80-3747-11eb-12d2-33d4c5322b52
# ╠═cebfb9a2-3743-11eb-06df-27ca3aecb1a1
# ╠═ced3419e-3743-11eb-0766-93ae85a72895
# ╠═cee5b830-3743-11eb-1bea-55606f41ea57
# ╠═cefce9ae-3743-11eb-10b0-715d2ad4a2fd
# ╠═cf0bddd0-3743-11eb-3799-db73c43b0aac
