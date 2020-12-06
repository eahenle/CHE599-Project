### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 6438b5e0-3758-11eb-0594-c7267760665f
using CSV, DataFrames, ScikitLearn, StatsBase, LinearAlgebra, PyPlot

# ╔═╡ a0953640-35b8-11eb-1357-37314995bbd0
md"
# Annealed Steel
"

# ╔═╡ b84dac80-3758-11eb-27a8-0f51c7f7b8c3
md"""
![](https://miro.medium.com/max/2400/1*3MTAI9UL6AcqYidc4RSG7A.jpeg)
"""

# ╔═╡ af48970e-3755-11eb-28e7-2d3565bebbe2
md"""
## Goal

Imagine an automated factory that makes milled steel parts from iron ore.  Somewhere along the line, you'll see something like the picture above: orange-hot steel rolling down a very long conveyor.  That steel is undergoing a process called annealing, and it has a major effect on the ultimate working properties of the steel.  If the annealing machines experience process deviations, the milling machines will need to adapt to the steel's new properties.  There is no time to get a human metallurgist to test each piece of steel and determine its grade, so we would like to train an AI on simple characterization data for rapid prediction of steel quality.
"""

# ╔═╡ 8f4f34e0-3756-11eb-2c5e-fbd1769a5605
begin
	@sk_import preprocessing : (OneHotEncoder, StandardScaler)
	@sk_import decomposition : PCA
	@sk_import svm: SVC
	@sk_import metrics : roc_auc_score
end;

# ╔═╡ 6075f110-3722-11eb-228e-67a4b6fd8e79
md"""
## Data

The UC Irvine ML dataset [Annealing](https://archive.ics.uci.edu/ml/datasets/Annealing) contains a total of 868 examples of characterization data from samples of steel annealed under various conditions and labeled by grade. 100 examples are reserved for testing the model (as "operational data"). These data are split between two files, `anneal.data` and `anneal.test`, with the column names provided in `anneal.names`.
"""


# ╔═╡ 40338f72-3759-11eb-2f23-a56baa6bb364
begin
	# read data files
	df_train = CSV.read("anneal.data", DataFrame, header=0)
	df_test = CSV.read("anneal.test", DataFrame, header=0)
	# anneal.names is full of info, not just column names
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
	# apply column names to dataframes
	names!(df_train, [names..., :class])
	names!(df_test, [names..., :class])
end

# ╔═╡ 27244ebe-3759-11eb-2d88-0ba8438faee3
md"""
There are six grade classes (1-5 and "U") and there are 38 data features per example.  Of the data features, 7 are numerical and the rest are categorical. The classes are fairly imbalanced, with class 4 totally unrepresented.
"""

# ╔═╡ b55f45b0-3785-11eb-38f7-e1bfd7e819ca
# pie chart
begin
	local df = by(df_train, :class, g -> DataFrame(proportion=nrow(g)/nrow(df_train)))
	clf()
	figure()
	title("Class Label Proportions")
	pie(df.proportion, labels=unique(df.class), autopct="%d%%", shadow=true, explode=0.1 .* ones(nrow(df)), startangle=38)
	gcf()
end

# ╔═╡ be196aa0-37e4-11eb-1d80-1beaf28195f2
md"""
Histograms of feature distributions:
"""

# ╔═╡ 9441cc20-3755-11eb-189a-3f44dd62bc13
begin
	clf()
	fig, axs = subplots(6, 6, sharey=true, figsize=(10,10))
	for i = 1:(length(names)-2) # just show 36 of 38, for aesthetics
		axs[i].hist(df_train[names[i]], bins=15)
		axs[i].set_title(names[i])
	end
	subplots_adjust(hspace=0.5)
	gcf()
end

# ╔═╡ 2c667e10-3783-11eb-33b4-fdc606f5ab89
md"""
We can see there are some features with no data.  `anneal.names` claims these factors should have some missing values, and some "not applicable" values.  The data have been corrupted, and that distinction no longer exists.

We can also see that our numerical variables have some decent-looking variance.  Unfortunately, some of these measurements are not applicable to our scenario: we will not be getting our operational data from sample bits of metal with recordable length, width, thickness, and/or bore size.  That leaves carbon content, hardness rating, and tensile strength, all of which can be measured by non-destructive methods [[$1$]](https://www.matec-conferences.org/articles/matecconf/pdf/2018/04/matecconf_nctam2018_05007.pdf) [[$2$]](https://www.buehler.com/nondestructive-testing.php#:~:text=Nondestructive%20testing%20(NDT)%20or%20Nondestructive,without%20altering%20or%20destroying%20it.&text=Typical%20types%20and%20method%20of,Rebound%2C%20and%20Ultrasonic%20Contact%20Impedance.) [[$3$]](https://www.bruker.com/products/x-ray-diffraction-and-elemental-analysis/handheld-xrf/applications/pmi/non-destructive-testing-ndt-xrf.html).

Correlation plots of numeric feature spaces:
"""

# ╔═╡ b0fb4f50-3785-11eb-0b70-b35912d6f61f
# correlation plots
begin
	clf()
	figure()
	for class in groupby(df_train, :class)
		scatter(class[:hardness], class[:carbon], label=class[1, :class])
	end
	gcf()
end

# ╔═╡ a2124920-3754-11eb-2468-fdde8eec9a6d
md"""
## Approach

To render the problem more approachable, we decided to reduce the dimensionality of the input data via Principal Component Analysis (PCA). After dimensional reduction, we train an ensemble of Support Vector Machines (SVMs) to perform multi-class classification.

Here is the Data Flow Diagram describing the ML architechture we employed:

![DFD](https://github.com/eahenle/CHE599-Project/raw/main/CHE599_DFD.png)
"""

# ╔═╡ edd07970-3756-11eb-2032-978d73caf455
md"""
Categorical data must be cast to pseudo-numeric for PCA.  We used one-hot-drop-one encoding, meaning that for each categorical variable with $n$ represented categories, there will be $n-1$ new Boolean columns indicating whether or not the example belongs to any given category.

The `String`-type columns of data are the only ones that should be encoded.  Column 39 is the classification label, so it is also excluded from encoding.  The net result is an addition of 5 feature columns.
"""

# ╔═╡ 478fde60-35d1-11eb-0f84-3f8344536cc0
begin
	one_hot_encoder = OneHotEncoder(drop="first", sparse=false)
	# peel off all String-type columns
	str_cols = [names[i] for (i,col) in enumerate(eachcol(df_train)) if typeof(col[1]) == String && i < 39]
	# one-hot-drop-one-encode
	one_hot_encoded = one_hot_encoder.fit_transform(convert(Matrix, df_train[str_cols]))
	# put one_hot_encoded back into DF form, attach col names
	converted = convert(DataFrame, one_hot_encoded)
	names!(converted, [convert(Symbol, name) for name in one_hot_encoder.get_feature_names()])
end

# ╔═╡ ba96b11e-3763-11eb-201b-a7d75f5c2407
md"""
The numerical columns need their own pre-processing for PCA: normalization.
"""

# ╔═╡ ecd57112-3729-11eb-37a5-61ae29db9853
begin
	num_cols = [col for col ∈ names if !(col ∈ str_cols)]
	numerics_train = convert(Matrix, df_train[:, num_cols])
	stdscaler = StandardScaler()
	numerics_train = convert(DataFrame, stdscaler.fit_transform(numerics_train))
	names!(numerics_train, num_cols)
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
	select!(clean_training_data, Not(:thick))
end

# ╔═╡ 14accaa0-3728-11eb-1a83-43ac80710b65
unique(clean_training_data[:strength])

# ╔═╡ 5aa46890-3680-11eb-30af-a7af3580042b
size(clean_training_data)

# ╔═╡ d32689c0-35d0-11eb-3e99-0ba92d7041b5


# ╔═╡ 81ce29b0-3752-11eb-3cce-f1fae2ecd0ee
# re-vis histograms from above, but only for important features

# ╔═╡ 7a3c1a50-3751-11eb-1322-bda70c8dabb3
# do prelim. pca w/ all features
# plot total var. captured vs. # features used

# ╔═╡ 28cc25b0-3685-11eb-3f53-0b32f5135f2c


# ╔═╡ 813ad410-3736-11eb-2b7f-0f26997e054a
# note: test-train sub-splits would probably be better than bootstrapping

# ╔═╡ 19c28e60-3734-11eb-0d8b-8f8ee1a84329


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
	select!(clean_test_data, Not(:thick))
end

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
findmax.(eachrow(pca.components_))

# ╔═╡ cf0bddd0-3743-11eb-3799-db73c43b0aac
[t[2] for t in findmax.(eachrow(pca.components_))]

# ╔═╡ a539b1c2-375d-11eb-0cbb-2f5269f0023f
[DataFrames.names(clean_training_data)[t[2]] for t in findmax.(eachrow(pca.components_))[1:nb_components]]

# ╔═╡ 27b66352-375e-11eb-24bf-1bfeed9b7386
str_cols[[7, 9, 29]]

# ╔═╡ Cell order:
# ╟─a0953640-35b8-11eb-1357-37314995bbd0
# ╟─b84dac80-3758-11eb-27a8-0f51c7f7b8c3
# ╟─af48970e-3755-11eb-28e7-2d3565bebbe2
# ╠═6438b5e0-3758-11eb-0594-c7267760665f
# ╠═8f4f34e0-3756-11eb-2c5e-fbd1769a5605
# ╟─6075f110-3722-11eb-228e-67a4b6fd8e79
# ╠═40338f72-3759-11eb-2f23-a56baa6bb364
# ╟─27244ebe-3759-11eb-2d88-0ba8438faee3
# ╠═b55f45b0-3785-11eb-38f7-e1bfd7e819ca
# ╟─be196aa0-37e4-11eb-1d80-1beaf28195f2
# ╠═9441cc20-3755-11eb-189a-3f44dd62bc13
# ╟─2c667e10-3783-11eb-33b4-fdc606f5ab89
# ╠═b0fb4f50-3785-11eb-0b70-b35912d6f61f
# ╟─a2124920-3754-11eb-2468-fdde8eec9a6d
# ╟─edd07970-3756-11eb-2032-978d73caf455
# ╠═478fde60-35d1-11eb-0f84-3f8344536cc0
# ╟─ba96b11e-3763-11eb-201b-a7d75f5c2407
# ╠═ecd57112-3729-11eb-37a5-61ae29db9853
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
# ╠═ce828990-3743-11eb-19f0-916a3d16b085
# ╠═ce983470-3743-11eb-28d6-57d29ddb834f
# ╠═ceac0a90-3743-11eb-1cce-e7b4c8eb5a55
# ╠═e99c6a80-3747-11eb-12d2-33d4c5322b52
# ╠═cebfb9a2-3743-11eb-06df-27ca3aecb1a1
# ╠═ced3419e-3743-11eb-0766-93ae85a72895
# ╠═cee5b830-3743-11eb-1bea-55606f41ea57
# ╠═cefce9ae-3743-11eb-10b0-715d2ad4a2fd
# ╠═cf0bddd0-3743-11eb-3799-db73c43b0aac
# ╠═a539b1c2-375d-11eb-0cbb-2f5269f0023f
# ╠═27b66352-375e-11eb-24bf-1bfeed9b7386
