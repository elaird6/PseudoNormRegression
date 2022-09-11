#params file that is modified as needed to run different experiments
#there are dependencies in setting up params (mainly keyword structs)

using DrWatson
@quickactivate "PseudoNormRegression"

using MLJ #for the @load of base model
using Zenodo3968503Functions
include(srcdir("similarity_functions.jl"))

#explicit dictionary structure to avoid error creating dictionary with
#@strdict keyword method (forget to modify if add/remove paramters)

paramsSKknn = Dict(
  "method" => "SingleKernel",
  "p" => collect(range(0.1, stop=2.0, length=21)),
  "dbpath" => datadir("exp_pro"),
  "db" => ["JHUtdoa.mat", "JHUrss.mat", "JHUaoa.mat", "DSI1.mat", "UTIf2.mat", "LIB1.mat"],
  "DataParams" => [
#                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  "K" => collect(1:2:7), 
  "metric" => Euclidean(),
  "algorithm" => :brutetree, "leafsize" => 0, "reorder"=>false,#default is :kdtree 
  "base_model" => @load KNNRegressor verbosity=0  #due to quirk of loading, this needs to be last
 )

paramsSTD = Dict(
  "method" => "Standard",
  "dbpath" => datadir("exp_pro"),
  "db" => ["JHUtdoa.mat", "JHUrss.mat", "JHUaoa.mat", "DSI1.mat", "UTIf2.mat", "LIB1.mat"],
  "DataParams" => [
                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  "K" => collect(1:2:7), 
  "metric" => [Minkowski(x) for x in collect(range(0.1, stop=2.0, length=21))], 
  "algorithm" => :brutetree, "leafsize" => 0, "reorder"=>false,#default is :kdtree 
  "base_model" => @load KNNRegressor verbosity=0  #due to quirk of loading, this needs to be last
 )

paramsSK = Dict(
  "method" => "SingleKernel",
  "p" => collect(range(0.1, stop=2.0, length=21)),
  "dbpath" => datadir("exp_pro"),
  "db" => ["JHUtdoa.mat", "JHUrss.mat", "JHUaoa.mat", "UTIf2.mat", "LIB1.mat"],
  "DataParams" => [
#                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  "lambda" => 10^(-6), 
  "base_model" => [@load RidgeRegressor pkg ="MLJLinearModels" verbosity=0
                   @load LassoRegressor pkg ="MLJLinearModels" verbosity=0
                  ]
 )

#DSI1 has better performance with regularization parameter 1e-4
paramsSK_DSI1 = Dict(
  "method" => "SingleKernel",
  "p" => collect(range(0.1, stop=2.0, length=21)),
  "dbpath" => datadir("exp_pro"),
  "db" => ["DSI1.mat"],
  "DataParams" => [
#                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  "lambda" => 10^(-4), 
  "base_model" => [@load RidgeRegressor pkg ="MLJLinearModels" verbosity=0
                   @load LassoRegressor pkg ="MLJLinearModels" verbosity=0
                  ]
 )
###############################################################
## need to do some basic parameter compatibility checking here...
##


@info "Using this process: $(paramsSKknn["method"]) to process these files: $(paramsSKknn["db"])..."
@info "Using this process: $(paramsSTD["method"]) to process these files: $(paramsSTD["db"])..."
@info "Using this process: $(paramsSK["method"]) to process these files: $(paramsSK["db"])..."
@info "Using this process: $(paramsSK_DSI1["method"]) to process these files: $(paramsSK_DSI1["db"])..."

param_combinations = append!(dict_list(paramsSKknn), dict_list(paramsSTD), 
                             dict_list(paramsSK), dict_list(paramsSK_DSI1));
