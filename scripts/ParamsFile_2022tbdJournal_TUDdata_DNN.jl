#params file that is modified as needed to run different experiments
#there are dependencies in setting up params (mainly keyword structs)

using DrWatson
@quickactivate "PseudoNormRegression"

using MLJ #for the @load of base model
using Zenodo3968503Functions
include(srcdir("similarity_functions.jl"))

#explicit dictionary structure to avoid error creating dictionary with
#@strdict keyword method (forget to modify if add/remove parameters)

paramsSK = Dict(
  "method" => "SingleKernel",
  "p" => collect(range(0.1, stop=2.0, length=21)),
  "dbpath" => datadir("exp_pro"),
  "db" => ["TUD.mat"],
  "DataParams" => [
#                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  #efficientnet specific
  "blocks" => 3, 
  "use_cuda" => true,
  "base_model" => "efficientnet-b0",
  "in_channels" => 1, "nclasses"=>2,
  "batchsize" => 32, "epochs"=>100, "η"=>1e-2, "stem_k"=>(3,1) 
               )  

paramsSTD = Dict(
  "method" => "Standard",
  "p" => nothing,
  "dbpath" => datadir("exp_pro"),
  "db" => ["TUD.mat"],
  "DataParams" => [
                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
                  ],
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
                      ],
  #efficientnet specific
  "blocks" => 4,
  "use_cuda" => true,
  "base_model" => "efficientnet-b0",
  "in_channels" => 1, "nclasses"=>2,
  "batchsize" => [32], "epochs"=>100, "η"=>[1e-3], "stem_k"=>(3,1) 
               )  

@info "Using $(paramsSK["method"]) with $(paramsSK["base_model"]) to process these files: $(paramsSK["db"])..."
@info "Using $(paramsSTD["method"]) with $(paramsSTD["base_model"]) to process these files: $(paramsSTD["db"])..."

param_combinations = append!(dict_list(paramsSK), dict_list(paramsSTD))[22:end];
