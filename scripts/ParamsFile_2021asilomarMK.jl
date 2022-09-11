#params file that is modified as needed to run different experiments
#there are dependencies in setting up params (mainly keyword structs)

using DrWatson
@quickactivate "PseudoNormRegression"

using MLJ #for the @load of base model
using Zenodo3968503Functions
include(srcdir("similarity_functions.jl"))

#explicit dictionary structure to avoid error creating dictionary with
#@strdict keyword method (forget to modify if add/remove paramters)

params = Dict(
  #type of regression: Standard, SingleKernel, MultiKernel
#  "method" => "Standard",
  #setup kernel parameters (need first as can affect model parameters)
  #kernel parameters == composite MLJ model parameters
#  "method" => "SingleKernel",
#  "p" => collect(range(0.1, stop=2.0, length=21)),
  #Multi-kernel -- multiple p's
  "method" => "MultiKernel",
  #"p" => [[x y z] for x in collect(range(0.2, stop=2.0, length=21)) for y in collect(range(0.2, stop=2.0, length=21)) for z in collect(range(0.2, stop=2.0, length=21))],
   "p" => [[x x x] for x in collect(range(0.2, stop=2.0, length=21)) ],
#  "p" => [[1.3 1.5 1.1]],
   "λ_kern" => [[999.0 999.0 999.0]], # -- let be default. Multiple values for multi-kernel
  "f_counts" => [[112 16 8]],
#  "f_counts" => [[112 16]],

  #set up data parameters
  "dbpath" => datadir("exp_pro"),
  "db" => ["JHUall.mat"],
#  "db" => ["UTIf2.mat", "LIB1.mat", "MAN1.mat"],  #need to rerun as seg fault
#  "db" => filter!(s->!occursin(r"_[0-9].*\.mat|SIM.*mat|LTS01|UJI|UBL|TUT", s), 
#                  readdir(datadir("Zenodo_3968503","databases"))),
#  "db" => filter!(s->occursin(r"U[A-Z]{2}\d.mat", s), readdir(datadir("Zenodo_3968503","databases"))),
  #set these values in Zenodo data keyword struct (in src/ZenodoFunctions.jl) below:
  #avgmeas=true, flag100=true, flagval=-999.0, verbose=true, datarep = nothing
  #                                                             DTpowed/DTexponential/DTpositive
  "DataParams" => [
                   ReadZenodoMATParams(avgmeas=false, flag100=true, verbose=false, datarep=nothing)      #for standard approach
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=nothing)      #for singlekernel approach
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=DTpowed)
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=DTexponential)
#                   ReadZenodoMATParams(avgmeas=true, flag100=true, verbose=false, datarep=DTpositive)
                  ],
  ## setup training indices parameters
  "train_size" => 1,#if <= 1.0, treat as percentages otherwise abs values
#  "SamplingParams" => SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1),
  #set these values in data keyword struct (in src/sampling_functions.jl) below:
  #       sphere_packing=false, packing_dir, rand_offset=0.0, grp_flag=false,
  #       grp_val=5, num_runs=1, bagging_runs=1
#  "train_size" => [9, 18, 35, 71, 141, 282],
  "SamplingParams" => [
                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=1)
#                       SampleParamsStruct(sphere_packing=false, packing_dir=datadir("packing_coords"), num_runs=20)
#                       SampleParamsStruct(sphere_packing=true, packing_dir=datadir("packing_coords"), rand_offset=0.0, bagging_runs=1);
#                       SampleParamsStruct(sphere_packing=true, packing_dir=datadir("packing_coords"), rand_offset=0.7, grp_flag=true, bagging_runs=1);
##                       SampleParamsStruct(sphere_packing=true, packing_dir=datadir("packing_coords"), rand_offset=1.5, bagging_runs=20)
                      ],

  #setup base model and associated model specific parameters for KNN
  #metric should be euclidean for Kernel and minkowski for Standard
#  "K" => 5, 
#  "K" => collect(1:2:7), 
#  "K" => collect(9:2:19), 

#  "metric" => Euclidean(),
#  "metric" => [Minkowski(x) for x in collect(range(0.1, stop=2.0, length=21))], 

#  "algorithm" => :brutetree, "leafsize" => 0, "reorder"=>false,#default is :kdtree 
#  "base_model" => @load KNNRegressor verbosity=0  #due to quirk of loading, this needs to be last

  #setup base model and associated model specific parameters for Ridge & LASSO
  "lambda" => 10^(-6), 
  "base_model" => [@load RidgeRegressor pkg ="MLJLinearModels" verbosity=0
                   @load LassoRegressor pkg ="MLJLinearModels" verbosity=0
                  ]
#           
 )
###############################################################
## need to do some basic parameter compatibility checking here...
##
##
@info "Using this process: $(params["method"]) to process these files: $(params["db"])..."

param_combinations = dict_list(params);
