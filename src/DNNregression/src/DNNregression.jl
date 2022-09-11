module DNNregression

#Note that a source file within the MetalHead EfficientNet package has to be 
#modified to deal with 1-D vectors versus 2-D images. 
#
#Some modifications in this DNNregression module to address is dimension
#expansion of feature and location (X_df, y_df) from (features, observations)
#to (features, 1, 1, observations) to match WHCB format to minimize this
#source hacking. A second modification in this module is passing a kernel
#tuple rather than integer (both of which Metalhead EfficientNet supports) via
#the `efficientnet_block_configs` variable The source hacking is as below which 
#is changing an initial conv kernel, specifically the stem kernel.
#
#    ~/.julia/packages/Metalhead/????/src/convnets/efficientnet.jl
#       -- add `stem_k = (3,3)` to each EfficientNet/efficienet function declaration 
#       -- add `stem_k = stem_k` to the efficient function call 
#       -- replace `(3,3)` with `stem_k` in conv_bn() in stem layer declaration
#
#If used, results are stored in Tensorboard data -- see Tensorboard log directory --
#start up tensorboard using the following command:

#    conda update tensorboard                   # update to latest version (note may need to do install instead)
#    conda activate tensorboard                 # activate python environment for tensorboard
#    tensorboard --logdir tensorboard_logs/     # start tensorboard


using QuasinormRegression
using Zenodo3968503Functions
using Statistics
using Tables: matrix

using Metalhead: EfficientNet
include("EfficientNetConfigs.jl")
export efficientnet_block_configs, efficientnet_global_configs

export getdata
export loss
export spread
export EfficientNet

function getdata(args)
  #load data file
  #figure out a way to unpack in function call (simply change function)
  #this works as long as DataParam is defined as kw_struct (Zenodo3968503Functino)
  (; avgmeas, flag100, flagval, verbose) = args["DataParams"]
  X_df, y_df, X_df_validation, y_df_validation = ReadZenodoMatFiles(joinpath(args["dbpath"], args["db"]),
                                                                    avgmeas=avgmeas, flag100=flag100, flagval=flagval);

  #format data
  ytrain = (matrix(y_df)[:,1:2]') 
  ytest  = (matrix(y_df_validation)[:,1:2]') 

  #create and keep track of λ_kern
  any("λ_kern" .== keys(args)) ? λ_kern = args["λ_kern"] : λ_kern = 999.0 

  #kernel method
  if args["method"] == "SingleKernel"
    #convert from row-oriented to column-oriented (adjoint or transpose)
    #p-norm exponential distance doesn't work well, simply calculate p-norm distance
    xtrain, λ_kern = datakernelization(X_df, args["p"], λ_kern=λ_kern)
    #transpose
    xtrain = matrix(xtrain)'
    #add add extra dimensions
    xtrain = xtrain[:, [CartesianIndex()], [CartesianIndex()] , :]
    
    #now kernelized test/validation data
    xtest = datakernelization(X_df_validation, X_df, args["p"], λ_kern)
    xtest = matrix(xtest)'
    xtest = xtest[:, [CartesianIndex()], [CartesianIndex()] , :]
    
  #non kernel method
  elseif args["method"] == "Standard"

    xtrain = ((matrix(X_df).-mean(matrix(X_df),dims=2))./(std(matrix(X_df),dims=2)))'
    xtrain = xtrain[:, [CartesianIndex()], [CartesianIndex()], :]  #singleton axis whcbn so that it works with conv

    xtest = ((matrix(X_df_validation).-mean(matrix(X_df_validation),dims=2))./(std(matrix(X_df_validation),dims=2)))'
    xtest = xtest[:, [CartesianIndex()], [CartesianIndex()], :]  #singleton axis whcbn so that it works with conv

  else
    error("Passed wrong method, should be Standard or SingleKernel")
  end

  return xtrain, ytrain, xtest, ytest
  
end #end getdata function

#loss function for DNN -- calculated every batch
function loss(data_loader, model, device)
  ls = 0.0
  num = 0
  for (x, y) in data_loader
    x, y = device(x), device(y)
    ŷ = model(x)
    ls += sum(sqrt.(sum((ŷ .- y).^2,dims=1)))
    num +=  size(x, length(size(x)))          #get number of dimensions of input, use to get number of observations
    #@show num
  end
  return ls / num
end

#standard deviation for DNN -- calculated every batch
function spread(data_loader, model, device)
  ls = 0.0
  num = 0
  for (x, y) in data_loader
    x, y = device(x), device(y)
    ŷ = model(x)
    ls += std(sqrt.(sum((ŷ .- y).^2,dims=1)))
    num +=  1
  end
  return ls / num
end

end #end of module
