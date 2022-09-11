module DNNregModule

using DrWatson
@quickactivate "PseudoNormRegression"

#include other packages/functions as necessary
using MLJBase
using Flux, MLUtils
using Flux.Data: DataLoader
using Flux.Losses: mse
using CUDA
using Logging
using ProgressMeter, Dates
using DataFrames
using QuasinormRegression
using Zenodo3968503Functions
using DNNregression

#prompt for params file, which contains multiple sets of params typically
println("Enter in params file (ParamsFile_2022tbdJournal_DNN.jl ParamsFile_2022tbdJournal_TUDdata_DNN.jl etc.): ")
paramfile = readline()
include(scriptsdir(paramfile))


#to keep output clean, MLJ models have lot of Info/Warn messages
#Logging.disable_logging(Logging.LogLevel(-3000))
#println("Disabled Info and Warn logging -- this will negatively affect TBLogger/Tensorboard")

#loop over all possible combination and save results, make thread safe
compLock = ReentrantLock();
println("There are $(size(param_combinations,1)) combinations to iterate over...\n")
track_progress = Progress( size(param_combinations,1), barlen=50 )

#Threads.@threads for idx_params in param_combinations
for (countidx, idx_params) in enumerate(param_combinations)

  #unpack parameter
  @unpack use_cuda, base_model, blocks, in_channels, stem_k, nclasses, η, 
  method, SamplingParams, batchsize, epochs, db = idx_params 

  #quick CUDA checking and such
  (CUDA.functional() && use_cuda) ? (CUDA.allowscalar(false); device=gpu) : (device=cpu; @error "using cpu as functional is $(CUDA.functional()) and use_cuda flag is $(use_cuda)")
                                                                                  
  #create models 
  @debug "creating model"
  model = DNNregression.EfficientNet(efficientnet_global_configs[Symbol(split(base_model, "-")[2])][2], 
                                     efficientnet_block_configs[1:blocks],
                                     inchannels = in_channels, 
                                     stem_k = stem_k,
                                     nclasses = nclasses) |> device
#  @info model

  # model parameters
  ps = Flux.params(model)
  ## Optimizer
  opt = RMSProp(η)

  #get/configure data based on passed parameters
  #used for dataloaders
  @debug "getting data"
  xtrain, ytrain, xtest, ytest = getdata(idx_params)
  @debug "size(xtrain): $(size(xtrain))\n size(ytrain): $(size(ytrain))\n size(xtest): $(size(xtest))\n size(ytest): $(size(ytest))"

 
  #get values for saving later
  mdl_base_type = base_model
  mdl_type = method

  #create a loop for multiple runs
  mee_runs = Float64[]; σee_runs = Float64[];
  @debug "entering multiple run loop"
  for idx_runs in 1:SamplingParams.num_runs
  
    #create a loop for bagging situation
    yhat = zeros(size(ytest))
    bagging_runs = SamplingParams.bagging_runs

    @debug "entering bagging loop"
    for idx_bagging in 1:bagging_runs
      
      ##get training and test sample indices, need to pass coordinates
      #getindices is a function in src/sampling_functions.jl, expects
      #dataframe
      train_idx, _, _ = getindices(DataFrame(ytrain',[:x, :y]), idx_params["train_size"], idx_params["SamplingParams"])
      @debug "size of train_idx: $(size(train_idx))"
      #break out of both bagging and multiple run loop if indices won't work
      if train_idx== nothing @warn "train_idx is $train_idx (getindices failed)"; @goto runloopbrk; end
  
      #train/fit (can utilize tensorboard to track performance over epochs
      @debug "entering epoch loop"
      for epoch in 1:epochs
          @debug "epoch number: $epoch"
          for (x, y) in eachobs((xtrain[:,:,:,train_idx],ytrain[:,train_idx]), batchsize=batchsize)
              x, y = device(x), device(y) # transfer data to device
              gs = gradient(() -> mse(model(x), y), ps) # compute gradient
              Flux.Optimise.update!(opt, ps, gs) # update parameters
           end #end data loader
 
          #print to screen
#          @printf("\rp:%0.2f, epoch: %5d, train_loss: %0.2f, \t test_loss: %0.2f \t test_std: %0.2f \t.",
#                  args.p, epoch, train_loss, test_loss, test_std)
      end #end epoch

      #if put in bagging, need to add up estimates...
      ytemp = []
      for (x,y) in eachobs((xtest,ytest), batchsize=batchsize)
          x = device(x)
          isempty(ytemp) ? ytemp = model(x) : ytemp = hcat(ytemp, model(x))
      end
      yhat += cpu(ytemp)

    end #end of bagging loop

    #average the prediction... if do bagging at some point
    yhat /= bagging_runs

    #mean euclidean distance error (and standard dev)
#    test_loader  = DataLoader((xtest, ytest), batchsize=batchsize)
#    mee = loss(test_loader, model, device)
    mee = mean(sqrt.(sum((yhat .- ytest).^2,dims=1)))
#    σee = spread(test_loader, model, device)
    σee = std(sqrt.(sum((yhat .- ytest).^2,dims=1)))
    #save for μ and σ calcs
    push!(mee_runs, mee); push!(σee_runs, σee);

  end #end multiple run loop

  #get mean and average over multiple runs, don't track meex, meey, meez σ
  mee = mean(mee_runs); σee = mean(σee_runs);
  SamplingParams.num_runs>1 ?  mee_σ = std(mee_runs) : mee_σ = 0
    
  #make thread safe -- don't write to same file
  begin
    lock(compLock)
    try
      safesave(datadir("exp_pro","results", db, mdl_type, mdl_base_type, 
                       savename((dbname=db,mdlType=mdl_type,mdlBase=mdl_base_type), "jld2";   #keep name simpler for now
                                ignores=["packing_dir"])), ( merge((@strdict mdl_type mdl_base_type mee mee_σ σee), idx_params) )) 
    finally
      unlock(compLock)
    end
  end #end thread lock

  @label runloopbrk
  #tracking how much completed
  next!(track_progress; showvalues = [(:countidx, countidx)])
end #param combination loop

#change logging level back to standard
#Logging.disable_logging(Logging.Debug);

#save a record of this successful script run
#@info "Saving a record of script for each database used..."
#for idx in params["db"]
#    for idx2 in params["base_model"]
#        @info  *(Dates.format(now(), "yyyymmdd-HH:MM:SS_"), idx, params["method"], MLJBase.name(idx2), ".jld2")
#        safesave(datadir("exp_pro", "testing_history", 
#                         *(Dates.format(now(), "yyyymmdd-HH:MM:SS_"), idx, params["method"], 
#                           MLJBase.name(idx2), ".jld2") ), params)
#    end
#end


println("finished running DNNregModule using $paramfile")

end #end of module
