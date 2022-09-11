module LinearRegModuleLoop

#this version of LinearRegModule is used for scenarios where
#data needs to be generated multiple times
using DrWatson
@quickactivate "PseudoNormRegression"

#include other packages/functions as necessary
using MLJ, MLJBase
using Logging
using ProgressMeter, Dates
using QuasinormRegression
using Zenodo3968503Functions

#quick check on number of processes, should (hopefully) be larger than 1 or 2
if Threads.nthreads() <= 2
    println("The number of available threads is quite low at $(Threads.nthreads()), do you want to continue or do you want to fix [y/N]? ")
    any(readline() .== ["","N","n","no","NO","No"]) ? error("Stopping... refer to Multithreading in Julia docs (https://docs.julialang.org) on how to set number of threads") : println("Potentially bad idea but continuing...")
else
    print("\nThe number of available threads is set at $(Threads.nthreads()) out of ")
#    println("$(run(`nproc`))")
    run(pipeline(`nproc`, stdout=stdout, stderr=devnull));
    println("")
end

for BigIdx in collect(1:10)

    println("##################################################\n
            This is loop: $BigIdx in LinearRegModuleLoop\n
            ##################################################")
    #generate new set of simulation data
    include(scriptsdir("SimulationDataScript.jl"))

    #params file, which contains multiple sets of params typically
    paramfile = "ParamsFile_2022tbdJournalSims.jl"
    include(scriptsdir(paramfile))

    #to keep output clean, MLJ models have lot of Info/Warn messages
    Logging.disable_logging(Logging.Warn)
    println("Disabled Info and Warn logging -- this will negatively affect TBLogger/Tensorboard")

    #loop over all possible combination and save results, make thread safe
    compLock = ReentrantLock();
    println("There are $(size(param_combinations,1)) combinations to iterate over...")
    track_progress = Progress( size(param_combinations,1), barlen=50 )

    Threads.@threads for idx_params in param_combinations

        #create models and machines (either vanilla, SK, or MK)
        if idx_params["method"] == "Standard"
            mdl_x = idx_params["base_model"]()
            mdl_y = idx_params["base_model"]()
            mdl_z = idx_params["base_model"]()
        elseif idx_params["method"] == "SingleKernel"
            mdl_x = SingleKernelRegressor(mdl= idx_params["base_model"]())
            mdl_y = SingleKernelRegressor(mdl= idx_params["base_model"]())
            mdl_z = SingleKernelRegressor(mdl= idx_params["base_model"]())
        elseif idx_params["method"] == "MultiKernel"
            mdl_x = MultipleKernelRegressor(mdl= idx_params["base_model"]())
            mdl_y = MultipleKernelRegressor(mdl= idx_params["base_model"]())
            mdl_z = MultipleKernelRegressor(mdl= idx_params["base_model"]())
        else
            error("Check whether standard, single-kernel, or multi-kernel parameter")
        end
 
        mdl_base_type = MLJBase.name(idx_params["base_model"])
        mdl_type = idx_params["method"]
        #global mdl_type = MLJBase.name(mdl_x)

        #set params of models (utility function defined in kernel_functions.jl)
        setmdlparams!(mdl_x, idx_params)
        setmdlparams!(mdl_y, idx_params)
        setmdlparams!(mdl_z, idx_params)

        #load data file
        #figure out a way to unpack in function call (simply change function)
        @unpack avgmeas, flag100, flagval, verbose = idx_params["DataParams"]
        X_df, y_df, X_df_validation, y_df_validation = ReadZenodoMatFiles(joinpath(idx_params["dbpath"], idx_params["db"]), 
                                                                          avgmeas=avgmeas, flag100=flag100, flagval=flagval);

        #create a loop for multiple runs
        mee_runs = Float64[]; mee_x_runs = Float64[]; mee_y_runs = Float64[]; mee_z_runs = Float64[]; σee_runs = Float64[];
        for idx_runs in 1:idx_params["SamplingParams"].num_runs
  
            #create a loop for bagging situation
            yhat_x = yhat_y = yhat_z = zeros(size(X_df_validation,1))
            bagging_runs = idx_params["SamplingParams"].bagging_runs

            for idx_bagging in 1:bagging_runs
     
                ##get training and test sample indices, need to pass coordinates
                #getindices is a function in src/sampling_functions.jl
                train_idx, val_idx, test_idx = getindices(y_df, idx_params["train_size"], idx_params["SamplingParams"])
                #break out of both bagging and multiple run loop if indices won't work
                if train_idx== nothing @warn "train_idx is $train_idx"; @goto runloopbrk; end
      
                #set machine
                opt_mc_x = machine(mdl_x, X_df[train_idx, :], y_df[train_idx, :x])
                opt_mc_y = machine(mdl_y, X_df[train_idx, :], y_df[train_idx, :y])
                opt_mc_z = machine(mdl_z, X_df[train_idx, :], y_df[train_idx, :z])

                #fit the machines
                MLJ.fit!(opt_mc_x, verbosity=0)
                MLJ.fit!(opt_mc_y, verbosity=0);
                MLJ.fit!(opt_mc_z, verbosity=0);

                #predict
                yhat_x += MLJ.predict(opt_mc_x, X_df_validation)
                yhat_y += MLJ.predict(opt_mc_y, X_df_validation);
                yhat_z += MLJ.predict(opt_mc_z, X_df_validation);
            end #end of bagging loop

            #average the estimation
            yhat_x /= bagging_runs
            yhat_y /= bagging_runs
            yhat_z /= bagging_runs

            #mean euclidean distance error (and standard dev)
            mee = mean(sqrt.((yhat_x- y_df_validation[:, :x]).^(2) + (yhat_y - y_df_validation[:, :y]).^2 + (yhat_z - y_df_validation[:, :z]).^2))
            σee = std(sqrt.((yhat_x- y_df_validation[:, :x]).^(2) + (yhat_y - y_df_validation[:, :y]).^2 + (yhat_z - y_df_validation[:, :z]).^2))
            mee_x=mean(sqrt.((yhat_x- y_df_validation[:, :x]).^(2)))
            mee_y=mean(sqrt.((yhat_y- y_df_validation[:, :y]).^(2)))
            mee_z=mean(sqrt.((yhat_z- y_df_validation[:, :z]).^(2)))
            #save for μ and σ calcs
            push!(mee_runs, mee); push!(mee_x_runs, mee_x); push!(mee_y_runs, mee_y); push!(mee_z_runs, mee_z); push!(σee_runs, σee);

        end #end multiple run loop

        #get mean and average over multiple runs, don't track meex, meey, meez σ
        mee = mean(mee_runs); mee_x = mean(mee_x_runs); mee_y = mean(mee_y_runs); mee_z = mean(mee_z_runs); σee = mean(σee_runs);
        idx_params["SamplingParams"].num_runs>1 ?  mee_σ = std(mee_runs) : mee_σ = 0
    
        #make thread safe -- don't write to same file
        begin
            lock(compLock)
            try
                safesave(datadir("exp_pro","results", idx_params["db"], mdl_type, mdl_base_type, 
                             savename((dbname=idx_params["db"],mdlType=mdl_type,mdlBase=mdl_base_type), "jld2";   #keep name simpler for now
                                ignores=["packing_dir"])), ( merge((@strdict mdl_type mdl_base_type mee mee_σ σee mee_x mee_y mee_z), idx_params) )) 
            finally
                unlock(compLock)
            end
        end #end thread lock

        @label runloopbrk
        #tracking how much completed
        next!(track_progress)
    end #param combination loop

    #change logging level back to standard
    Logging.disable_logging(Logging.Debug);

    #save a record of this successful script run
#    @info "Saving a record of script for each database used..."
#    for idx in params["db"]
#        @info  *(Dates.format(now(), "yyyymmdd-HH:MM:SS_"), idx, params["method"], MLJBase.name(params["base_model"]), ".jld2") 
#        safesave(datadir("exp_pro", "testing_history",
#                   *(Dates.format(now(), "yyyymmdd-HH:MM:SS_"), idx, params["method"], MLJBase.name(params["base_model"]), ".jld2") ),  
#           params)
#    end #saving record loop

end #big idx loop

println("finished running LinearRegModuleLoop.jl")

end #end module
