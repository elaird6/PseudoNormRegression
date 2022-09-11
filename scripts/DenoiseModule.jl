module DenoiseModule

using DrWatson
@quickactivate "PseudoNormRegression"

using QuasinormRegression
using Zenodo3968503Functions
using ProgressMeter, Logging
using DataFrames, MAT

#should probably turn this into a local source package
include(scriptsdir("JDE.jl"))
import .JDE

#prompt for params file, which contains multiple sets of params typically
println("Enter in params file (e.g. ParamsFile_2022asilomarDenoise.jl): ")
paramfile = readline()
include(scriptsdir(paramfile))

#quick check on number of processes, should (hopefully) be larger than 1 or 2
if Threads.nthreads() <= 2
    println("The number of processing threads is quite low at $(Threads.nthreads()), do you want to continue or do you want to fix [y/N]? ")
    any(readline() .== ["","N","n","no","NO","No"]) ? error("Stopping... refer to Multithreading in Julia docs (https://docs.julialang.org) on how to set number of threads") : println("Potentially bad idea but continuing...")
else
    print("\nThe number of processing threads is set at $(Threads.nthreads()) out of ")
#    println("$(run(`nproc`))")
    run(pipeline(`nproc`, stdout=stdout, stderr=devnull));
    println("")
end
        
#to keep output clean, MLJ models have lot of Info/Warn messages
Logging.disable_logging(Logging.Warn)
println("Disabled Info and Warn loggings for the most part-- this will negatively affect TBLogger/Tensorboard if used")
println("Will still get some Proximal GD convergence warnings... can ignore")

#loop over all possible combination and save results, make thread safe
println("There are $(size(param_combinations,1)) combinations to iterate over...\n")

for idx_params in param_combinations

  filename = idx_params["db"]
  X_df, y_df, X_df_validation, y_df_validation  = ReadZenodoMatFiles(datadir(idx_params["dbpath"], filename))
 
  #get col names and types for creation of transformed dataframes
  col_type = eltype.(eachcol(X_df))
  col_name = Symbol.(names(X_df))
  named_tuple = (; zip(col_name, type[] for type in col_type)...)

  #transform training dataframe
  X_dfnew = DataFrames.DataFrame(named_tuple)
  @showprogress "Iterating over $filename training data..." for (idx, xrow) in enumerate(eachrow(X_df))
    machMLJ = JDE.denoise((xrow), idx_params["λ"], idx_params["γ"])
    push!(X_dfnew, machMLJ)
  end
  # transform validation dataframe
  X_dfVnew= DataFrames.DataFrame(named_tuple)
  @showprogress "Iterating over $filename validation data..." for (idx, xrow) in enumerate(eachrow(X_df_validation))
    machMLJ = JDE.denoise((xrow), idx_params["λ"], idx_params["γ"])
    push!(X_dfVnew, machMLJ)
  end

  #save the dataframes
  #create database dictionary in format use by Zenodo
  database = Dict("trainingMacs"=>Matrix(X_dfnew),
                     "testMacs"=>Matrix(X_dfVnew),
                     "trainingLabels"=>Matrix(y_df),
                     "testLabels"=>Matrix(y_df_validation))
  database_wrap = Dict("database"=>database)
  
  newname = savename(split(filename,".")[1]*idx_params["prefix"], idx_params, "mat"; ignores=["db","dbpath","prefix"])

  MAT.matwrite(datadir(idx_params["dbpath"], newname), database_wrap);
  println("Saved $newname ...")

end #param combination loop (thread lock)

#change logging level back to standard
Logging.disable_logging(Logging.Debug);

println("finished running DenoiseModule (creating denoised measurement files) using $paramfile")


end #module
