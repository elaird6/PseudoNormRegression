using DrWatson

@quickactivate "PseudoNormRegression"

using Random, Distributions, CSV
using Plots, Logging
using KernelDensity
using Distances
#pyplot();
gr()
using LinearAlgebra: norm, diag
using DataFrames, Latexify
using QuasinormRegression
using Zenodo3968503Functions

#################################################################################
#plot data specific figures used in paper
#need data/results
#################################################################################
dbname_dict = Dict("JHUtdoa"=>"JHU TDoA", "JHUrss"=>"JHU RSS", "JHUaoaRAW"=>"JHU AoA", 
                   "TUD" => "TUD TDoA", "DSI1"=>"DSI1 RSS", "DSI2"=>"DSI2 RSS")
@show dbname_dict
mdlType_dict = Dict("Standard"=>"Std", "SingleKernel"=>"SK", "MultiKernel"=>"MK")
@show mdlType_dict
mdlBase_dict = Dict("KNNRegressor"=>"KNN", "RidgeRegressor"=>"Ridge", "LassoRegressor"=>"LASSO")
@show mdlBase_dict


#simple functions to pull out paramters from structs saved
#in result files
using MLJBase:name
getMetricP(x) = x.p
getDataParamsDataRep(x) = name(x.datarep)
getDataParamsAvgMeas(x) = x.avgmeas
getSampleParamSpherePacking(x) = x.sphere_packing
getSampleParamRandOffset(x) = x.rand_offset
getSampleParamBaggingRuns(x) = x.bagging_runs
getSampleParamGrpFlag(x) = x.grp_flag
function dropMissingCols(df::DataFrame)
  for idx in names(df)
    if length(unique(df[:,idx])) == 1 && ismissing(unique(df[:,idx])[1])
       select!(df, Not(idx))
    end
  end
end

#########################################################################
#function to plot results of data (assumes performance results are available)
# - plots results for psuedonorm regressions 
# - re-calling function (if using pyplot) will layer subsequent data on same
# plot 
# - plot variance or not based on passed params -- too much on same plot figure
#
# matfile - dir of results (dataset and directory have same name)
# mdlType - Standard, SingleKernel, MultiKernel
# mdlBase - KNNRegressor, RidgeRegressor, LassoRegressor
# varBool - boolean on whether to plot variance
# plt - name of plot figure on which to plot data
# legendposition - position of legend used in plot
# 
#########################################################################
#function is -- 58 lines
function plotdatachoice(matfile, mdlType, mdlBase, varBool, plt, legendposition)
#  dbname = split(matfile,".")[1]                  
  dbname = chop(matfile, head=0, tail=4)
  @info "$dbname: collecting formatting result files generated for $mdlType:$mdlBase regression"
  @info "$dbname: turning off warning/info messages while loading..."
  Logging.disable_logging(Logging.Warn)
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), subfolders=true, 
                       update=false, verbose=false)
  Logging.disable_logging(Logging.Debug)
  @info "$dbname: turning warning/info messages back on..."
  @info "$dbname: drop columns that only have missing values in them"
  dropMissingCols(df)
  dpcols = [:mee_x, :mee_y, :mee_z, :dbpath]
  @info "$dbname: drop these columns... $dpcols"
  for idx in dpcols select!(df, Not(idx)) end
  @info "$dbname: done collecting and formatting files for $mdlType:$mdlBase regression"
  @info "$dbname: transform some structs into parameters"
  @info "$dbname: sort by p"
  df_subset = subset(df, :mdl_base_type=>x->x.=="$mdlBase", :method=>x->x.=="$mdlType", 
                     :train_size=>x->x.==1)
  #this only happens if "Standard" which has metric with "p" in it
  if mdlType == "Standard"
    transform!(df_subset, :metric => ByRow(getMetricP) => :p)
  end
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamSpherePacking) => :sphere_packing)
  #remove any sphere packing
  df_subset = subset(df_subset, :sphere_packing=>x->x.==false)
  @info "$dbname: pull out and print best values"
  #if type is standard, need to change metric to p
  if mdlBase == "KNNRegressor"
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K]]
    df_subset = groupby(sort(df_subset,[:p, :K]), [:K, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestKstd = df_subset[bestGrpKey[2]][1,:K]
  else
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda]]
    df_subset = groupby(sort(df_subset,[:p, :lambda]), [:lambda, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
  end
  
  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(dbname_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(mdlType_dict, mdlType, mdlType)
  mdlBaseLeg = get(mdlBase_dict, mdlBase, mdlBase)
  #get best set of curves and plot
  sdf = combine(groupby(df_subset[bestGrpKey[2]], :p), :mee => mean => :mee, 
                :mee => std => :σ, :σee => mean => :σee)
  if varBool
    labelstr="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") 
    Plots.plot!(plt, sdf.p, sdf.mee, ribbon=(min.(sdf.mee,sdf.σee), sdf.σee), ls=:auto, ms=6, 
                shape=:auto, label=labelstr, legend_position=legendposition)
  else
    labelstr="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") 
    Plots.plot!(plt, sdf.p, sdf.mee, ls=:auto, shape=:auto, ms=6,
                label=labelstr, legend_position=legendposition)
  end 

  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)
end

function plotdatachoice(matfile, mdlType, mdlBase, p, K, spherePackBool, randOffset, 
        grpFlag, varBool, plt, legendposition)
  dbname = chop(matfile, head=0, tail=4)
  @info "$dbname: collecting formatting result files generated for $mdlType:$mdlBase regression"
  @info "$dbname: turning off warning/info messages while loading..."
  Logging.disable_logging(Logging.Warn)
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), subfolders=true, 
                       update=false, verbose=false)
  Logging.disable_logging(Logging.Debug)
  @info "$dbname: turning warning/info messages back on..."
  @info "$dbname: drop columns that only have missing values in them"
  dropMissingCols(df)
  dpcols = [:mee_x, :mee_y, :mee_z, :dbpath]
  @info "$dbname: drop these columns... $dpcols"
  for idx in dpcols select!(df, Not(idx)) end
  @info "$dbname: done collecting and formatting files for $mdlType:$mdlBase regression"
  @info "$dbname: transform some structs into parameters"
  @info "$dbname: sort by p"
  df_subset = subset(df, :mdl_base_type=>x->x.=="$mdlBase", :method=>x->x.=="$mdlType")
  #this only happens if "Standard" which has metric with "p" in it
  if mdlType == "Standard"
    transform!(df_subset, :metric => ByRow(getMetricP) => :p)
  end
#  @info "$dbname: unique p's:$(sort(unique(df_subset[:,:p]))) ##########################"
  df_subset = subset(df_subset, :p =>x->x.== p)
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamRandOffset) => :rand_offset)
  @info "$dbname: rand_offsets in data: $(unique(sort(df_subset[:, :rand_offset])))"
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamGrpFlag) => :grp_flag)
  @info "$dbname: grp_flag in data: $(unique(sort(df_subset[:, :grp_flag])))"
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamSpherePacking) => :sphere_packing)
  @info "$dbname: sphere_packing in data: $(unique(sort(df_subset[:, :sphere_packing])))"

  @warn "$dbname: using 2469 as max train data size, appropriate?"
  transform!(df_subset, :train_size => ByRow(x-> x==1 ? 2469 : x) => :train_size)
  trainVec = unique(sort(df_subset[:, :train_size]))
  @info "$dbname: train_size in data: $(unique(sort(df_subset[:, :train_size])))"
  @info "$dbname: pull out and print best values"
  #if type is standard, need to change metric to p
  if mdlBase == "KNNRegressor"
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K, :train_size]]
    @info "fixing K to be $K"
    @info "filtering results according to: K=$K, rand_offset=$randOffset, grp_flag=$grpFlag, sphere_packing=$spherePackBool"
    df_subset = subset(df_subset, :K=>x->x.==K, :rand_offset=>x->x.==randOffset, 
                       :grp_flag=>x->x.==grpFlag, :sphere_packing=>x->x.==spherePackBool)
    df_subset = groupby(sort(df_subset,[:train_size, :K]), [:K, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestKstd = df_subset[bestGrpKey[2]][1,:K]
  else
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda]]
    @info "filtering results according to: K=$K, rand_offset=$randOffset, grp_flag=$grpFlag, sphere_packing=$spherePackBool"
    df_subset = subset(df_subset, :rand_offset=>x->x.==randOffset, :grp_flag=>x->x.==grpFlag,
                      :sphere_packing=>x->x.==spherePackBool)
    df_subset = groupby(sort(df_subset,[:train_size, :lambda]), [:lambda, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
  end
  
  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(dbname_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(mdlType_dict, mdlType, mdlType)
  mdlBaseLeg = get(mdlBase_dict, mdlBase, mdlBase)
  #get best set of curves and plot
  sdf = combine(groupby(df_subset[bestGrpKey[2]], :train_size), :mee => mean => :mee, 
                :mee => std => :σ, :σee => mean => :σee)
  if varBool
    Plots.plot!(plt, sdf.train_size, sdf.mee, ribbon=(min.(sdf.mee,sdf.σee), sdf.σee), 
                ls=:auto, shape=:auto, xscale=:log10, ms=6,
                label="$mdlTypeLeg:$mdlBaseLeg "*plt_title*" p=$p"*( mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " "), 
                legend_position=legendposition)
  else
    ticks= sdf.train_size
    ticklabels =  [ "$x" for x in ticks]
    Plots.plot!(plt, sdf.train_size, sdf.mee, ls=:auto, shape=:auto, xscale=:log10, ms=6,
                label="$mdlTypeLeg:$mdlBaseLeg "*plt_title*" p=$p"*( mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " "), 
                legend_position=legendposition, xticks=(ticks, ticklabels))
  end 

  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)
end

########################################################################
#function to get top three values based on passed parameters
#passes out sorted dataframe and a latexify output
#
function bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing, 
    trainSize=nothing, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)

  dbname = chop(matfile, head=0, tail=4)
  @info "$dbname: collecting formatting result files generated for $mdlType:$mdlBase regression"
  @info "$dbname: turning off warning/info messages while loading..."
  Logging.disable_logging(Logging.Warn)
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), 
                       subfolders=true, update=false, verbose=false)
  Logging.disable_logging(Logging.Debug)
  @info "$dbname: turning warning/info messages back on..."
  @info "$dbname: drop columns that only have missing values in them"
  dropMissingCols(df)
  dpcols = [:mee_x, :mee_y, :mee_z, :dbpath]
  @info "$dbname: drop these columns... $dpcols"
  for idx in dpcols select!(df, Not(idx)) end
  @info "$dbname: done collecting and formatting files for $mdlType:$mdlBase regression"
  @info "$dbname: transform some structs into parameters"
  @info "$dbname: sort by p"
  df_subset = subset(df, :mdl_base_type=>x->x.=="$mdlBase", :method=>x->x.=="$mdlType")
  #this only happens if "Standard" which has metric with "p" in it
  if mdlType == "Standard"
    transform!(df_subset, :metric => ByRow(getMetricP) => :p)
  end
  @info "$dbname: p in data: $(unique(sort(df_subset[:, :p])))"
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamRandOffset) => :rand_offset)
  @info "$dbname: rand_offsets in data: $(unique(sort(df_subset[:, :rand_offset])))"
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamGrpFlag) => :grp_flag)
  @info "$dbname: grp_flag in data: $(unique(sort(df_subset[:, :grp_flag])))"
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamSpherePacking) => :sphere_packing)
  @info "$dbname: sphere_packing in data: $(unique(sort(df_subset[:, :sphere_packing])))"
  @warn "$dbname: using 2469 as max train data size, appropriate?"
  transform!(df_subset, :train_size => ByRow(x-> x==1 ? 2469 : x) => :train_size)
  @info "$dbname: train_size in data: $(unique(sort(df_subset[:, :train_size])))"

  #now, start filtering based on passed parameters
  if !isnothing(p) df_subset = subset(df_subset, :p =>x->x.== p) end
  if !isnothing(K) df_subset = subset(df_subset, :K =>x->x.== K) end
  if !isnothing(spherePackBool) df_subset = subset(df_subset, :sphere_packing =>x->x.== spherePackBool) end
  if !isnothing(randOffset) df_subset = subset(df_subset, :rand_offset =>x->x.== randOffset) end
  if !isnothing(grpFlag) df_subset = subset(df_subset, :grp_flag =>x->x.== grpFlag) end
  if !isnothing(trainSize) df_subset = subset(df_subset, :train_size =>x->x.== trainSize) end

  @info "$dbname: pull out and print best values"

  #function bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing,
  #    trainSize, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)

  if mdlBase == "KNNRegressor"
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K, :train_size, 
                                       :sphere_packing, :rand_offset, :grp_flag]]
  else
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda, :train_size, 
                                       :sphere_packing, :rand_offset, :grp_flag]]
  end
  

  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)
  
  return df_subset
end #bestresults function
############################################################################
#function to plot average distance between nearest training points
#primarily aimed at (sorta) quantifying performance against training density
#
function plotdensitycurve(matfile, mdlType, mdlBase, plt, legendposition)
#  dbname = split(matfile,".")[1]                  
  dbname = chop(matfile, head=0, tail=4)
  @info "$dbname: collecting formatting result files generated for $mdlType:$mdlBase regression"
  @info "$dbname: turning off warning/info messages while loading..."
  Logging.disable_logging(Logging.Warn)
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), subfolders=true, 
                       update=false, verbose=false)
  Logging.disable_logging(Logging.Debug)
  @info "$dbname: turning warning/info messages back on..."
  @info "$dbname: drop columns that only have missing values in them"
  dropMissingCols(df)
  dpcols = [:mee_x, :mee_y, :mee_z, :dbpath]
  @info "$dbname: drop these columns... $dpcols"
  for idx in dpcols select!(df, Not(idx)) end
  
  #get coordinates of training samples
  _, y_df, _, _= ReadZenodoMatFiles(datadir("exp_pro", matfile))
  
  #get different train_sizes
  trainVec = unique(sort(df[:,:train_size]))
  if trainVec[1] == 1
#    append!(trainVec, popfirst!(trainVec))
    popfirst!(trainVec)
  end
  @info "trainVec: $trainVec"

  #setup vector to hold mean distance
  meanDisVec = []
  #loop over different training sizes
  for idx in trainVec
    if idx != 1
      train_idx, _, _= getindices(y_df, idx, 
                                  SampleParamsStruct(sphere_packing=true, 
                                                     packing_dir=datadir("packing_coords"), 
                                                     rand_offset=1.5, grp_flag=false, 
                                                     bagging_runs=1));
    else
      train_idx, _, _= getindices(y_df, idx, 
                                  SampleParamsStruct(sphere_packing=false, 
                                                     packing_dir=datadir("packing_coords"), 
                                                     rand_offset=1.5, grp_flag=false, 
                                                     bagging_runs=1));

    end
    y_df_sphereLocs = y_df[train_idx, [:x, :y]];
   
    #set up euclidean calculation using kernel matrix
    dKt = KernelFunctions.compose(PnormKernelStd(2.0), KernelFunctions.ScaleTransform(1))
    #get euclidean kernel matrix
    dKKt=KernelFunctions.kernelmatrix(dKt, Tables.matrix(y_df_sphereLocs), obsdim=1);

    #get all distances
    tmp = []
#    @info size(dKKt,1)
    for idx in 1:size(dKKt,1)
      append!(tmp, [diag(dKKt,idx)...])
      append!(tmp, [diag(dKKt,-idx)...])
    end
    #sort, get train_size closest ones
    append!(meanDisVec, mean(sort(tmp)[1:size(dKKt,1)]))

  end #end loop over trainvec
  #fix the "1" in the trainVec
  if trainVec[end]==1 trainVec[end]=size(y_df,1) end

  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(dbname_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(mdlType_dict, mdlType, mdlType)
  mdlBaseLeg = get(mdlBase_dict, mdlBase, mdlBase)

  #now plot
  @info trainVec, meanDisVec
  Plots.plot!(plt, trainVec, meanDisVec, ls=:solid, lc=:black, shape=:none, xscale=:log10, 
              label="Avg Sample Spacing", legend_position=legendposition)

end #plotdensity function

###########################################################################
#function to plot original reference measurements and the new denoised
#measurements
#
function plotmeasurements(matfileO, matfileN, legendposition)

  #get coordinates of training samples
  X_dfO, _, _, _= ReadZenodoMatFiles(datadir("exp_pro", matfileO))
  X_dfN, _, _, _= ReadZenodoMatFiles(datadir("exp_pro", matfileN))

  @info "size of $matfileO is $(size(X_dfO))"
  @info "size of $matfileN is $(size(X_dfN))"
  #now plot all measurements
  idx1, idx2 = rand(collect(1:size(X_dfO,1)), 2)
  idx1, idx2 = (1051, 102)
  @info "indices: idx1:$idx1, idx2:$idx2"
  plt1 = Plots.plot(Vector(X_dfO[idx1, : ]), 
                    #shape=:circle,msa=0.0,
                    label="original", legend_position=legendposition)
  Plots.plot!(plt1, Vector(X_dfN[idx1, : ]), 
              #ma=0.2, msa=0.0, shape=:circle, 
              label="denoised", legend_position=legendposition)

  plt2 = Plots.plot(Vector(X_dfO[idx2, : ]), 
                    #seriestype=:sticks,
                    label="original", legend_position=legendposition)
  Plots.plot!(plt2, Vector(X_dfN[idx2, : ]), 
              #seriestype=:sticks, #ma=0.2, msa=0.0, shape=:circle, 
              label="denoised", legend_position=legendposition)

  return [plt1 plt2]
end

############################################################################
#function to plot sphere packing training locations
#
function plotspherelocs(matfile, trainSize, grpFlag, randOffset, plt, legendposition)
  
  #get coordinates of training samples
  _, y_df, _, _= ReadZenodoMatFiles(datadir("exp_pro", matfile))
  
  #loop over different training sizes
  train_idx, _, _= getindices(y_df, trainSize, 
                              SampleParamsStruct(sphere_packing=true, 
                                                 packing_dir=datadir("packing_coords"), 
                                                 rand_offset=randOffset, grp_flag=grpFlag, 
                                                 bagging_runs=1));
  y_df_sphereLocs = y_df[train_idx, [:x, :y]];
   
  #now plot all locations
  Plots.scatter!(plt, y_df[:,:x], -y_df[:,:y], ma=0.2, msa=0.0, shape=:circle, 
                 label=nothing, legend_position=legendposition)
  #plot sphere packing locations
  Plots.scatter!(plt, y_df_sphereLocs[:,:x], -y_df_sphereLocs[:,:y], ma=0.7, msa=0.0, 
                 shape=:circle, label=nothing, legend_position=legendposition, xaxis=(-5,25))

end

pyplot()

#plot sample locations and sensors
sensor_locs = CSV.read(datadir("exp_raw", "JHU_locations.csv"), DataFrame)
_, TDOAlocs, _, _ = ReadZenodoMatFiles(datadir("exp_pro", "JHUtdoa.mat"));
#plt_testbed = Plots.plot(xlabel="East-West (m)", ylabel="North-South (m)", 
plt_testbed = Plots.plot(xlabel="⟵  ∼30 meters  ⟶", ylabel="⟵  ∼70 meters  ⟶" ,
                         reuse=false, grid=false, aspect_ratio=:equal, 
                         frame=:none, size=(400,600), legend_position=:outertop)
Plots.scatter!(plt_testbed, TDOAlocs[:, :x], -TDOAlocs[:, :y], ma=0.2, msa=0.0,
               shape=:circle, label="measurement locations")
Plots.scatter!(plt_testbed, sensor_locs[:, :x], -sensor_locs[:, :y], ma=0.7, msa=0.0,
               shape=:circle, label="sensors locations")
display(plt_testbed)
print("Do you want to save testbed figure? (y/N) ");
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_testbed.pdf") : println("no save")

Plots.scatter!(plt_testbed, [TDOAlocs[1051, :x]], [-TDOAlocs[10, :y]], ma=0.15, msa=0.0,
               shape=:circle, ms=155, label=nothing)
display(plt_testbed)
print("Do you want to save testbed figure? (y/N) ");
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_sensorimposedoutlier.pdf") : println("no save")


#plot mean, variance of best std/sk ridge/lasso/k-nn performance for each dataset
for dbname in ("JHUtdoa.mat", "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat")
  plt_test = Plots.plot()
  plotdatachoice(dbname, "Standard", "KNNRegressor", true, plt_test, :topleft)
  plotdatachoice(dbname, "SingleKernel", "KNNRegressor", true, plt_test, :topleft)
  try
    plotdatachoice(dbname, "SingleKernel", "RidgeRegressor", true, plt_test, :topleft)
  catch 
    @warn "No RidgeRegressor for $dbname ..." 
  end
  try
    plotdatachoice(dbname, "SingleKernel", "LassoRegressor", true, plt_test, :topleft)
  catch 
    @warn "No LassoRegressor for $dbname ..." 
  end
  display(plt_test)
  print("Do you want to save $dbname figure? (y/N) ");
  #empty string is "identity" so check if empty and if is, set to default
  getinput = lowercase(readline())
  contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_$(dbname)Pnorms.pdf") : println("no save")
end

#function plotmeasurements(matfileO, matfileN, plt, legendposition)
plt_vec = plotmeasurements("JHUtdoa.mat", "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", :bottomright)
plt_measurements = Plots.plot(plt_vec..., layout=(2,1), xlabel=["" "measurements index"], ylabel="TDE (ns)", reuse=false, grid=false)
display(plt_measurements)
print("Do you want to save the measurement outlier figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_TDoAoutlier.pdf") : println("no save")

#plotting preprocessing
plt_Elast = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title="ElastPreprocess")
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.001.mat", "Standard", "KNNRegressor", false, 
#               plt_Elast, :topleft)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor", false, 
#               plt_Elast, :topleft)
plotdatachoice("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "Standard", "KNNRegressor", false, 
               plt_Elast, :topleft)
plotdatachoice("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor", false, 
               plt_Elast, :topleft)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.0001.mat", "Standard", "KNNRegressor", false, 
#               plt_Elast, :topleft)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.0001.mat", "SingleKernel", "KNNRegressor", false, 
#               plt_Elast, :topleft)
plotdatachoice("JHUtdoa.mat", "Standard", "KNNRegressor", false, plt_Elast, :topleft)
plotdatachoice("JHUtdoa.mat", "SingleKernel", "KNNRegressor", false, plt_Elast, :topleft)
display(plt_Elast)


#randOffsetVal=1.5; Kval=7; grpFlagBool=true; varPlot=false
randOffsetVal=0.7; Kval=7; grpFlagBool=true; varPlot=false; sphereBool=true;

p=2.0
filename= "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat"
plt_ElastTrain = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=false, 
                            title="ElastPreprocess, Train Size")
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain, :topright)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright)
filename= "JHUtdoa.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain, :topright)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright)
plotdensitycurve(filename, "SingleKernel", "KNNRegressor", plt_ElastTrain, :topright)
plt_ElastTrain.series_list[1][:label] = "Std:KNN($Kval),denoise"
#plt_ElastTrain.series_list[2][:label] = "K:KNN($Kval),denoise"
plt_ElastTrain.series_list[2][:label] = "K:Ridge,denoise"
plt_ElastTrain.series_list[3][:label] = "Std:KNN($Kval)"
#plt_ElastTrain.series_list[5][:label] = "K:KNN($Kval)"
plt_ElastTrain.series_list[4][:label] = "K:Ridge"
plt_ElastTrain.series_list[5][:label] = "Avg Sample Spacing"
title!(plt_ElastTrain, " ")
display(plt_ElastTrain)
print("Do you want to save the last sphere packing curve figure (L2: K=$Kval, spherePacking=$sphereBool, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_L2DenoiseSpherePackingK=$(Kval)spherePacking=$(sphereBool)RandOff=$(randOffsetVal).pdf") : println("no save")

#function plotdatachoice(matfile, mdlType, mdlBase, p, K, randOffset, grpFlag, varBool, plt, legendposition)

p=1.05
plt_ElastTrain1 = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=false, 
                             title="DenoisePreprocess, Train Size, p=$p")
filename= "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
filename= "JHUtdoa.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright)
plotdensitycurve(filename,  "SingleKernel", "KNNRegressor", plt_ElastTrain1, :topright)
display(plt_ElastTrain1)
print("Do you want to save the last sphere packing curve figure (L1: K=$Kval, spherePacking=$sphereBool, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_L1DenoiseSpherePackingK=$(Kval)spherePacking=$(sphereBool)RandOff=$(randOffsetVal).pdf") : println("no save")



#plot example training locations, locations chosen by circle packing method
plt_SphereLocs = Plots.plot(xlabel="East-West (m)", ylabel="North-South (m)", reuse=false, grid=false, aspect_ratio=:equal,
                            frame=:none, size=(400,600))
numS=35
plotspherelocs("JHUtdoa.mat", numS, grpFlagBool, randOffsetVal, plt_SphereLocs, :outerright)
display(plt_SphereLocs)
print("Do you want to save the last sphere packing locs figure (numS=$numS, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_SphereLocsPlotNumS=$(numS)randOff=$(randOffsetVal).pdf") : println("no save")

                   
#generate histogram of TDoA dataset to use
#this function calculates percentage of outliers based on provided limit
#returns both percentage and filtered values
function outlier(data::Vector, limit::Integer)
    if limit ≤ 0 error("limit ≤ 0, should be positive") end
    origsize = size(data,1)
    filtdata = filter(x->x.>-limit, data)
    filter!(x->x.<limit, filtdata)
    finalsize = size(filtdata,1)
    return ((1-finalsize/origsize)*100, filtdata) 
end #outlier function

function calcIdealTDOA(sampleLocs::Array, sensorLocs::Array; tol::Real=0.5)
    #generate distances from sampleLocs to sensorLocs,
    #change distance to nanoseconds (multiply by 3)
    euc_dist = pairwise(Euclidean(), sampleLocs, sensorLocs).*3.0;

    #now get differential distance (tdoa)
    #set container for diff distance
    diff_dist = zeros(size(euc_dist,1), binomial(size(euc_dist,2),2))
    #iterate over samples to get diff distance values
    count=1
    for o_idx in 1:(size(euc_dist,2)-1)
        for i_idx in (o_idx+1):size(euc_dist,2)
            diff_dist[:,count]=euc_dist[:,o_idx] .- euc_dist[:,i_idx]
            count+=1
        end
    end
    return filter!(x->abs(x).>tol, [Matrix(diff_dist)...])
end #end of calcIdealTDOA function

#get tdoa empirical values 
TDOAvals, TDOAlocs, _, _ = ReadZenodoMatFiles(datadir("exp_pro", "JHUtdoa.mat"));
TDOAvals = [Matrix(TDOAvals)...];
#get sensor locactions
sensor_locs = CSV.read(datadir("exp_raw", "JHU_locations.csv"), DataFrame)
#get ideal tdoa values based on empirical locations
TDOAidealvals = calcIdealTDOA(Array(Matrix(TDOAlocs[:, [:x,:y]])'), 
                              Array(Matrix(sensor_locs[:, [:x, :y]])'))

limplotval = 250; 
outlier250percentage, _ = outlier(TDOAvals, limplotval)
_, TDOAvalsplot = outlier(TDOAvals, 750)
plt_histTDOA = Plots.histogram(TDOAvalsplot, normalize=:true, label="empirical", 
                               la=0, alpha=0.8); 
Plots.histogram!(plt_histTDOA, TDOAidealvals, normalize=:true, label="ideal", 
                 la=0, alpha=0.7); 
Plots.vline!(plt_histTDOA, [-limplotval,limplotval], label="outlier\ndemarcation", 
             ylabel="Density",xlabel="TDoA estimates (ns)", lw=3, ls=:dash)
display(plt_histTDOA)
println("Outlier percentage is: $outlier250percentage")
print("Do you want to save the TDoA histogram? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_TDOAhistogram.pdf") : println("no save")



#create basic table used in paper
println("bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing, trainSize=nothing, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)")
dfbest1 = bestresults("JHUtdoa.mat", "Standard", "KNNRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
dfbest2 = bestresults("JHUtdoa.mat", "SingleKernel", "KNNRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
dfbest2r= bestresults("JHUtdoa.mat", "SingleKernel", "RidgeRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
dfbest3 = bestresults("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "Standard", "KNNRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
dfbest4 = bestresults("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
dfbest4r= bestresults("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "SingleKernel", "RidgeRegressor"; 
                      p=2.0, K=nothing, trainSize=nothing, spherePackBool=nothing, 
                      randOffset=nothing, grpFlag=nothing);
@info "latexify(df, env = :tabular, fmt = \"%0.2f\")"
cols = (:db, :method,:mdl_base_type, :mee, :σee, :p, :K)
@info "using these cols: $cols" 
dfbest = DataFrames.DataFrame()
append!(dfbest, first(dfbest1[:,[cols...]],1))
append!(dfbest, first(dfbest2[:,[cols...]],1))
append!(dfbest, first(dfbest2r[:,[cols...]],1))
append!(dfbest, first(dfbest3[:,[cols...]],1))
append!(dfbest, first(dfbest4[:,[cols...]],1))
append!(dfbest, first(dfbest4r[:,[cols...]],1))
#remove .mat from db filename, breaks latexify which attempts to parse as
#latex equation
transform!(dfbest, :db => ByRow(x->chop(x,tail=4))=>:db)
@show dfbest;
println("this is raw latex table of results shown just above used in paper prior to cleaning up names and adding in table bells and whistles")
latexify(dfbest, env = :tabular, fmt = "%0.2f")
