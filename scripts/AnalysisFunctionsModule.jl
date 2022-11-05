module AnalysisFunctionsModule

#list of modules to import
using DrWatson
@quickactivate "PseudoNormRegression"

using MLJBase:name
using DataFrames
using Logging
using LinearAlgebra: diag
using Statistics: mean, std
#using Random, Distributions, CSV
using Plots 
using KernelDensity
using Distances
using QuasinormRegression
using Zenodo3968503Functions


#list functions to export
export getMetricP
export getDataParamsDataRep
export getDataParamsAvgMeas
export getSampleParamSpherePacking
export getSampleParamRandOffset
export getSampleParamBaggingRuns
export getSampleParamGrpFlag
export dropMissingCols
export plotdatachoice
export bestresults
export plotdensitycurve
export plotspherelocs
export plotmeasurements
export outlier
export calcIdealTDOA
export calcIdealRSS
export kerneldistancesort

#simple functions to pull out parameters from structs saved
#in result files
getMetricP(x) = x.p 
getDataParamsDataRep(x) = name(x.datarep)
getDataParamsAvgMeas(x) = x.avgmeas
getSampleParamSpherePacking(x) = x.sphere_packing
getSampleParamRandOffset(x) = x.rand_offset
getSampleParamBaggingRuns(x) = x.bagging_runs
getSampleParamGrpFlag(x) = x.grp_flag
getUnique(x) = unique(x)[1]
#remove columns that have only missing values
function dropMissingCols(df::DataFrame)
  for idx in names(df)
    if length(unique(df[:,idx])) == 1 && ismissing(unique(df[:,idx])[1])
       select!(df, Not(idx))
    end 
  end 
end


"""
function plotdatachoice(matfile, mdlType, mdlBase, varBool, plt, legendposition, 
        label_dict; passedlabel=nothing)
        
Plot performance results against Lp (assumes performance results are available)
 - plots best results for quasinorm regressions 
 - removes sphere packing
 - re-calling function (if using pyplot) will layer subsequent data on same
 plot 
 - plot variance or not based on passed params -- too much on same plot figure

 matfile - dir of results (dataset and directory have same name)
 mdlType - Standard, SingleKernel, MultiKernel
 mdlBase - KNNRegressor, RidgeRegressor, LassoRegressor
 varBool - boolean on whether to plot variance
 plt - name of plot figure on which to plot data
 legendposition - position of legend used in plot
 label_dict - substitutes in text name for database
 passedlabel - manual substitution for database name -- overrides label_dict
 
"""
function plotdatachoice(matfile, mdlType, mdlBase, varBool, plt, legendposition, 
        label_dict; passedlabel=nothing)
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
  df_subset = subset(df, :mdl_base_type=>x->x.=="$mdlBase", :method=>x->x.=="$mdlType", 
                     :train_size=>x->x.==1)
  #this only happens if "Standard" which has metric with "p" in it
  if mdlType == "Standard"
    transform!(df_subset, :metric => ByRow(getMetricP) => :p)
  end
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  transform!(df_subset, 
             :SamplingParams => ByRow(getSampleParamSpherePacking) => :sphere_packing)

  #remove any sphere packing
  df_subset = subset(df_subset, :sphere_packing=>x->x.==false)
  @info "$dbname: pull out and print best values"
  #if type is standard, need to change metric to p
  if mdlBase == "KNNRegressor"
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K]]
    #filtering out multikernel p's which are not identical
    if mdlType == "MultiKernel"
        @warn "*********************************************************
        filtering out multikernel results that have different p's,
        unable to sort for plotting purposes.
        *********************************************************"
        df_subset = subset(df_subset, :p=>x->size.(unique.(x),1).==1)
        rename!(df_subset, :p => :p_array)
        #replacing array of identical p's with scalar p. should throw error if
        #array of different p's make it through filter
        transform!(df_subset, :p_array => ByRow(getUnique) => :p)
    end
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K]]
    df_subset = groupby(sort(df_subset,[:p, :K]), [:K, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestKstd = df_subset[bestGrpKey[2]][1,:K]
  else
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda]]
    #filtering out multikernel p's which are not identical
    if mdlType == "MultiKernel"
        @warn "*********************************************************
        filtering out multikernel results that have different p's,
        unable to sort for plotting purposes.
        *********************************************************"
        df_subset = subset(df_subset, :p=>x->size.(unique.(x),1).==1)
        rename!(df_subset, :p => :p_array)
        #replacing array of identical p's with scalar p. should throw error if
        #array of different p's make it through filter
        transform!(df_subset, :p_array => ByRow(getUnique) => :p)
    end
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K]]
    df_subset = groupby(sort(df_subset,[:p, :lambda]), [:lambda, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
  end
  
  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(label_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(label_dict, mdlType, mdlType)
  mdlBaseLeg = get(label_dict, mdlBase, mdlBase)
  #get best set of curves and plot
  sdf = combine(groupby(df_subset[bestGrpKey[2]], :p), :mee => mean => :mee, 
                :mee => std => :σ, :σee => mean => :σee)
  if varBool
    labelstr="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") 
    isnothing(passedlabel) ? _ = labelstr : labelstr = passedlabel
    Plots.plot!(plt, sdf.p, sdf.mee, ribbon=(min.(sdf.mee,sdf.σee), sdf.σee), ls=:auto, ms=6, 
                shape=:auto, label=labelstr, legend_position=legendposition)
  else
    labelstr="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") 
    isnothing(passedlabel) ? _ = labelstr : labelstr = passedlabel
    Plots.plot!(plt, sdf.p, sdf.mee, ls=:auto, shape=:auto, ms=6,
                label=labelstr, legend_position=legendposition)
  end 

  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)

end #function plotdatachoice

"""
plotdatachoice(matfile, mdlType, mdlBase, p, K, spherePackBool, randOffset, 
        grpFlag, varBool, plt, legendposition, label_dict; passedlabel=nothing)

Plot performance curve against trainsize as filtered by the majority of other
parameters. Also passed is plot handle, plot label, and a renaming label dictionary.

 matfile - dir of results (dataset and directory have same name)
 mdlType - Standard, SingleKernel, MultiKernel
 mdlBase - KNNRegressor, RidgeRegressor, LassoRegressor
 p       - filter results (0.1 : 2.0 length 21)
 K       - filter results (1,3,5, 7 are typical, appropriate for kNN)
 spherePackBool - filter results (true/false values)
 randoffset- filter results (0.7, 1.5 are typical values)
 grpFlag   - filter results (true/false values)
 varBool - boolean on whether to plot variance
 plt - name of plot figure on which to plot data
 legendposition - position of legend used in plot
 label_dict - substitutes in text name for database
 passedlabel - manual substitution for database name -- overrides label_dict
 
"""
function plotdatachoice(matfile, mdlType, mdlBase, p, K, spherePackBool, randOffset, 
        grpFlag, varBool, plt, legendposition, label_dict; passedlabel=nothing)
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
  @info "$dbname: unique p's:$(sort(unique(df_subset[:,:p])))"
  df_subset = subset(df_subset, :p =>x->x.== p)
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamRandOffset) => :rand_offset)
  @info "$dbname: rand_offsets in data: $(unique(sort(df_subset[:, :rand_offset])))"
  transform!(df_subset, :SamplingParams => ByRow(getSampleParamGrpFlag) => :grp_flag)
  @info "$dbname: grp_flag in data: $(unique(sort(df_subset[:, :grp_flag])))"
  transform!(df_subset, 
             :SamplingParams => ByRow(getSampleParamSpherePacking) => :sphere_packing)
  @info "$dbname: sphere_packing in data: $(unique(sort(df_subset[:, :sphere_packing])))"

  #this warning is for the transformation in next line
  @warn "$dbname: using 2469 as max train data size, appropriate?"
  transform!(df_subset, :train_size => ByRow(x-> x==1 ? 2469 : x) => :train_size)
  trainVec = unique(sort(df_subset[:, :train_size]))
  @info "$dbname: train_size in data: $(unique(sort(df_subset[:, :train_size])))"
  @info "$dbname: pull out and print best values"
  #if type is standard, need to change metric to p
  if mdlBase == "KNNRegressor"
    @show first(sort!(df_subset, :mee),3)[:, [:mee, :σee, :mee_σ, :p, :K, :train_size]]
    @info "fixing K to be $K"
    @info "filtering results according to: K=$K, rand_offset=$randOffset, grp_flag=$grpFlag, sphere_packing=$spherePackBool"
    df_subset = subset(df_subset, :K=>x->x.==K, :rand_offset=>x->x.==randOffset, 
                       :grp_flag=>x->x.==grpFlag, :sphere_packing=>x->x.==spherePackBool)
    #group by trainsize and k
    #this is pointless.  already filtered on k
    df_subset = groupby(sort(df_subset,[:train_size, :K]), [:K, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    #find the group that has minimum mee across the (trainsize, k) groups
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    #remember that K value
    bestKstd = df_subset[bestGrpKey[2]][1,:K]
  else
    @show first(sort!(df_subset, :mee),3)[:, [:mee, :σee, :mee_σ, :p, :lambda]]
    @info "filtering results according to: K=$K, rand_offset=$randOffset, grp_flag=$grpFlag, sphere_packing=$spherePackBool"
    df_subset = subset(df_subset, :rand_offset=>x->x.==randOffset, 
                       :grp_flag=>x->x.==grpFlag, :sphere_packing=>x->x.==spherePackBool)
    df_subset = groupby(sort(df_subset,[:train_size, :lambda]), [:lambda, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
  end
  
  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(label_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(label_dict, mdlType, mdlType)
  mdlBaseLeg = get(label_dict, mdlBase, mdlBase)
  #get best set of curves and plot
  sdf = combine(groupby(df_subset[bestGrpKey[2]], :train_size), :mee => mean => :mee, 
                :mee => std => :σ, :σee => mean => :σee)
  if varBool
    isnothing(passedlabel) ? labelstr= "$mdlTypeLeg:$mdlBaseLeg "*plt_title*" p=$p"*( mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") : labelstr=passedlabel
    Plots.plot!(plt, sdf.train_size, sdf.mee, ribbon=(min.(sdf.mee,sdf.σee), sdf.σee), 
                ls=:auto, shape=:auto, xscale=:log10, ms=6,
                label=labelstr,
                legend_position=legendposition)
  else
    isnothing(passedlabel) ? labelstr= "$mdlTypeLeg:$mdlBaseLeg "*plt_title*" p=$p"*( mdlBase=="KNNRegressor" ? " k=$(bestKstd)" : " ") : labelstr=passedlabel
    ticks= sdf.train_size
    ticklabels =  [ "$x" for x in ticks]
    Plots.plot!(plt, sdf.train_size, sdf.mee, ls=:auto, shape=:auto, xscale=:log10, ms=6,
                label=labelstr,
                legend_position=legendposition, xticks=(ticks, ticklabels))
  end 

  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)

end #function plotdatachoice

"""
function bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing, 
    trainSize=nothing, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)

function to print top three values based on passed parameters, 
returns a sorted dataframe of all the values
"""
function bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing, 
    trainSize=nothing, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)

  dbname = chop(matfile, head=0, tail=4)

  @info "############### best results for these parameters ###############"
  @info ""
  @info "$dbname : $mdlType:$mdlBase : p=$p, K=$K, trainSize=$trainSize"
  @info "         spherePackBool=$spherePackBool, randOffset=$randOffset, grpFlag=$grpFlag"
  @info ""
  @info "############### best results for these parameters ###############"

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
  df_subset = subset(df, :mdl_base_type=>x->x.=="$mdlBase", :method=>x->x.=="$mdlType")
  #this only happens if "Standard" which has metric with "p" in it
  if mdlType == "Standard"
    transform!(df_subset, :metric => ByRow(getMetricP) => :p)
  end
  if mdlType != "MultiKernel"
      @info "$dbname: p in data: $(unique(sort(df_subset[:, :p])))"
  else
      @info "$dbname: p in data: $(unique(df_subset[:, :p]))"
  end
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
  if mdlType != "MultiKernel"
      if !isnothing(p) df_subset = subset(df_subset, :p =>x->x.== p) end
  else
      #simplify.  if passed p, then assume p-array is same
      if !isnothing(p) df_subset = subset(df_subset, :p=>x->getUnique.(x).==p) end
  end
  if !isnothing(K) df_subset = subset(df_subset, :K =>x->x.== K) end
  if !isnothing(spherePackBool) df_subset = subset(df_subset, :sphere_packing =>x->x.== spherePackBool) end
  if !isnothing(randOffset) df_subset = subset(df_subset, :rand_offset =>x->x.== randOffset) end
  if !isnothing(grpFlag) df_subset = subset(df_subset, :grp_flag =>x->x.== grpFlag) end
  if !isnothing(trainSize) df_subset = subset(df_subset, :train_size =>x->x.== trainSize) end

  @info "$dbname: pull out and print best values"

  #function bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing,
  #    trainSize, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)

  if mdlBase == "KNNRegressor"
#    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K, :train_size, 
#                                       :sphere_packing, :rand_offset, :grp_flag]]
    @show first(sort!(df_subset, :mee),3)[:, [:mee, :σee, :mee_σ, :p, :K, :train_size, 
                                              :sphere_packing, :rand_offset, :grp_flag]]
  else
#    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda, :train_size, 
#                                       :sphere_packing, :rand_offset, :grp_flag]]
    @show first(sort!(df_subset, :mee),3)[:,[:mee, :σee, :mee_σ, :p, :lambda, :train_size, 
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
function plotdensitycurve(matfile, mdlType, mdlBase, plt, legendposition, 
        label_dict; passedlabel=nothing)
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
  #plt_title = get(label_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  #mdlTypeLeg = get(label_dict, mdlType, mdlType)
  #mdlBaseLeg = get(label_dict, mdlBase, mdlBase)

  #now plot
  @info trainVec, meanDisVec
  Plots.plot!(plt, trainVec, meanDisVec, ls=:solid, lc=:black, shape=:none, xscale=:log10, 
              label="Avg Sample Spacing", legend_position=legendposition)

end #plotdensitycurve function

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

############################################################################
#function to generate histogram of TDoA dataset to use
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


############################################################################
"""
    calcIdealTDOA(sampleLocs::Array, sensorLocs::Array; tol::Real=0.5)

Calculate the ideal TDoA values given sensor and sample locations in 2D
Cartesian coordinates. Also pass a tolerance value.  It removes values
for when sensor and sample location is less than that value -- due to 
antenna null pattern.

 sampleLocs - 2D array of Cartesian coordinates (sample locations)
 sensorLocs - 2D array of Cartesian coordinates (sensors)
 tol=0.5    - if distance between sensor and sample locs is less, remove

"""
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


############################################################################
"""
    calcIdealRSS(sampleLocs::Array, sensorLocs::Array, RSSvals::DataFrame; 
                ple::Real=2.4, dref::Real=1.0)

Calculate the ideal RSS values given sensor and sample locations in 2D
Cartesian coordinates. Also pass a tolerance value.  It removes values
for when sensor and sample location is less than that value -- due to 
antenna null pattern. Path loss estimate can be passed as well

 sampleLocs - 2D array of Cartesian coordinates (sample locations)
 sensorLocs - 2D array of Cartesian coordinates (sensors)
 RSSvals    - DataFrame of sensor-based RSS and distances
 ple=2.4    - path loss estimate (in dB)
 dref=1.0   - use this distance and less for reference power

 Returns array [original RSS; distances; ideal RSS]

"""
function calcIdealRSS(sampleLocs::Array, sensorLocs::Array, RSSvals::DataFrame; 
                        ple::Real=2.4, dref::Real=1.0)

    #get p0 average value to set as reference value
    #get col dist names
    coldistnames = filter(x->occursin(r"dist",x), names(RSSvals))
    #split out the sensor names (A0,...,G1)
    sensornames = [split(idx,"_")[1] for idx in coldistnames];
    #stack dist value and RSS values for all sensors
    rssdistvals = nothing
    for idx in sensornames
        colstr = filter(x->occursin(Regex(idx),x),names(RSSvals))
        isnothing(rssdistvals) ? rssdistvals=Matrix(RSSvals[:, colstr]) : rssdistvals = vcat(rssdistvals, Matrix(RSSvals[:, colstr]))
    end
    #calc pref
    pref = mean(rssdistvals[rssdistvals[:,2] .< dref, :][:,1])
    #calc ideal power
    RSSideal = pref .- 10.0 .* ple .* log10.(rssdistvals[:,2]./dref)
    
    return hcat(rssdistvals, RSSideal)
end #end of calcIdealRSS function


############################################################################
#function to generate kernel matrix and sort "distances" between measurements
"""

    kerneldistancesort(sampleMeas::Array, p::Float64; locIdx::Integer=nothing)

Passes in feature matrix and generates a L_p kernel matrix.  Returns an index set
that corresponds to the sorted L_p distances for a random row.  To choose which
row, pass in locIdx value.  As a convenience, returns used locIdx. 

Returns (sorted index, locIdx)

"""
function kerneldistancesort(sampleMeas::Array, p::Float64; locIdx::Integer=-1)


    #set up distance calculation using kernel matrix
    dKt = KernelFunctions.compose(PnormKernelStd(p), KernelFunctions.ScaleTransform(1))
    #get distance  kernel matrix
    dKKt=KernelFunctions.kernelmatrix(dKt, sampleMeas, obsdim=1);
    #coulde use datakernelization function...

    #randomly pick location (also row of kernel matrix) unless passed value
    locIdx<0 ? idx = rand(collect(1:size(sampleMeas,1))) : idx = locIdx
    distVec = dKKt[idx, :]
    
    #sort distances
    sortDistidx = sortperm(distVec)

    return (sortDistidx, idx)

end#function kerneldistancesort    

############################################################################
############################################################################
end #end module




