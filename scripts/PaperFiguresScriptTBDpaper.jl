using DrWatson

@quickactivate "PseudoNormRegression"

using Random, Distributions
using Plots, Logging, Printf
using KernelDensity
pyplot();
#gr()
using LinearAlgebra: norm
using DataFrames, Latexify
using QuasinormRegression
using Zenodo3968503Functions

###############################
#plot of p-norm contour lines
##############################
x_pts = collect(-1:0.01:1)
p_val=0.5
plt_pnorms = Plots.plot(legend=:outerright, grid=false, title="p-norm unit contours", reuse=false)

#plots unit norm contour line
for p_val in [0.3 0.5 1.0 1.5 2]

    y_pts = (1.0.^(p_val) .- abs.(x_pts).^(p_val)).^(1/p_val)
    Plots.plot!(plt_pnorms, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal,
        label="p="*string(p_val),  framestyle=:origin, legend_position=:topleft)

end

#declaring a type recipe to display KDE contour of data
#simplified version of marginalKDE in StatsPlot
mutable struct contourKDE end
################################################
#plot p-norm contour lines on top of kernel
#density lines
###############################################
@recipe function f(kc::contourKDE, x::AbstractArray, y::AbstractArray; levels=5)
    x = vec(x)
    y = vec(y)
    
    k = KernelDensity.kde((x, y))
    kx = KernelDensity.kde(x)
    ky = KernelDensity.kde(y)

    ps = [pdf(k, xx, yy) for (xx, yy) in zip(x,y)]

    ls = []
    for p in range(1.0/levels, stop=1-1.0/levels, length=levels-1)
    #for p in 10.0.^(range(-levels, stop=0, length=levels))
        push!(ls, quantile(ps, p))
    end

    seriestype := :contour
    levels := ls
    fill := false
    colorbar := false
    label --> nothing

    (collect(k.x), collect(k.y), k.density')
end

##############################################################################
#section plots out p=1,2 contour lines with density contour lines of gaussian
#and laplacian distributions
#
##############################################################################

#plot p-norm contours first
p_circ=2.0
plots_circle = Plots.plot(title=string(Int(p_circ))*"-norm contours", grid=false, legend=:outerright)

p_spar=1.0
plots_sparse = Plots.plot(title=string(Int(p_spar))*"-norm contours", grid=false, legend=:outerright)

for radius in 1:1:3
    local x_pts = collect(-radius:0.01:radius)
    linestyle=:dash

    y_pts = (radius.^(p_circ) .- abs.(x_pts).^(p_circ)).^(1/p_circ)
    Plots.plot!(plots_circle, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal, 
                label=nothing,  framestyle=:origin, lc=:green, ls=linestyle)
    
    y_pts = (radius.^(p_spar) .- abs.(x_pts).^(p_spar)).^(1/p_spar)
    Plots.plot!(plots_sparse, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal, 
                label=nothing,  framestyle=:origin, lc=:green, ls=linestyle)
end
xlims!(-Inf,Inf); ylims!(-Inf,Inf)
limstuple = (-3.25,3.25)
#title!("p-norm unit ball plot")

#now set up gaussian or laplacian distribution (also have impulsive, but not
#used 
ratio=1.25; bratio=0.1; num_pts = 2000

#plot circle, gaussian points 
awgn_pts = randn(Float16, (num_pts,2)).*ratio
Plots.plot!(plots_circle, contourKDE(), awgn_pts[:,1], awgn_pts[:,2],  xlims=limstuple, ylims=limstuple, label="awgn data")

#plot sparse, laplacian points
lapl_pts = rand(Laplace(0, 0.75), (num_pts,2))
#either using distribution or impulsive noise
color_pts = awgn_pts .* rand(Bernoulli(bratio), (num_pts,2)) .+ awgn_pts/5
color_pts = lapl_pts
Plots.plot!(plots_sparse, contourKDE(), color_pts[:,1], color_pts[:,2], xlims=limstuple, ylims=limstuple, label="laplacian data")

plt_combine = Plots.plot(plt_pnorms, plots_circle, plots_sparse, wsize=(1600,600), layout=(1,3), reuse=false)
display(plt_combine)
print("Do you want to save the combined contours figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormAllContours.png") : println("no save") 

#################################################################################
#plot data specific figures used in paper
#need data/results
#################################################################################
dbname_dict = Dict("JHUtdoa"=>"JHU TDoA", "JHUrss"=>"JHU RSS", "JHUaoa"=>"JHU AoA", "TUD" => "TUD TDoA", 
                   "DSI1"=>"DSI1 RSS", "DSI2"=>"DSI2 RSS")
@show dbname_dict
mdlType_dict = Dict("Standard"=>"Std", "SingleKernel"=>"SK", "MultiKernel"=>"MK")
@show mdlType_dict
mdlBase_dict = Dict("KNNRegressor"=>"KNN", "RidgeRegressor"=>"Ridge", "LassoRegressor"=>"LASSO",
                   "efficientnet-bo" => "DNN")
@show mdlBase_dict

#change backend used for plots (gr better for ribbon plots)
#pyplot()
gr()

#simple functions to pull out parameters from structs saved
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
function plotdatachoice(matfile::String, mdlType::String, mdlBase::String, varBool::Bool, plt, legendposition::Symbol)
#  dbname = split(matfile,".")[1]                  
  dbname = chop(matfile, head=0, tail=4)
  @info "$dbname: collecting formatting result files generated for $mdlType:$mdlBase regression"
  @info "$dbname: turning off warning/info messages while loading..."
  Logging.disable_logging(Logging.Warn) #log level below debug=-1000 Logging.LogLevel(-2000) Logging.Warn
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), subfolders=true, update=false, verbose=false)
  @debug first(df, 6)
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
  transform!(df_subset, :DataParams => ByRow(getDataParamsDataRep) => :datarep)
  transform!(df_subset, :DataParams => ByRow(getDataParamsAvgMeas) => :avgmeas)
  @info "$dbname: pull out and print best values"
  #if type is standard, need to change metric to p
  if mdlBase == "KNNRegressor"
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K]]
    df_subset = groupby(sort(df_subset,[:p, :K]), [:K, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestKstd = df_subset[bestGrpKey[2]][1,:K]
    legendStr = " k=$(bestKstd)"
  elseif ((mdlBase == "efficientnet-b0") && (mdlType != "Standard"))
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :blocks]]
    df_subset = groupby(sort(df_subset,[:p, :blocks]), [:blocks, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestBlockstd = df_subset[bestGrpKey[2]][1,:blocks]
    legendStr = " "
  elseif any(mdlBase .== ("RidgeRegressor", "LassoRegressor"))
    size(df_subset,1) ≥ 3 ? numrows=3 : numrows=size(df_subset,1)
    @show sort!(df_subset, :mee)[1:numrows, [:mee, :σee, :mee_σ, :p, :lambda]]
    df_subset = groupby(sort(df_subset,[:p, :lambda]), [:lambda, :datarep, :avgmeas])
    @info "$dbname: get key for best group"
    bestGrpKey = findmin(combine(df_subset, :mee => minimum)[:,:mee_minimum])
    bestλstd = @sprintf("%1.0e", df_subset[bestGrpKey[2]][1,:lambda])
    legendStr = " k=$(bestλstd)"
  else
    @warn "passed method and base model combination this isn't supported: $mdlType:$mdlBase"
  end
  
  #bounce dbname against dict, if maps then get modified value else use dbname as is
  plt_title = get(dbname_dict, dbname, dbname)
  #bounce mdlType and mdlBase against dict, if maps then get modified value pretty plot labels
  mdlTypeLeg = get(mdlType_dict, mdlType, mdlType)
  mdlBaseLeg = get(mdlBase_dict, mdlBase, mdlBase)
  #get best set of curves and plot
  sdf = combine(groupby(df_subset[bestGrpKey[2]], :p), :mee => mean => :mee, :mee => std => :σ, :σee => mean => :σee)
  if varBool
    Plots.plot!(plt, sdf.p, sdf.mee, ribbon=(min.(sdf.mee,sdf.σee), sdf.σee), ls=:auto, shape=:auto, 
                label="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(legendStr), 
                legend_position=legendposition)
  else
    Plots.plot!(plt, sdf.p, sdf.mee, ls=:auto, shape=:auto, 
                label="$mdlTypeLeg:$mdlBaseLeg "*plt_title*(legendStr), 
                legend_position=legendposition)
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
  df = collect_results(datadir("exp_pro","results","$matfile","$mdlType"), subfolders=true, update=false, verbose=false)
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
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :K, :train_size, :sphere_packing, :rand_offset, :grp_flag]]
  else
    @show sort!(df_subset, :mee)[1:3, [:mee, :σee, :mee_σ, :p, :lambda, train_size, :sphere_packing, :rand_offset, :grp_flag]]
  end


  #back to normal logging messages
  @info "turned standard logging messages back on"
  Logging.disable_logging(Logging.Debug)

  return df_subset
end #bestresults function


#plot mean, variance of best std/sk ridge/lasso/k-nn performance for each dataset
for dbname in ("JHUtdoa.mat", "TUD.mat", "UTIf2.mat", "JHUrss.mat", "DSI1.mat", "LIB1.mat")
  plt_test = Plots.plot()
  plotdatachoice(dbname, "Standard", "KNNRegressor", true, plt_test, :topleft)
  plotdatachoice(dbname, "SingleKernel", "KNNRegressor", true, plt_test, :topleft)
  plotdatachoice(dbname, "SingleKernel", "efficientnet-b0", true, plt_test, :topleft)
#  plotdatachoice(dbname, "SingleKernel", "RidgeRegressor", true, plt_test, :topleft) #uncomment
#  plotdatachoice(dbname, "SingleKernel", "LassoRegressor", true, plt_test, :topleft) #uncomment
  display(plt_test)
  print("Do you want to save the $dbname figure? (y/N) "); 
  #empty string is "identity" so check if empty and if is, set to default
  local getinput = lowercase(readline())
  contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_$(chop(dbname, tail=4))Pnorms.pdf") : println("no save") 
end
pyplot()

#plot heavy tail empirical data 
plt_tdoa =Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, size=(800,600), 
                     title="Empirical - Heavy Tail Distributions")
plt_tdoa2 =Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, size=(800,600)) 
plt_tdoa3 =Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, size=(800,600), ylims=(0.48,1.00)) 

#iters = collect(Iterators.product(("KNNRegressor", "RidgeRegressor","LassoRegressor"),("Standard","SingleKernel")))
iters = collect(Iterators.product(["KNNRegressor"],["Standard","SingleKernel"]))
for (idx,(mdlBase, mdlType)) in enumerate(iters)
  if idx ==1 || idx > length(iters)/2
    plotdatachoice("JHUtdoa.mat", mdlType, mdlBase, false, plt_tdoa, :topleft)
    plotdatachoice("JHUaoa.mat", mdlType, mdlBase, false, plt_tdoa, :topright) 
    plotdatachoice("TUD.mat", mdlType, mdlBase, false, plt_tdoa2, :topleft)   #uncomment
    plotdatachoice("UTIf2.mat", mdlType, mdlBase, false, plt_tdoa3, :topleft)  #uncomment?
    #plotdatamean("UTIf3.mat", :bottomright, plt_tdoa3)
  end
end
l = @layout [a{0.5h};b;c]
plt_heavy =Plots.plot(plt_tdoa, plt_tdoa2, plt_tdoa3, layout=l)#, size=(100,500))
display(plt_heavy)
print("Do you want to save the empirical heavy tail figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormEmpHeavyTail.pdf") : println("no save") 

#plot guassian empirical data
plt_rss =Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, size=(800,600),
                    title="Empirical - Gaussian Distributions")
for (mdlBase, dbname) in Iterators.product(["KNNRegressor"],["JHUrss.mat", "DSI1.mat", "LIB1.mat"])
  if (mdlBase == "KNNRegressor") plotdatachoice(dbname, "Standard", mdlBase, false, plt_rss, :topright); end  #uncomment
  plotdatachoice(dbname, "SingleKernel", mdlBase, false, plt_rss, :topright)
end
display(plt_rss)
print("Do you want to save the empirical gaussian figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormEmpGaussian.pdf") : println("no save") 

#plot simulation data
plt_lapl = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title="Laplacian")
plt_logi = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title="Logistic")
plt_gauss= Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title="Gaussian")
plt_none = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false)#,, title="no errors simulation")
plotdatachoice("SimLaplaceTDoA_θ=62.mat", "Standard", "KNNRegressor", false, plt_lapl, :top)
plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "KNNRegressor", false, plt_lapl, :top)
#plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "LassoRegressor", false, plt_lapl, :top) #testing, keep or remove?
#plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "RidgeRegressor", false, plt_lapl, :top) #testing, keep or remove?

plotdatachoice("SimLogisticTDoA_θ=44.mat", "Standard", "KNNRegressor", false, plt_logi, :top)
plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "KNNRegressor", false, plt_logi, :top)
#plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "LassoRegressor", false, plt_logi, :top)#testing, keep or remove?
#plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "RidgeRegressor", false, plt_logi, :top) #testing, keep or remove?

plotdatachoice("SimNormalTDoA_θ=75.mat", "Standard", "KNNRegressor", false, plt_gauss, :top)
plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "KNNRegressor", false, plt_gauss, :top)
#plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "LassoRegressor", false, plt_gauss, :top)#testing, keep or remove?
#plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "RidgeRegressor", false, plt_gauss, :top) #testing, keep or remove?

plotdatachoice("SimNormalTDoA_θ=0.mat", "Standard", "KNNRegressor", false, plt_none, :right)
plotdatachoice("SimNormalTDoA_θ=0.mat", "SingleKernel", "KNNRegressor", false, plt_none, :right)
#plt_pnorm_sim = Plots.plot(plt_lapl, plt_logi, plt_gauss, layout=(1,3), ylims=(5.0,6.75), size=(1500,500))
plt_pnorm_sim = Plots.plot(plt_lapl, plt_logi, plt_gauss, layout=(1,3), size=(1500,500))
display(plt_pnorm_sim)

print("Do you want to save the simulation performance figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormSim.pdf") : println("no save") 

#using tukeylambda to modify the heaviness of tail
plt_tukey = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title=nothing)
plotdatachoice("SimTukeyLambda_m=0.14_σ=25.mat", "Standard", "KNNRegressor", false, plt_tukey, :topright)
plotdatachoice("SimTukeyLambda_m=0.0001_σ=25.mat", "Standard", "KNNRegressor", false, plt_tukey, :topright) 
plotdatachoice("SimTukeyLambda_m=-0.14_σ=25.mat", "Standard", "KNNRegressor", false, plt_tukey, :topright)
plotdatachoice("SimTukeyLambda_m=-0.3_σ=25.mat", "Standard", "KNNRegressor", false, plt_tukey, :topright)
display(plt_tukey)
print("Do you want to save the Tukey Lambda simulation performance figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_TukeyLambdaSimSTDknn.pdf") : println("no save") 


plt_tukey2 = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title=nothing)
plotdatachoice("SimTukeyLambda_m=0.14_σ=25.mat", "SingleKernel", "KNNRegressor", false, plt_tukey2, :topright)
plotdatachoice("SimTukeyLambda_m=0.0001_σ=25.mat", "SingleKernel", "KNNRegressor", false, plt_tukey2, :topright) 
plotdatachoice("SimTukeyLambda_m=-0.14_σ=25.mat", "SingleKernel", "KNNRegressor", false, plt_tukey2, :topright)
plotdatachoice("SimTukeyLambda_m=-0.3_σ=25.mat", "SingleKernel", "KNNRegressor", false, plt_tukey2, :topright)
display(plt_tukey2)
print("Do you want to save the Tukey Lambda simulation (normalized) performance figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_TukeyLambdaSimSKknn.pdf") : println("no save") 


#plotting preprocessing... not part of Journal?
#plt_Elast = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title="ElastPreprocess")
#plotdatachoice("ELnetNormJHUtdoa_γ=1.0_λ=0.001.mat", "Standard", "KNNRegressor", false, plt_Elast, :topright)
#plotdatachoice("ELnetNormJHUtdoa_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor", false, plt_Elast, :topright)
#plotdatachoice("JHUtdoa.mat", "Standard", "KNNRegressor", false, plt_Elast, :topright)
#plotdatachoice("JHUtdoa.mat", "SingleKernel", "KNNRegressor", false, plt_Elast, :topright)
#display(plt_Elast)

println("bestresults(matfile, mdlType, mdlBase; p=nothing, K=nothing, trainSize=nothing, spherePackBool=nothing, randOffset=nothing, grpFlag=nothing)")
#tuple of tuples to get table
param_tuple = [("JHUaoa.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHUaoa.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHUtdoa.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHUtdoa.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("TUD.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("TUD.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("UTIf2.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("UTIf2.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("JHUrss.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHUrss.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("DSI1.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("DSI1.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("LIB1.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("LIB1.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("UJI1.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("UJI1.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("TUT1.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("TUT1.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),

               ("SimTukeyLambda_m=0.14_σ=25.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=0.14_σ=25.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=0.0001_σ=25.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=0.0001_σ=25.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=-0.14_σ=25.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=-0.14_σ=25.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=-0.3_σ=25.mat", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=-0.3_σ=25.mat", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing),

               ("JHU_fingerprints.csv", "Standard", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHU_fingerprints.csv", "SingleKernel", "KNNRegressor", nothing, nothing, nothing, nothing, nothing, nothing)
              ]

dfbest = DataFrames.DataFrame()
cols = (:db, :method,:mdl_type, :mee, :σee, :p, :K)
for idx in param_tuple
    dfbesttmp = bestresults(idx[1:3]...; 
                            p=idx[4], K=idx[5], trainSize=idx[6], spherePackBool=idx[7], randOffset=idx[8], grpFlag=idx[9]);
    append!(dfbest, first(dfbesttmp[:,[cols...]],1))
end

@info "latexify(df, env = :tabular, fmt = \"%0.2f\")"
@info "using these cols: $cols"
#remove .mat from db filename, breaks latexify which attempts to parse as
#latex equation
transform!(dfbest, :db => ByRow(x->chop(x,tail=4))=>:db)
@show dfbest;
println("this is raw latex table of results shown just above used in paper prior to cleaning up names and adding in table bells and whistles")
latexify(dfbest, env = :tabular, fmt = "%0.2f")

