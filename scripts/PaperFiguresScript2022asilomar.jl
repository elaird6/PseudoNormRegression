using DrWatson

@quickactivate "PseudoNormRegression"


#using Random, Distributions, CSV
using CSV
using Plots
gr()
using DataFrames, Latexify
using QuasinormRegression
using Zenodo3968503Functions

include(scriptsdir("AnalysisFunctionsModule.jl"))
using .AnalysisFunctionsModule

#################################################################################
#dictionary to replace names to prettify plots and such
label_dict = Dict("JHUtdoa"=>"JHU TDoA", "JHUrss"=>"JHU RSS", "JHUaoaRAW"=>"JHU AoA", 
                   "TUD" => "TUD TDoA", "DSI1"=>"DSI1 RSS", "DSI2"=>"DSI2 RSS", 
                   "Standard"=>"Std", "SingleKernel"=>"SK", "MultiKernel"=>"MK",
                   "KNNRegressor"=>"KNN", "RidgeRegressor"=>"Ridge", "LassoRegressor"=>"LASSO"
                  )
@show label_dict


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
print("Do you want to save TDoA outliers figure? (y/N) ");
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_sensorimposedoutlier.pdf") : println("no save")


##############################################################################
#plot mean, variance of best std/sk ridge/lasso/k-nn performance for each dataset
for dbname in ("JHUtdoa.mat", "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat")
  plt_test = Plots.plot()
  plotdatachoice(dbname, "Standard", "KNNRegressor", true, plt_test, :topleft, label_dict; passedlabel=nothing)
  plotdatachoice(dbname, "SingleKernel", "KNNRegressor", true, plt_test, :topleft, label_dict; passedlabel=nothing)
  try
    plotdatachoice(dbname, "SingleKernel", "RidgeRegressor", true, plt_test, :topleft, label_dict; passedlabel=nothing)
  catch 
    @warn "No RidgeRegressor for $dbname ..." 
  end
  try
    plotdatachoice(dbname, "SingleKernel", "LassoRegressor", true, plt_test, :topleft, label_dict; passedlabel=nothing)
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
#               plt_Elast, :topleft, label_dict; passedlabel=nothing)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor", false, 
#               plt_Elast, :topleft, label_dict; passedlabel=nothing)
plotdatachoice("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "Standard", "KNNRegressor", false, 
               plt_Elast, :topleft, label_dict; passedlabel=nothing)
plotdatachoice("JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat", "SingleKernel", "KNNRegressor", false, 
               plt_Elast, :topleft, label_dict; passedlabel=nothing)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.0001.mat", "Standard", "KNNRegressor", false, 
#               plt_Elast, :topleft, label_dict; passedlabel=nothing)
#plotdatachoice("ELnetJHU01_γ=1.0_λ=0.0001.mat", "SingleKernel", "KNNRegressor", false, 
#               plt_Elast, :topleft, label_dict; passedlabel=nothing)
plotdatachoice("JHUtdoa.mat", "Standard", "KNNRegressor", false, plt_Elast, :topleft, label_dict; passedlabel=nothing)
plotdatachoice("JHUtdoa.mat", "SingleKernel", "KNNRegressor", false, plt_Elast, :topleft, label_dict; passedlabel=nothing)
display(plt_Elast)


##############################################################################
#plotting performance versus trainsize for given parameters
#
#randOffsetVal=1.5; Kval=7; grpFlagBool=true; varPlot=false
randOffsetVal=0.7; Kval=7; grpFlagBool=true; varPlot=false; sphereBool=true;

p=2.0
filename= "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat"
plt_ElastTrain = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=false, 
                            title="ElastPreprocess, Train Size")
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
filename= "JHUtdoa.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdensitycurve(filename, "SingleKernel", "KNNRegressor", plt_ElastTrain, :topright, 
                 label_dict; passedlabel=nothing)
plt_ElastTrain.series_list[1][:label] = "Std:KNN("*L"L_2"*"), k=$Kval, denoise"
#plt_ElastTrain.series_list[2][:label] = "K:KNN($Kval),denoise"
plt_ElastTrain.series_list[2][:label] = "SK:Ridge, p=$p, denoise"
plt_ElastTrain.series_list[3][:label] = "Std:KNN("*L"L_2"*"), k=$Kval"
#plt_ElastTrain.series_list[5][:label] = "K:KNN($Kval)"
plt_ElastTrain.series_list[4][:label] = "SK:Ridge, p=$p"
plt_ElastTrain.series_list[5][:label] = "Avg Sample Spacing"
title!(plt_ElastTrain, " ")
display(plt_ElastTrain)
print("Do you want to save the last sphere packing curve figure (L2: K=$Kval, spherePacking=$sphereBool, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_L2DenoiseSpherePackingK=$(Kval)spherePacking=$(sphereBool)RandOff=$(randOffsetVal).pdf") : println("no save")



###########################################
#use p=1 rather than setting to 2
p=1.05
plt_ElastTrain1 = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=false, 
                             title="DenoisePreprocess, Train Size, p=$p")
filename= "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
filename= "JHUtdoa.mat"
plotdatachoice(filename, "Standard", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
#plotdatachoice(filename, "SingleKernel", "KNNRegressor", p, Kval, sphereBool, randOffsetVal, 
#               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "RidgeRegressor", p, Kval, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain1, :topright, label_dict; passedlabel=nothing)
plotdensitycurve(filename,  "SingleKernel", "KNNRegressor", plt_ElastTrain1, :topright, 
                 label_dict; passedlabel=nothing)
display(plt_ElastTrain1)
print("Do you want to save the last sphere packing curve figure (L1: K=$Kval, spherePacking=$sphereBool, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_L1DenoiseSpherePackingK=$(Kval)spherePacking=$(sphereBool)RandOff=$(randOffsetVal).pdf") : println("no save")


####################################################################
#pick optimal p rather than setting to 2
pVal=nothing; randOffsetVal=0.7; Kval=nothing; grpFlagBool=true; varPlot=false; sphereBool=true;

filename= "JHUtdoaDenoiseELN_γ=1.0_λ=0.001.mat"

df_results=bestresults(filename, "Standard",     "KNNRegressor";   p=pVal, K=Kval, trainSize=trainVal, 
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
STDknn=df_results[1, [:p, :K]]

df_results=bestresults(filename, "SingleKernel", "KNNRegressor"; p=pVal, K=Kval, trainSize=trainVal, 
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
SKridge=df_results[1, [:p, :K]]


plt_ElastTrain = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=false, 
                            title="ElastPreprocess, Train Size")

plotdatachoice(filename, "Standard", "KNNRegressor", STDknn.p, STDknn.K, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "KNNRegressor", SKridge.p, SKridge.K, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)



filename= "JHUtdoa.mat"

df_results=bestresults(filename, "Standard", "KNNRegressor";   p=pVal, K=Kval, trainSize=trainVal, 
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
STDknn2=df_results[1, [:p, :K]]

df_results=bestresults(filename, "SingleKernel", "KNNRegressor"; p=pVal, K=Kval, trainSize=trainVal, 
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
SKridge2=df_results[1, [:p, :K]]

plotdatachoice(filename, "Standard", "KNNRegressor", STDknn2.p, STDknn2.K, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "KNNRegressor", SKridge2.p, SKridge2.K, sphereBool, randOffsetVal, 
               grpFlagBool, varPlot, plt_ElastTrain, :topright, label_dict; passedlabel=nothing)
plotdensitycurve(filename, "SingleKernel", "KNNRegressor", plt_ElastTrain, :topright, 
                 label_dict; passedlabel=nothing)
plt_ElastTrain.series_list[1][:label] = "Denoise: Std:kNN("*L"L_p"*"), k=$(STDknn.K), p=$(STDknn.p)"
plt_ElastTrain.series_list[2][:label] = "Denoise: SK:kNN("*L"L_2"*"), k=$(SKridge.K), p=$(SKridge.p)"
#plt_ElastTrain.series_list[2][:label] = "Denoise: SK:Ridge, p=$(SKridge.p)"
plt_ElastTrain.series_list[3][:label] = "Std:kNN("*L"L_p"*"), k=$(STDknn2.K), p=$(STDknn2.p)"
plt_ElastTrain.series_list[4][:label] = "SK:kNN("*L"L_2"*"), k=$(SKridge2.K), p=$(SKridge2.p)"
#plt_ElastTrain.series_list[4][:label] = "SK:Ridge, p=$(SKridge.p)"
plt_ElastTrain.series_list[5][:label] = "Avg Sample Spacing"
title!(plt_ElastTrain, " ")
display(plt_ElastTrain)
print("Do you want to save the last sphere packing curve figure (L2: K=$Kval, spherePacking=$sphereBool, randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_DenoiseSpherePackingOptimalP.pdf") : println("no save")






##############################################################################
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



##############################################################################
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



##############################################################################
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
