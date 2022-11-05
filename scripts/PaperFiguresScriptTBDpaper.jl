using DrWatson

@quickactivate "PseudoNormRegression"

using Random, Distributions
using Plots, Logging, Printf
using KernelDensity
pyplot();
#gr()
using LinearAlgebra: norm
using DataFrames, Latexify, LaTeXStrings
using QuasinormRegression
using Zenodo3968503Functions

includet(scriptsdir("AnalysisFunctionsModule.jl"))
using .AnalysisFunctionsModule

#################################################################################
#dictionary to replace names to prettify plots and such
label_dict = Dict("JHUtdoa"=>"JHU TDoA", "JHUrss"=>"JHU RSS", "JHUaoaRAW"=>"JHU AoA",
                   "TUD" => "TUD TDoA", "DSI1"=>"DSI1 RSS", "DSI2"=>"DSI2 RSS",
                   "Standard"=>"Std", "SingleKernel"=>"SK", "MultiKernel"=>"MK",
                   "KNNRegressor"=>"KNN", "RidgeRegressor"=>"Ridge", "LassoRegressor"=>"LASSO"
                  )
#@show label_dict

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
plots_circle = Plots.plot(title=string(Int(p_circ))*"-norm contours", 
                          grid=false, legend=:outerright)

p_spar=1.0
plots_sparse = Plots.plot(title=string(Int(p_spar))*"-norm contours", 
                          grid=false, legend=:outerright)

p_bern = 0.5
plots_bern = Plots.plot(title=string(Float16(p_bern))*"-norm contours", 
                        grid=false, legend=:outerright)

for radius in 1:1:3
    local x_pts = collect(-radius:0.01:radius)
    linestyle=:dash

    y_pts = (radius.^(p_circ) .- abs.(x_pts).^(p_circ)).^(1/p_circ)
    Plots.plot!(plots_circle, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal, 
                label=nothing,  framestyle=:origin, lc=:green, ls=linestyle)
    
    y_pts = (radius.^(p_spar) .- abs.(x_pts).^(p_spar)).^(1/p_spar)
    Plots.plot!(plots_sparse, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal, 
                label=nothing,  framestyle=:origin, lc=:green, ls=linestyle)

    y_pts = (radius.^(p_bern) .- abs.(x_pts).^(p_bern)).^(1/p_bern)
    Plots.plot!(plots_bern, vcat(x_pts, x_pts), vcat(y_pts, -y_pts),  aspect_ratio = :equal, 
                label=nothing,  framestyle=:origin, lc=:green, ls=linestyle)

end
xlims!(-Inf,Inf); ylims!(-Inf,Inf)
limstuple = (-3.25,3.25)
#title!("p-norm unit ball plot")

#now set up gaussian or laplacian distribution (also have impulsive, but not used) 
ratio=1.25; bratio=0.2; num_pts = 2000

#plot circle, gaussian points 
awgn_pts = randn(Float16, (num_pts,2)).*ratio
Plots.plot!(plots_circle, contourKDE(), awgn_pts[:,1], awgn_pts[:,2],  xlims=limstuple, 
            ylims=limstuple, label="awgn data")

#plot sparse, laplacian points
lapl_pts = rand(Laplace(0, 0.75), (num_pts,2))
#either using distribution or impulsive noise
#color_pts = 3.0.*awgn_pts .* rand(Bernoulli(bratio), (num_pts,2)) .+ 3.0.*awgn_pts/5
color_pts = lapl_pts
#color_pts = lapl_pts
Plots.plot!(plots_sparse, contourKDE(), color_pts[:,1], color_pts[:,2], xlims=limstuple, 
            ylims=limstuple, label="laplacian data")

#plot mixed gaussian/bernoulli... either using distribution or impulsive noise
bratio=0.10
#color_pts = awgn_pts./bratio .* rand(Bernoulli(bratio), (num_pts,2)) .+ awgn_pts.*bratio
color_pts = lapl_pts./1e-1 .* rand(Bernoulli(bratio), (num_pts,2)) .+ lapl_pts.*1e-1
#color_pts = lapl_pts
Plots.plot!(plots_bern, contourKDE(), color_pts[:,1], color_pts[:,2], xlims=limstuple, 
            ylims=limstuple, label="laplacian data", levels=10)

plt_combine = Plots.plot(plt_pnorms, plots_circle, plots_sparse, plots_bern,
                         wsize=(1600,600), layout=(1,4), reuse=false)
display(plt_combine)
print("Do you want to save the combined contours figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormAllContours.png") : println("no save") 

##############################################################################
#get best values for these models
filename= "JHUtdoa.mat"
randOffsetVal=0.7; Kval=nothing; grpFlagBool=true; varPlot=false; sphereBool=true;
trainVal=nothing; pVal=nothing

df_results=bestresults(filename, "Standard",     "KNNRegressor";   p=pVal, K=Kval, trainSize=trainVal,
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
STDknn=df_results[1, [:p, :K]]

df_results=bestresults(filename, "SingleKernel", "KNNRegressor";   p=pVal, K=Kval, trainSize=trainVal,
            spherePackBool=sphereBool, randOffset=randOffsetVal, grpFlag=grpFlagBool)
SKknn=df_results[1, [:p, :K]]

##############################################################################
#plotting performance versus trainsize for given parameters
#
#randOffsetVal=1.5; Kval=7; grpFlagBool=true; varPlot=false

plt_trainsize = Plots.plot(xlabel="train size", ylabel="MSE(m)", reuse=false, grid=true,
                          title="")

plotdatachoice(filename, "Standard", "KNNRegressor", STDknn.p, STDknn.K, sphereBool, randOffsetVal,
               grpFlagBool, varPlot, plt_trainsize, :topright, label_dict; passedlabel=nothing)
plotdatachoice(filename, "SingleKernel", "KNNRegressor", SKknn.p, SKknn.K, sphereBool, randOffsetVal,
               grpFlagBool, varPlot, plt_trainsize, :topright, label_dict; passedlabel=nothing)
plotdensitycurve(filename, "SingleKernel", "KNNRegressor", plt_trainsize, :topright,
                 label_dict; passedlabel=nothing)

plt_trainsize.series_list[1][:label] = "Std:kNN("*L"L_p"*"), k=$(STDknn.K), p=$(STDknn.p)"
plt_trainsize.series_list[2][:label] = "SK:kNN("*L"L_2"*"), k=$(SKknn.K), p=$(SKknn.p)"
plt_trainsize.series_list[3][:label] = "Avg Sample Spacing"
#    title!(plt_trainsize, " ")
display(plt_trainsize)
#print("Do you want to save the last sphere packing curve figure (P=$p, K=$Kval, spherePacking=$sphereBool, 
print("Do you want to save the last sphere packing curve figure (spherePacking=$sphereBool, 
      randOff=$randOffsetVal, grpFlag=$grpFlagBool)? (y/N) ");
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig(plt_trainsize, "fig_PerformanceSpherePackingKNN.pdf") : println("no save")

break



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


#plot mean, variance of best std/sk ridge/lasso/k-nn performance for each dataset
#function plotdatachoice(matfile, mdlType, mdlBase, varBool, plt, legendposition, 
#        label_dict; passedlabel=nothing)


for dbname in ("JHUtdoa.mat", "TUD.mat", "UTIf2.mat", "JHUrss.mat", "DSI1.mat", "LIB1.mat")
  plt_test = Plots.plot()
  plotdatachoice(dbname, "Standard", "KNNRegressor", true, plt_test, :topleft, label_dict)
  plotdatachoice(dbname, "SingleKernel", "KNNRegressor", true, plt_test, :topleft, label_dict)
  plotdatachoice(dbname, "SingleKernel", "efficientnet-b0", true, plt_test, :topleft, label_dict)
#  plotdatachoice(dbname, "SingleKernel", "RidgeRegressor", true, plt_test, :topleft, label_dict) #uncomment
#  plotdatachoice(dbname, "SingleKernel", "LassoRegressor", true, plt_test, :topleft, label_dict) #uncomment
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
    plotdatachoice("JHUtdoa.mat", mdlType, mdlBase, false, plt_tdoa, :topleft, label_dict)
    plotdatachoice("JHUaoa.mat", mdlType, mdlBase, false, plt_tdoa, :topright, label_dict) 
    plotdatachoice("TUD.mat", mdlType, mdlBase, false, plt_tdoa2, :topleft, label_dict)   #uncomment
    plotdatachoice("UTIf2.mat", mdlType, mdlBase, false, plt_tdoa3, :topleft, label_dict)  #uncomment?
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
  if (mdlBase == "KNNRegressor") plotdatachoice(dbname, "Standard", mdlBase, false, plt_rss, :topright, label_dict); end  #uncomment
  plotdatachoice(dbname, "SingleKernel", mdlBase, false, plt_rss, :topright, label_dict)
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
plotdatachoice("SimLaplaceTDoA_θ=62.mat", "Standard", "KNNRegressor", false, plt_lapl, :top, label_dict)
plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "KNNRegressor", false, plt_lapl, :top, label_dict)
#plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "LassoRegressor", false, plt_lapl, :top, label_dict) #testing, keep or remove?
#plotdatachoice("SimLaplaceTDoA_θ=62.mat", "SingleKernel", "RidgeRegressor", false, plt_lapl, :top, label_dict) #testing, keep or remove?

plotdatachoice("SimLogisticTDoA_θ=44.mat", "Standard", "KNNRegressor", false, plt_logi, :top, label_dict)
plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "KNNRegressor", false, plt_logi, :top, label_dict)
#plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "LassoRegressor", false, plt_logi, :top, label_dict)#testing, keep or remove?
#plotdatachoice("SimLogisticTDoA_θ=44.mat", "SingleKernel", "RidgeRegressor", false, plt_logi, :top, label_dict) #testing, keep or remove?

plotdatachoice("SimNormalTDoA_θ=75.mat", "Standard", "KNNRegressor", false, plt_gauss, :top, label_dict)
plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "KNNRegressor", false, plt_gauss, :top, label_dict)
#plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "LassoRegressor", false, plt_gauss, :top, label_dict)#testing, keep or remove?
#plotdatachoice("SimNormalTDoA_θ=75.mat", "SingleKernel", "RidgeRegressor", false, plt_gauss, :top, label_dict) #testing, keep or remove?

plotdatachoice("SimNormalTDoA_θ=0.mat", "Standard", "KNNRegressor", false, plt_none, :right, label_dict)
plotdatachoice("SimNormalTDoA_θ=0.mat", "SingleKernel", "KNNRegressor", false, plt_none, :right, label_dict)
#plt_pnorm_sim = Plots.plot(plt_lapl, plt_logi, plt_gauss, layout=(1,3), ylims=(5.0,6.75), size=(1500,500))
plt_pnorm_sim = Plots.plot(plt_lapl, plt_logi, plt_gauss, layout=(1,3), size=(1500,500))
display(plt_pnorm_sim)

print("Do you want to save the simulation performance figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_PnormSim.pdf") : println("no save") 

#using tukeylambda to modify the heaviness of tail
plt_tukey = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title=nothing)
plotdatachoice("SimTukeyLambda_m=0.14_σ=25.mat", "Standard", "KNNRegressor", 
               false, plt_tukey, :topright, label_dict)
plotdatachoice("SimTukeyLambda_m=0.0001_σ=25.mat", "Standard", "KNNRegressor", 
               false, plt_tukey, :topright, label_dict) 
plotdatachoice("SimTukeyLambda_m=-0.14_σ=25.mat", "Standard", "KNNRegressor", 
               false, plt_tukey, :topright, label_dict)
plotdatachoice("SimTukeyLambda_m=-0.3_σ=25.mat", "Standard", "KNNRegressor", 
               false, plt_tukey, :topright, label_dict)
display(plt_tukey)
print("Do you want to save the Tukey Lambda simulation performance figure? (y/N) "); 
#empty string is "identity" so check if empty and if is, set to default
getinput = lowercase(readline())
contains("yes", (isempty(getinput) ? "N" : getinput)) ? Plots.savefig("fig_TukeyLambdaSimSTDknn.pdf") : println("no save") 


plt_tukey2 = Plots.plot(xlabel="p", ylabel="MSE(m)", reuse=false, grid=false, title=nothing)
plotdatachoice("SimTukeyLambda_m=0.14_σ=25.mat", "SingleKernel", "KNNRegressor", 
               false, plt_tukey2, :topright, label_dict)
plotdatachoice("SimTukeyLambda_m=0.0001_σ=25.mat", "SingleKernel", "KNNRegressor", 
               false, plt_tukey2, :topright, label_dict) 
plotdatachoice("SimTukeyLambda_m=-0.14_σ=25.mat", "SingleKernel", "KNNRegressor", 
               false, plt_tukey2, :topright, label_dict)
plotdatachoice("SimTukeyLambda_m=-0.3_σ=25.mat", "SingleKernel", "KNNRegressor", 
               false, plt_tukey2, :topright, label_dict)
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

               ("SimTukeyLambda_m=0.14_σ=25.mat", "Standard", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=0.14_σ=25.mat", "SingleKernel", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=0.0001_σ=25.mat", "Standard", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=0.0001_σ=25.mat", "SingleKernel", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=-0.14_σ=25.mat", "Standard", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=-0.14_σ=25.mat", "SingleKernel", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing),
               ("SimTukeyLambda_m=-0.3_σ=25.mat", "Standard", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing), 
               ("SimTukeyLambda_m=-0.3_σ=25.mat", "SingleKernel", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing),

               ("JHU_fingerprints.csv", "Standard", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing), 
               ("JHU_fingerprints.csv", "SingleKernel", "KNNRegressor", 
                nothing, nothing, nothing, nothing, nothing, nothing)
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

