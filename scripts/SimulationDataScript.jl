# Using provided sensor locations, generate TDoA values
# Intent is to test multivariate laplacian idea as basis
# or to vary kurtosis of distribution ala PearsonVII
# for pseudo-p-norm good performance
using DrWatson
@quickactivate "PseudoNormRegression"
using Distances, Distributions, MAT, MLJ, Random
using PyCall, SpecialFunctions
#using pycall to get a distribution not supported by julia
py"""
from scipy.stats import tukeylambda
"""

#how many simulation points
num_samples = 1000
train_percentage = 0.7

#using dictionary to enable various multiple trials to be run
#######################################################
#using Distributions.jl which has standard distributions
#######################################################
paramsNorm = Dict(
    "σ" => [0, 75], #scale factor, 0 perfect knowledge
    "μ" => 0,
    "d" => "Normal"
   )
paramsLap = Dict(
    "σ" => 62, #scale factor, 0 perfect knowledge
    "μ" => 0, 
    "d" => "Laplace"
   )
paramsLog = Dict(
    "σ" => 44, #scale factor, 0 perfect knowledge
    "μ" => 0, 
    "d" => "Logistic"
   )
#######################################################
#alternatively use Python's scipy.stats to generate tukeylambda
#shift from normal to logistic to heavier tail
#######################################################
paramsTuk = Dict(
    "m" => [0.14, 0.0001, -0.14, -0.30],
    "σ" => 25,
    "d" => "Tukeyλ"
   )

#combine all possible param combinations in a vector dictionary  list
param_combinations = append!(dict_list(paramsNorm), dict_list(paramsLap), 
                             dict_list(paramsLog), dict_list(paramsTuk))

#sensor locations
sensor_loc_x = [-1.06218975e+00, -1.03530849e+00,  2.43950167e+01,  2.43993900e+01,
         2.31431881e+01,  2.31297744e+01,  2.01587585e+01,  2.01716803e+01,
         2.50323597e+01,  2.50160214e+01, -7.30883698e-03, -2.79811639e-03,
         1.41513095e-02,  1.99385175e-02, -3.86488952e+00, -3.87211412e+00];
sensor_loc_y = [43.28882161, 43.30217052, 26.39448072, 26.3879422 , 46.75151122,
        46.73347571, 65.69210798, 65.6980855 ,  2.45848709,  2.44071491,
        25.64770495, 25.64835465,  0.62168871,  0.62204427, 61.35754919,
        61.3685754];
sensor_locs = vcat(sensor_loc_x', sensor_loc_y')

#loop over the possible combinations
for idx_params in param_combinations

  #ensure changing seed
  Random.seed!()

  #generate appropriate distribution from idx_params
  #first determine whether typical distribution or tukeyλ
  "Tukeyλ" == idx_params["d"] ? tukeyλ=true : tukeyλ=false
  #now set values to be used later
  if tukeyλ
      m = idx_params["m"]
      σ = idx_params["σ"]
  elseif idx_params["d"] == "Normal"
      d = Normal(idx_params["μ"], idx_params["σ"])
  elseif idx_params["d"] == "Laplace"
      d = Laplace(idx_params["μ"], idx_params["σ"])
  elseif idx_params["d"] == "Logistic"
      d = Logistic(idx_params["μ"], idx_params["σ"])
  else
      error("wrong or unsupportable distributions specified")
  end


  #generate random sample locations
  #sample_locs_x = [5.0 5.0 10.0; 10.0 20.0 20.0]
  sample_locs_x = rand(Float64, num_samples).*43.778 .-15.238
  sample_locs_y = rand(Float64, num_samples).*77.483 .-4.88
  sample_locs = vcat(sample_locs_x', sample_locs_y')

  #generate distances from sample_locs to sensor_locs,
  #change distance to nanoseconds (multiply by 3)
  euc_dist = pairwise(Euclidean(), sample_locs, sensor_locs).*3.0;

  #now get differential distance (tdoa)
  #set container for diff distance
  diff_dist = zeros(size(euc_dist,1), binomial(size(euc_dist,2),2))
  #iterate over samples to get diff distance values
  count=1
  for o_idx in 1:(size(euc_dist,2)-1)
    for i_idx in (o_idx+1):size(euc_dist,2)
      if tukeyλ
        #try to "normalize" via calculated variance?
        var = (2.0/(m^2)) * (1/(1+2*m) - ((gamma(m+1))^2)/(gamma(2*m+2)))
        diff_dist[:,count]=euc_dist[:,o_idx] .- euc_dist[:,i_idx] .+ py"tukeylambda.rvs($m, scale=$(σ/sqrt(var)), size=$num_samples)"
      else
        diff_dist[:,count]=euc_dist[:,o_idx] .- euc_dist[:,i_idx] .+ rand(d, num_samples)
      end
      count+=1
    end
  end

  #add in z, floor, bldg  to match the standard format used by Zenodo data
  sample_locs = vcat(sample_locs, zeros(3,num_samples))
  #partition data
  train_idx, test_idx  = partition(eachindex(diff_dist[:,1]), train_percentage, shuffle=true)

  #create database dictionary in format use by Zenodo
  database = Dict("trainingMacs"=>Matrix(diff_dist[train_idx, :]),
                  "testMacs"=>Matrix(diff_dist[test_idx, :]),
                  "trainingLabels"=>Matrix(sample_locs[:, train_idx]'),
                  "testLabels"=>Matrix(sample_locs[:, test_idx]'))
  database_wrap = Dict("database"=>database)

  #write file
  if tukeyλ
    filename = "SimTukeyLambda_m=$(m)_σ=$(σ).mat"
  else
    filename = "Sim$(String(nameof(typeof(d))))TDoA_θ=$(idx_params["σ"]).mat"
  end
  MAT.matwrite(datadir("sims", filename), database_wrap);
  @info "Created/overwrote $(filename)"
end #end of loop over params
