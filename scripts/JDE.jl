module JDE

export denoise
export jde

using LinearAlgebra, DataFrames
using MLJ, Logging, ProgressMeter
using Random: MersenneTwister

function denoise(Ydf::DataFrameRow, λ::Float64, γ::Float64)
  """
  we want to solve the following problem (which is elasticnet):
   (dictionary form,  rows are features, columns are examples)

    ||Y - DB||^2_2 + λ||B||_2 + γ||B||_1 

    Y: Fx1
    D: [d; I] = [Fx(M-F); FxF] = FxM
    B: Mx1
 
  attach an identity matrix to end of D to impulsive noise cancellation

  assume that passed Y has features as columns according to general
  conventions. 

  return denoised version of Y
  """
  #suppress warning messages
  Logging.disable_logging(Logging.Warn)

  #coerce type for MLJ model
  Y=coerce(Vector(Ydf), Continuous)
  #get sizes
  f = size(Y,1)

  #generate random dictionary... with Identity matrix appended... for now
  #letting M = f*11
  m = f*11
  d = 1.0*rand(MersenneTwister(123), Float64, (f,m-f)).-1/2.0
  D = [d diagm(ones(f))]
  #normalize D (p=2 norm foreach col is 1.0)
  colnorms = [norm(x) for x in eachcol(D)]
  D = D ./ colnorms'
  Dt = table(D)

  #create elastic net model, pass in data, solve
  model = (@load ElasticNetRegressor pkg ="MLJLinearModels" verbosity=0)(lambda=λ, gamma=γ, scale_penalty_with_samples=false)
  mach  = machine(model, Dt, Y)
  MLJ.fit!(mach, verbosity=0)

  #unsuppress warning messages
  Logging.disable_logging(Logging.Debug)

  #return transformed Y (multiply by slope, add offset)
  return D[:,1:(m-f)]*(mach.fitresult[1][1:(m-f)]).+mach.fitresult[1][end]
end

function jde(Ydf::DataFrame, Ddf::DataFrame, λ::Float64, γ::Float64)
  """
  we want to solve the following problem (which is elasticnet):
   (dictionary form,  rows are features, columns are examples)

    ||Y - DX||^2_2 + λ||X||_2 + γ||X||_1 

    Y: Mx1
    D: [d; I] = [MxN; MxM] = MxN′
    X: [x x_I] = N′x1
 
  attach an identity matrix to end of D to impulsive noise cancellation

  assume that passed Y has features as columns according to general
  conventions

  return x 

  """
  #suppress warning messages
  Logging.disable_logging(Logging.Warn)

  #coerce type for MLJ model
#  Y=coerce(Matrix(Ydf), Continuous)
  #get feature-sizes, Ydf is row-major form, cols=features, rows=samples
  #Ddf is row-major form, N atoms (atom is row), 
  nydf = size(Ydf,1)
  fydf = size(Ydf,2)
  Nddf = size(Ddf,1)
  Mddf = size(Ddf,2) 

  #add Identity to end of dictionary D
  #Ddf is row-major, so convert
#  D = permutedims(Matrix(Ddf))
  D = [permutedims(Matrix(Ddf)) diagm(ones(fydf))]
  #normalize D (p=2 norm foreach col is 1.0)
  colnorms = [norm(x) for x in eachcol(D)]
  D = D ./ colnorms'
  Dt = table(D)

  #create matrix holding estimate coefficients
  X = zeros(Nddf, nydf) 

  #create elastic net model, pass in data, solve
#  model = (@load ElasticNetRegressor pkg ="MLJLinearModels" verbosity=0)(lambda=λ, gamma=γ, scale_penalty_with_samples=false)
  model = (@load LassoRegressor pkg ="MLJLinearModels" verbosity=0)(lambda=λ)

  #iterate over the samples (rows) of Ydf 
  @showprogress "Iterating over data..." for (idx, row) in enumerate(eachrow(Ydf))

    mach  = machine(model, Dt, Vector(row))
    MLJ.fit!(mach, verbosity=0)

    X[:, idx] = mach.fitresult[1][1:Nddf] #if want impulse portion, mach.fitresult[1] 
  end

  #unsuppress warning messages
  Logging.disable_logging(Logging.Debug)

  #return transformed Y (multiply by slope, add offset)
  return X
end
function jde(Ydf::DataFrame, Ddf::DataFrame, MLJmdl; denoise::Bool=true)

  #suppress warning messages
  Logging.disable_logging(Logging.Warn)

  #coerce type for MLJ model
#  Y=coerce(Matrix(Ydf), Continuous)
  #get feature-sizes, Ydf is row-major form, cols=features, rows=samples
  #Ddf is row-major form, N atoms (atom is row), 
  nydf = size(Ydf,1)
  fydf = size(Ydf,2)
  Nddf = size(Ddf,1)
  Mddf = size(Ddf,2) 

  #add Identity to end of dictionary D
  #Ddf is row-major, so convert
  denoise ? D = [permutedims(Matrix(Ddf)) diagm(ones(fydf))] :  D = permutedims(Matrix(Ddf))
  #normalize D (p=2 norm foreach col is 1.0)
  colnorms = [norm(x) for x in eachcol(D)]
  D = D ./ colnorms'
  Dt = table(D)

  #create matrix holding estimate coefficients
  X = zeros(Nddf, nydf) 

  #iterate over the samples (rows) of Ydf 
  for (idx, row) in enumerate(eachrow(Ydf))

    mach  = machine(MLJmdl, Dt, Vector(row))
    MLJ.fit!(mach, verbosity=0)

    #only return coefficients for Dictionary, not Identity/impulse
    X[:, idx] = mach.fitresult[1][1:Nddf] #if want impulse portion, mach.fitresult[1] 
  end

  #unsuppress warning messages
  Logging.disable_logging(Logging.Debug)

  #return coefficients X of Y=DX 
  return X
end

#############################################################################
#encodemax identifies the maximum value in each column of a matrix, sets it 
#to 1 and all others to 0
function encodemax(Xmatrix::Matrix)
  Ematrix = zeros(size(Xmatrix))
  for idx in 1:size(Xmatrix,2)
    Ematrix[:,idx]= Xmatrix[:,idx].==maximum(Xmatrix[:,idx])
  end
  return Ematrix
end

end #end of module
