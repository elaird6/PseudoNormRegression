using Distances, Parameters

#abstract type UnionMetric <: Metric end
struct GD <: Metric end
"""
  Calculate the gaussian distance

  GD(p,r,σ) = 1/√(2πσ^2) * exp(-(p-r)^2 / (2σ^2)

  Return 0 if either p or r is 0.
"""
function (::GD)(p::Float64, r::Float64, σ::Float64)
  if any((p,r).==0)
    return 0 
  else 
    return 1/sqrt(2*π*σ^2) * exp((-1*(p-r)^2)/(2*σ^2))
  end
end
const gd = GD()

struct LGD <: Metric end
"""
  Calculate the log Gaussian distance

  LGD(p,r,σ) = -Σ log max (G(p,r),ϵ)
  where G(p,r) = 1/√(2πσ^2) * exp(-(p-r)^2 / (2σ^2)
        p,r:= vectors of real numbers
        ϵ  := eps() to ensure there's not a log(0)
        σ  := scalar std of data
"""
function (::LGD)(p::AbstractArray{Float64,1}, r::AbstractArray{Float64,1}, σ::Float64=5.0)
  #calculate gaussian distance
  gds = gd.(p, r, σ)

  #calculate max of either g or machine epsilon
  #(avoiding zero) for each element of array
  #maxg = max.(gd, eps())
  maxg = max.(gds, 0.0001)

  #get log of each element then sum up
  return -1*sum(log.(maxg))
end

"""
  Calculate the log Gaussian distance

  LGD(p,r,σ) = -Σ log max (G(p,r),ϵ)
  where G(p,r) = 1/√(2πσ^2) * exp(-(p-r)^2 / (2σ^2)
        p,r:= vectors of real numbers
        ϵ  := eps() to ensure there's not a log(0)
        σ  := vector std of data
"""
function (::LGD)(p::AbstractArray{Float64,1}, r::AbstractArray{Float64,1}, σ::AbstractArray{Float64,1})
  #calculate gaussian distance
  gds = gd.(p, r, σ)

  #calculate max of either g or machine epsilon
  #(avoiding zero) for each element of array
  #maxg = max.(gd, eps())
  maxg = max.(gds, 0.0001)

  #get log of each element then sum up
  return -1*sum(log.(maxg))
end
const lgd = LGD()

#struct PLGD <: Metric 
@with_kw mutable struct PLGD <: Metric 
 α::Float64=0.10
 τ::Float64=15.0
 σ::Float64=5.0
end
"""
  Calculate the penalized log Gaussian distance

  PLGD(p,r,τ,σ,α) = LGD(p,r,σ) +α(ϕ(p,r,τ) + ϕ(r,p,τ))
  where α is scaling parameter, e.g. 10-40, and τ is 
  a maximum expected value for set of features. 

  τ, σ, α are scalars
"""
#function (::PLGD)(p::Vector{Float64}, r::Vector{Float64}; τ::Float64=15, σ::Float64=5, α::Float64=1/10)
#function (::PLGD)(p, r; τ::Float64=15, σ::Float64=5, α::Float64=1/10)
function (f::PLGD)(p::AbstractArray{Float64,1}, r::AbstractArray{Float64,1})
  p1 = sum( (p .- f.τ) .* (p .>= f.τ) .* (r .== 0)  )
  p2 = sum( (r .- f.τ) .* (r .>= f.τ) .* (p .== 0)  )
  return lgd(p, r, f.σ) + f.α*(p1 + p2)
end
const plgd10=PLGD(α=0.10)
const plgd40=PLGD(α=0.40)
#"""
#  Calculate the penalized log Gaussian distance

#  PLGD(p,r,τ,σ,α) = LGD(p,r,σ) +α(ϕ(p,r,τ) + ϕ(r,p,τ))
#  where α is scaling parameter, e.g. 10-40, and τ is 
#  a maximum expected value for set of features.

#  τ, σ, α are vectors same length as p,r
#"""
#function (::PLGD)(p::Vector{Float64}, r::Vector{Float64}, τ::Vector{Float64}, σ::Vector{Float64}, α::Vector{Float64})
#  p1 = sum( α .* (p .- τ) .* (p .>= τ) .* (r .== 0)  )
#  p2 = sum( α .* (r .- τ) .* (r .>= τ) .* (p .== 0)  )
#  return lgd(p, r, σ) + (p1 + p2)
#end

struct Neyman <: Metric end
"""
  Calculate the Neyman distance

  d(p,q) = Σ ( (p-q).^2 ./ p )

  p, q are vectors of real numbers. Elements in p 
  should not be zero (divisor).
"""
function (::Neyman)(p::AbstractArray{Float64,1}, q::AbstractArray{Float64,1})
  #check if any values are 0
  p0 = p
  #if so, assign eps() which is close to 0
  p0[p.==0].=eps()

  return sum( ((p0.-q).^2) ./ p0)
end
const neyman = Neyman()

struct Sorensen <: Metric end
"""
  Calculate Sorensen distance

  d(p,q) = Σ|p-q| / Σ|p+q| 
  where p,q are vectors
"""
function (::Sorensen)(p::AbstractArray{Float64,1}, q::AbstractArray{Float64,1})
  numer = sum(abs.(p .- q))
  denom = sum(abs.(p .+ q))
  return numer/denom
end
const sorensen = Sorensen()

