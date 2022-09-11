using DataFrames

abstract type DataTransform end

struct DTpositive <: DataTransform end
struct DTpowed <: DataTransform end
struct DTexponential <: DataTransform end

"""
  DTpositive(db)

  Shifts features into nonnegative real numbers
"""
function DTpositive(db::AbstractDataFrame)

  minVal = minimum(matrix(db))

  return db .- minVal 
end

"""
  DTpowed(db)

  Shifts features into nonnegative real numbers then normalizes using 
  minimum value
"""
function DTpowed(db::AbstractDataFrame)

  epow = exp(1)
  minVal = minimum(matrix(db))
  normVal = (-minVal)^epow
  
  
  return ((db .- minVal).^epow) ./ (normVal)
end

"""
  DTexponential(db)

  Shifts features into nonnegative real numbers, raised to exponential
  then normalizes using minimum value raised to exponential
"""
function DTexponential(db)

  minVal = minimum(matrix(db))

  return exp.((db .- minVal)./24) ./ exp(-minVal/24)
end
