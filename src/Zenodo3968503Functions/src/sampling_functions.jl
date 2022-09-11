#sampling_functions.jl

#using <add in any packages as necessary>
using LinearAlgebra: norm
using Random: shuffle
using Parameters
using DataFrames
using MultivariateStats
using CSV

#this is duplicative to what's in QuasinormRegression pkg,
#but needed here as well so AvgDistEstimate function works
using KernelFunctions, Distances
struct PnormKernelStdAvg <: KernelFunctions.SimpleKernel 
    pnorm::Real
end
KernelFunctions.kappa(::PnormKernelStdAvg, d1::Real) = d1
KernelFunctions.metric(self::PnormKernelStdAvg) = Minkowski(self.pnorm)

"""
    maxk(a_vector, k_int)

Mimic Matlab function to find largest 'k' values and returns
associated index values
"""
function maxk(a, k::Integer)
           b = partialsortperm(a, 1:k, rev=true)
           #return collect(zip(b, a[b]))
           return b
end

#a is vector to find min values, k is number of items
"""
    mink(a_vector, k_int)

Mimic Matlab function to find smallest 'k' values and returns 
associated index values 
"""
function mink(a, k::Integer)
           b = partialsortperm(a, 1:k, rev=false)
           #return collect(zip(b, a[b]))
           return b
end;

#utility function used to estimate the density of sampling points
#this is duplicative to what's in quasinormregression module, need to fix
""" 
    AvgDistEstimate(coords, NumPoints)
 
    Estimate average distance between a set number of points.
"""
function AvgDistEstimate(coords::Matrix; NumPoints::Int64=5)

    #construct kernel to calculate all possible distance pairs
    aKt = KernelFunctions.compose(PnormKernelStdAvg(2.0), ScaleTransform(1))
    aKKt = kernelmatrix(aKt, coords, obsdim=1)
 
    #get estimate of density (4 closet points, exclude same point comparison)
    density_est = Matrix{Float64}(undef, size(aKKt,1), NumPoints)
    for (idx, rowvals) in enumerate(eachrow(aKKt))
        density_est[idx,:] = rowvals[mink(rowvals, NumPoints)]
    end
    density_est = sum(density_est)*(NumPoints)/(length(density_est)*(NumPoints-1))
  
    return density_est
end#end function


#create parameters structure for to work with getindices below and DrWatson
#bare minimum to declare is:
#   SamplingParams = SampleParamsStruct(packing_dir=datadir("packing_coords"))
@with_kw mutable struct SampleParamsStruct{}
    sphere_packing::Bool = false
    packing_dir::String  = nothing
    rand_offset::Float64 = 0.0     #this is multiple of avg density 
    grp_flag::Bool       = false
    grp_val::Integer     = 5
    num_runs::Integer    = 1
    bagging_runs::Integer= 1
end

"""

  getindices(data_coords, train_size, SampleParamsStruct)

  Wrapper function that calls different sampling functions to generate
  a set of train, val, test indices.  
  
  data_coords is dataframe of sampling locations.  

  train_size is fraction or integer value.
  
  SampleParamStruct has following fields:

    - sphere_packing::Bool = false
    - packing_dir::String  = nothing (location of files)
    - rand_offset::Float64 = 0.0   #this value is used as multiplicative of avg sampling dist.
    - grp_flag::Bool       = false #whether to return group of indices around each sphere center
    - grp_val::Integer     = 5     #number of indices for each group
    - num_runs::Integer    = 1     #set for multiple runs to get avg, std of performance
    - bagging_runs         = 1     #to use with rand_offset for bagging
"""
function getindices(data_coords::DataFrame, train_size::Union{Integer, Float64}, Sstruct::SampleParamsStruct)

  #if using sphere packing to setup training points
  if Sstruct.sphere_packing

    if train_size < 1.0 
      error("train_size $train_size is not compatible with sphere_packing $(Sstruct.sphere_packing)")
    end
    #get ratio -- only works for x,y (2d)  not for x,y,z (3d) case 

    #run PCA to rotate major and minor axis to align with cartesian x,y
    #principle axis, then add rotate coords back into data
    M = MultivariateStats.fit(MultivariateStats.PCA, transpose(Matrix(data_coords[:,[:x, :y]])))
    data_coordsrot = MultivariateStats.transform(M, transpose(Matrix(data_coords[:,[:x, :y]])))
    #data_coords.x = data_coordsrot[1,:]; data_coords.y = data_coordsrot[2,:];
    data_coordsrot = DataFrame(x=data_coordsrot[1,:], y=data_coordsrot[2,:], z=data_coords.z)

    #get dimensions of locs -- use statistical version to avoid outlier
    #distortion
    x_width  = abs(maximum(data_coordsrot[:, :x])-minimum(data_coordsrot[:, :x]))
    y_width  = abs(maximum(data_coordsrot[:, :y])-minimum(data_coordsrot[:, :y]))
    #x_width  = std(data_coordsrot[:, :x])*2.5
    #y_width  = std(data_coordsrot[:, :y])*2.5
    #if x_width is greater than y_width, need to flip coords
    x_width > y_width ? flip_xy=false : flip_xy=true
    #round ratio to tenth decimal
    flip_xy ? ratio=round(x_width/y_width, digits=1) : ratio=round(y_width/x_width, digits=1) 
    #last error check
    if ratio > 1.0 error("bad ratio for packing: $ratio") end 
    #@info "packing ratio: $ratio"
    if ratio <0.3 
      @warn "width to length ratio is too low at $ratio."
      return nothing, nothing, nothing
    end
    #now that have ratio, need to get number of sample locations
    #first, make sure can handle number of points
    if train_size > size(data_coordsrot,1)
      @warn "train_size is too large"
      return nothing, nothing, nothing
    else
      #and then load correct data files. 
      packing_df = load_packing_coords(Sstruct.packing_dir, train_size, ratio=ratio);  #ratio is default at 0.6 for JHU data
      #average density to set random offset from packing location
      r_off = AvgDistEstimate(Matrix(data_coordsrot), NumPoints=5)
    end

    #now get indices based on sphere_packing
    train_percentage = train_size/size(data_coordsrot,1)
    train_idx = periodic_sampling(train_percentage, size(data_coordsrot,1), data_coordsrot, packing_df, 
                                  rand_float=Sstruct.rand_offset*r_off, group_flag=Sstruct.grp_flag)
    return train_idx, nothing, nothing

  #now for non sphere packing case
  else
    #passing specific number of samples (random)
    if train_size > 1
      train_percentage = train_size/size(data_coords,1)
      train_idx, valid_idx  = partition(eachindex(data_coords[:,1]), train_percentage, shuffle=true)
      return train_idx, valid_idx, nothing
    #use percentage of given samples (standard partition)
    elseif train_size < 1
      train_idx, valid_idx = partition(eachindex(data_coords[:,1]), train_size, shuffle=true)
      return train_idx, valid_idx, nothing
    #use all samples to train, just shuffle the indices
    elseif train_size == 1
      train_idx = shuffle(eachindex(data_coords[:,1]))
      return train_idx, nothing, nothing
    end

    #logic area to deal with other sampling situations (modify else)

  end 
end#end getindices function


###############################################################################
#  Sampling functions used to selectively pick different training locations. 
#   - method to pick based on order of measurements
#   - method to pick based on optimal sphere packing
#  
#  Packing parameter function to load coordinates based on files from
#  packomania.com
###############################################################################

"""
  load_packing_coords(file_location_str, num_locs_IntVec, ratio_float)

  Looks at files in <file_locations_st> and finds ones that corresponds to
  the passed integers in <num_locs>.  Here, assumes that <ratio> is 0.6. 
  Review https://packomania.com as standard ratios {0.1 : 0.1 : 1.0}.

  file_location_str is string with file location, e.g., "./data/packing/".
  Corresponds to location of files with sphere packing coordinates.

  num_locs_IntVec is vector of integers, e.g., [9, 18, 37]. Corresponds to the
  number of spheres that can be packed in rectangular with width:height ratio
  of <ratio_float>.  Together with <ratio_float>, determines which info is loaded.

  ratio_float is a float that corresponds to ratio (width:height) of rectangular 
  that encompasses all possible location values

  Returns as grouped dataframe with each group (reflected by group label) 
  corresponding to an integer in the provide <num_locs_IntVec>
"""
function load_packing_coords(file_location::String, num_locs::Vector{Int64}; ratio::Float64=0.6)
  
  #create empty dataframe to return
  packing_df = DataFrames.DataFrame()
  
  #cycle over each integer
  for idx in num_locs
    #find and get filename for desired packing coordinates
    filename = filter!(s->occursin(r"crc"*string(idx)*"_"*string(ratio),s),readdir(file_location))
    if size(filename,1) == 0 error("no files matched parameters, num pts="*string(idx)*", ratio="*string(ratio)) end
    @debug idx print(filename," ")
    temp_df = DataFrames.DataFrame(CSV.File(joinpath(file_location,filename[1]), header=["grp", "x", "y"], delim=' ', ignorerepeated=true))
    temp_df[!, :grp]=temp_df[!, :grp].*0 .+ idx
    append!(packing_df, temp_df)
  end
  
  #create grouped dataframe
  packing_df = DataFrames.groupby(packing_df, :grp)

  return packing_df
end

function load_packing_coords(file_location::String, num_locs::Integer; ratio::Float64=0.6)
  
  #create empty dataframe to return
  packing_df = DataFrames.DataFrame()
  
  idx = num_locs
  #find and get filename for desired packing coordinates
  filename = filter!(s->occursin(r"crc"*string(idx)*"_"*string(ratio),s),readdir(file_location))
  if size(filename,1) == 0 error("no files matched parameters, num pts="*string(idx)*", ratio="*string(ratio)) end
  @debug idx print(filename," ")
  temp_df = DataFrames.DataFrame(CSV.File(joinpath(file_location,filename[1]), header=["grp", "x", "y"], delim=' ', ignorerepeated=true))
  temp_df[!, :grp]=temp_df[!, :grp].*0 .+ idx
  
  #create grouped dataframe
  packing_df = DataFrames.groupby(temp_df, :grp)

  return packing_df
end
"""
    periodic_sampling(s_int, t_int)

    Periodically sample the range(1, t_int, step=1) of values and return. Used
    for generating a periodically sampled index.

    s_int is number of values to sample out of total t_int values
"""
function periodic_sampling(num_samples::Integer, num_total::Integer)

    sample_freq = num_total/num_samples

    return round.(Int, sample_freq.*collect(range(1, num_samples, step=1)))
end

"""
    periodic_sampling(s_float, t_int)

    Periodically sample the range(1, t_int, step=1) of values and return. Used
    for generating a periodically sampled index.

    s_float is percentage of total t_int values to sample
"""
function periodic_sampling(percentage_samples::Float64, num_total::Integer)

    sample_points = round(Int, percentage_samples*num_total)
    sample_freq   = 1.0/percentage_samples

    return floor.(Int, sample_freq.*collect(range(0, sample_points-1, step=1))).+1
end

"""
    periodic_sampling(s_float, t_int, locs_dataframe, pack_coords_dataframe, rand_float)

    Periodically sample the available locations based on max packing concept, i.e.,
    coordinates are chosen that maximums radius between sampled locations.  Assumes
    that locs_dataframe is all the possible sampled locations within a rectangular
    area.  pack_coords are the coordinates for maximum packing given in normalized
    units, see http://packomania.com/.

    s_float is percentage of total t_int values to sample.

    locs is assumed to be Nx2 dataframe with header names [:x, :y]

    pack_coords is assumped to be a grouped dataframe.  Each group is based on
    number of coords, e.g., groups 9, 18, 35, 71.

    rand_float (default=0.0) sets the maximum uniformly distributed random offset. This parameter
    is used to address possibility that chosen packing location is a particularly
    bad set of measurements.

    group_flag (default=false) grabs all values within rand_float dimensional offset
    of a given packing location.  This essentially identifies and selects a sampled 
    locations within a given radius (rand_flag*sqrt(2)) of packing location.

    Packing coordinates..
    - fill a rectangul that is centered on 0,0.
    - x_dimension is normalized to 1
    - y_dimension is normalized (and assumed less than x_dimension)

    Therefore packing coords need to be scaled to rectangular area of the sampled
    locations.  Then find closest sampled location to each scaled packing coords.
    Note the assumption is sampled area has rectangular shape oriented with axes
    along x and y dimension.  If rotated off axis then will distort resulting
    locations.

    Return the index for these locations.

    See http://packomania.com/
"""
function periodic_sampling(
        percentage_samples::Float64,
        num_total::Integer,
        locs::DataFrames.DataFrame,
        pack_locs_in::GroupedDataFrame{DataFrames.DataFrame};
        #optional parameters
        rand_float::Float64=0.0, 
        group_flag::Bool=false)


    #how many indices?
    if percentage_samples > 1.0 error("percentage samples is greater than 1: $percentage_samples") end
    sample_points = round(Int, percentage_samples*num_total)
    #check that pack_locs has appropriate size group
    temp=[0]
    for idx in keys(pack_locs_in) push!(temp, idx[1][1]) end
    if !any(temp .== sample_points) # if a match, then make false since good
        error("A optimal packing spheres file doesn't exist for given number of points: "*string(sample_points))
    end
    #get specific group that we care about (change frome subDataFrame)
    pack_locs = DataFrames.DataFrame(pack_locs_in[(sample_points, )])

    #get dimensions of locs
    x_width  = abs(maximum(locs[:, :x])-minimum(locs[:, :x]))
    y_width  = abs(maximum(locs[:, :y])-minimum(locs[:, :y]))
    #x_width  = std(locs[:, :x])*2.5
    #y_width  = std(locs[:, :y])*2.5
    scale_width = x_width
    
    #get center of locs
    x_center = (maximum(locs[:, :x])-minimum(locs[:, :x]))/2.0 + minimum(locs[:, :x])
    y_center = (maximum(locs[:, :y])-minimum(locs[:, :y]))/2.0 + minimum(locs[:, :y])
    #x_center = mean(locs[:, :x])
    #y_center = mean(locs[:, :y])

    #flip x,y for pack_locs?  yes if y_width is greater than x_width
    x_width > y_width ? flip_xy=false : flip_xy=true

    #scale, shift, and apply rand offset to pack_locs
    if flip_xy
        rename!(pack_locs, :x=>:y, :y=>:x)
        scale_width = y_width
    end

    if group_flag # if true, get transformed location with no offset
      transform!(pack_locs, :x => (x-> x.*scale_width.+x_center) => :x)
      transform!(pack_locs, :y => (y-> y.*scale_width.+y_center) => :y)
    else          # if false, get transformed location with random offset
      transform!(pack_locs, :x => (x-> x.*scale_width.+x_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :x)
      transform!(pack_locs, :y => (y-> y.*scale_width.+y_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :y)
    end


    #now find the closest sampled location to each packing location
    sampling_idx = Int[]
    #outer loop over pack_locs
    for idx_pack in range(1, sample_points, step=1)
        temp_pack = Vector(pack_locs[idx_pack, [:x, :y]])
        min_idx = 1
        min_dist= Inf
        #inner loop over all possible sample locations
        for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(temp_pack - Vector(locs[idx_locs, [:x, :y]]))

            #get single location or grouping of locs around packing locs?
            if temp_dist < min_dist
              min_dist = temp_dist
              min_idx = idx_locs
            end
        end

        #now that found closest training sample to packed location
        #search over training samples again to add to group if flag is set
        #add samples that are within the offset distance
        if group_flag
          #search over all training samples 
          for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(Vector(locs[min_idx,[:x, :y]]) - Vector(locs[idx_locs, [:x, :y]]))

            if temp_dist < sqrt(2)*rand_float
              push!(sampling_idx, idx_locs)
            end
          end #idx_locs loop
        #else simply save sampling idx
        else
          push!(sampling_idx, min_idx)
        end #group_flag loop
    end #idx_pack loop

    return sampling_idx
end;

"""
    periodic_sampling(s_int, t_int, locs_dataframe, pack_coords_dataframe, rand_float)

    Periodically sample the available locations based on max packing concept, i.e.,
    coordinates are chosen that maximums radius between sampled locations.  Assumes
    that locs_dataframe is all the possible sampled locations within a rectangular
    area.  pack_coords are the coordinates for maximum packing given in normalized
    units, see http://packomania.com/.

    s_int is number of values to sample. t_int is used to bound-check (s_int < t_int)

    locs is assumed to be Nx2 dataframe with header names [:x, :y]

    pack_coords is assumped to be a grouped dataframe.  Each group is based on
    number of coords, e.g., groups 9, 18, 35, 71.

    rand_float sets the maximum uniformly distributed random offset. This parameter
    is used to address possibility that chosen packing location is a particularly
    bad set of measurements

    Packing coordinates..
    - fill a rectangulare that is centered on 0,0.
    - x_dimension is normalized to 1
    - y_dimension is normalized (and assumed less than x_dimension)

    Therefore packing coords need to be scaled to rectangular area of the sampled
    locations.  Then find closest sampled location to each scaled packing coords.
    Return the index for these locations.

    See http://packomania.com/
"""
function periodic_sampling(
        sample_points::Integer,
        num_total::Integer,
        locs::DataFrames.DataFrame,
        pack_locs_in::GroupedDataFrame{DataFrames.DataFrame};
        #optional parameters
        rand_float::Float64=0.0)


    #how many indices?
    if sample_points > num_total 
      error("num_samples ("*string(sample_points)*") is greater than sample size ("*string(num_total)*")")
    end

    #check that pack_locs has appropriate size group
    temp=[0]
    for idx in keys(pack_locs_in) push!(temp, idx[1][1]) end
    if !any(temp .== sample_points) # if a match, then make false since good
        error("A optimal packing spheres file doesn't exist for given number of points: "*string(sample_points))
    end
    #get specific group that we care about (change frome subDataFrame)
    pack_locs = DataFrames.DataFrame(pack_locs_in[(sample_points, )])

    #get dimensions of locs
    x_width  = abs(maximum(locs[:, :x])-minimum(locs[:, :x]))
    y_width  = abs(maximum(locs[:, :y])-minimum(locs[:, :y]))
    #x_width  = std(locs[:, :x])*2.5
    #y_width  = std(locs[:, :y])*2.5
    scale_width = x_width
    
    #get center of locs
    x_center = (maximum(locs[:, :x])-minimum(locs[:, :x]))/2.0 + minimum(locs[:, :x])
    y_center = (maximum(locs[:, :y])-minimum(locs[:, :y]))/2.0 + minimum(locs[:, :y])
    #x_center = mean(locs[:, :x])
    #y_center = mean(locs[:, :y])

    #flip x,y for pack_locs?  yes if y_width is greater than x_width
    x_width > y_width ? flip_xy=false : flip_xy=true

    #scale, shift, and apply rand offset to pack_locs
    if flip_xy
        rename!(pack_locs, :x=>:y, :y=>:x)
        scale_width = y_width
    end
    transform!(pack_locs, :x => (x-> x.*scale_width.+x_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :x)
    transform!(pack_locs, :y => (y-> y.*scale_width.+y_center.+(rand_float.*(2.0.*rand(sample_points).-1.0))) => :y)


    #now find the closest sampled location to each packing location
    sampling_idx = Int[]
    #outer loop over pack_locs
    for idx_pack in range(1, sample_points, step=1)
        temp_pack = Vector(pack_locs[idx_pack, [:x, :y]])
        min_idx = 1
        min_dist= Inf
        #inner loop over all possible sample locations
        for idx_locs in range(1, num_total, step=1)
            temp_dist = norm(temp_pack - Vector(locs[idx_locs, [:x, :y]]))
            #get new index if distance is smaller
            if temp_dist < min_dist
                min_dist = temp_dist
                min_idx = idx_locs
            end
        end
        #save sampling idx
        push!(sampling_idx, min_idx)
    end

    return sampling_idx
end;

