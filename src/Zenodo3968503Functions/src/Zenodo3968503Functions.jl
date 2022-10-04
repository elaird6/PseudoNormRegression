module Zenodo3968503Functions

using MAT
using DataFrames
import MLJ: table, partition
import Statistics: mean, var
using Parameters
include("datatransform_functions.jl")
include("sampling_functions.jl")

export ReadZenodoMATParams
export ReadZenodoMatFiles
export SampleParamsStruct
export getindices
export ReadJHUcsvParams
export ReadJHUcsvFiles

#create parameters structure for ReadZenodoMATfiles
@with_kw struct ReadZenodoMATParams
    avgmeas::Bool    = true
    flag100::Bool    = true
    flagval::Float64 = -999.0
    verbose::Bool    = false
    datarep::Union{DataType,Nothing} = nothing 
end

"""
    ReadZenodoMatFiles(filename; avgmeas=true, flag100=true, flagval=-999.0, verbose=false, datarep=nothing)
    
    Reads in Matlab MAT files that are availabe at Zenodo.org under ID 3968503 or of the same
    format). Output are training and testing dataframes. Measurements can be averaged ("avgmeas") in
    the training data -- there are often multiple measurements at given locations.  Also, 
    in lieu of missing data, original data often puts in "100". If wish to change to different
    value, use remaining arguments after setting flag100=true. Note that setting flagval=-999.0 will 
    result in (min value in data -1) to replace "100" rather than passed value. 
    To transform the data, change 'datarep' to something like DTpowed, DTexponential, or DTpositive.

    Outputs tuple of dataframes: (train_data, train_locations, test_data, test_locations)

    Use Params keyword struct ReadZenodoMATParams to set and pass default values
"""
function ReadZenodoMatFiles(filename::String; 
    avgmeas::Bool=true, 
    flag100::Bool=true, 
    flagval::Float64=-999.0, 
    verbose::Bool=false, 
    datarep::Union{DataType,Nothing}=nothing)
 
    #read in the file
    zenodo_data = MAT.matread(filename);
    zenodo_data = zenodo_data["database"];
    
    #some flags
    flag_noise_val = flagval
    flag_avg       = avgmeas; #if want to average measurement at some locations (train data)

    #set up dataframe to store values
    X_df = DataFrames.DataFrame();
    y_df = DataFrames.DataFrame()
    X_df_validation = DataFrames.DataFrame();
    y_df_validation = DataFrames.DataFrame()

    #determine number of entries
    dict_strings = ["trainingLabels", "trainingMacs", "testLabels", "testMacs"]
    for idx in dict_strings
        #@printf("%s:\tsize=(%d, %d)\n",idx, size(zenodo_data[idx])...)
        s1, s2 = size(zenodo_data[idx])
        if verbose println("$idx: \tsize=($s1, $s2)") end
        @debug size(zenodo_data[idx])
    end

    #set up training dataframes according to max number of rows
    row_num = size(zenodo_data["trainingLabels"],1)

    for idx in 1:row_num

        #pull out a row of input data, locs
        temp_data = zenodo_data["trainingMacs"][idx,:]
        #temp_locs = zenodo_data["trainingLabels"][idx,1:3]
        temp_locs = zenodo_data["trainingLabels"][idx,:]

        #push data points onto dataframe
        #push coordinates onto dataframe (only taking x,y,height)
        try
            push!(X_df, (temp_data'))
            push!(y_df, temp_locs')  #technically should not be in same try, safe to assume
        catch
            X_df = DataFrames.DataFrame(table(temp_data'))
            y_df = DataFrames.DataFrame(table(temp_locs'))
        end
    end

    #replace "100" artifact with minimum value or given value
    if flag100 && flag_noise_val==-999.0
      flag_noise_val = minimum(Matrix(X_df))-1;
      X_df = DataFrames.DataFrame(replace(Matrix(X_df), 100.0=> flag_noise_val), :auto);
    elseif flag100
      X_df = DataFrames.DataFrame(replace(Matrix(X_df), 100.0=> flag_noise_val), :auto);
    end

    #set up validation dataframes according to max number of rows
    row_num = size(zenodo_data["testLabels"],1)
    for idx in 1:row_num

        #pull out a row of input data, locs
        temp_dataV = zenodo_data["testMacs"][idx,:]
        #for locations, only interested in x, y, z... not floor or bldg ID
        #temp_locsV = zenodo_data["testLabels"][idx,1:3]
        temp_locsV = zenodo_data["testLabels"][idx,:]

        #push data points onto dataframe
        try
            push!(X_df_validation, (temp_dataV'))  #should not be in same "try", but safe to assume 
            push!(y_df_validation, temp_locsV')
        catch
            X_df_validation = DataFrames.DataFrame(table(temp_dataV'))
            y_df_validation = DataFrames.DataFrame(table(temp_locsV'))
        end
    end
    
    #replace "100" artifact with minimum value or given value
    if flag100 && flag_noise_val==-999.0
      flag_noise_val = minimum(Matrix(X_df))-1;
      X_df_validation = DataFrames.DataFrame(replace(Matrix(X_df_validation), 100.0=> flag_noise_val), :auto);
    elseif flag100
      X_df_validation = DataFrames.DataFrame(replace(Matrix(X_df_validation), 100.0=> flag_noise_val), :auto);
    end

    #rename columns in dataframes
    rename!(y_df, [1=>:x, 2=>:y, 3=>:z, 4=>:floor, 5=>:bldg]);
    rename!(y_df_validation, [1=>:x, 2=>:y, 3=>:z, 4=>:floor, 5=>:bldg]);

    #average fingerprints at each locations or not? for kernel methods, generally a good idea
    if flag_avg
        temp = hcat(y_df, X_df)
        if verbose println("\nAveraging of training fingerprinting selected... \n") end
        #average measurements
        temp = DataFrames.groupby(temp, [:x, :y, :z, :floor, :bldg])
        temp = combine(temp, valuecols(temp) .=> mean)
        y_df = temp[!, [:x, :y, :z, :floor, :bldg]]
        X_df = DataFrames.select(temp, r"x[0-9]")
        @debug size(y_df)
        @debug size(X_df)

        if false  #should not average test points
            if verbose println("\nAveraging of testing fingerprinting selected...!!! Why?!?! \n") end
            temp = hcat(y_df_validation, X_df_validation)
            #average measurements
            temp = DataFrames.groupby(temp, [:x, :y, :z, :floor, :bldg])
            temp = combine(temp, valuecols(temp) .=> mean)
            y_df_validation = temp[!, [:x, :y, :z, :floor, :bldg]]
            X_df_validation = DataFrames.select(temp, r"x[0-9]")
            @debug size(y_df_validation)
            @debug size(X_df_validation)
        end
    end #end of measurement averaging

    #transform data based on passed representation
    if isa(datarep, DataType) 
      X_df = datarep(X_df)
      X_df_validation = datarep(X_df_validation)
    end
    
    return (X_df, y_df, X_df_validation, y_df_validation)

end #end of function

"""
    ReadZenodoMatResultsFiles(filename)

    Read in Zenodo 3968503 MAT results file and returns mean
    error along with database, KNN algorithm, and KNN distance. Note
    that function assumes filename includes directory structure as
    provided by Zenodo 3968503 repository:
    <../database/knn_alg/knn_distance/result_file(s)>
   
"""
function ReadZenodoMatResultsFiles(filename::String)
    #read in the file
    zenodo_data = MAT.matread(filename);
    zenodo_data = zenodo_data["results"];

    error = mean(zenodo_data["error"][:,2]) #3d error
    db, alg, dist = split(filename, "/")[end-3:end]

    return (db, alg, dist, error)

end; #end of function

"""
  ReadZenodoMatResultsDir(directory_name)

  Walks the Zenodo 3968503 MAT result directories and returns
  a dataframe with mean error, database, KNN algorithm, and KNN
  mean error.  

"""
function ReadZenodoMatResultsDir(pathname::String)
  #create dataframe to store data
  df = DataFrames.DataFrame(db=String[], alg=String[], dist=String[], mee=Float64[])
  #walk directory given
  for (root, dirs, files) in walkdir(pathname)
    if !isempty(files)
      for file in files
        db, alg, dist, error = ReadZenodoMatResultsFiles(joinpath(root, file)) # path to files
        push!(df, (db, alg, dist, error))
      end #end iterating over files
    end #end if check
  end #end for loop

  return df
end #end function
 
###################################################################
# Utility function to subsample Zenodo MAT files. Requires that
# sampling_function.jl file be included to work properly
#
###################################################################

"""
    SampleSaveZenodoMatFile(filename, train_size, SampleParamsStruct)

    Creates a sampling index for a Zenodo MAT file training labels and MAC,
    samples and saves the file. New file is saved next to old.

    Returns True if no issue and False otherwise (i.e., bad indices)

    #Example
    SampleSaveZenodoMatFile(/data/TUT/TUT6.mat, 9, SampleParamStruct)

    Saves TUT6_9.mat in /data/TUT
"""
function SampleSaveZenodoMatFile(filename::String, train_size::Union{Float64, Integer}, rzmp::ReadZenodoMATParams, sps::SampleParamsStruct)

  #read in MAT file and convert to dataframes... reduce to minimum samples to
  #check number of viable locations in packing spheres
  X_df,y_df, X_df_valid, y_df_valid = ReadZenodoMatFiles(filename; avgmeas=true, flag100=false) 

  if train_size > size(y_df,1)
    @warn "train_size $train_size is too large for $(size(y_df,1)) entries of $filename"
    return false
  end

  #now that number of viable locations is good, read in again
  X_df,y_df, X_df_valid, y_df_valid = ReadZenodoMatFiles(filename; avgmeas=rzmp.avgmeas, 
                                                         flag100=rzmp.flag100, flagval=rzmp.flagval, verbose=rzmp.verbose)

  #get sampling index (train_idx)
  train_idx, valid_idx, test_idx = getindices(y_df, train_size, sps) 

  if train_idx == nothing
    @warn "getindices failed for $filename"
    return false
  end

  #create database dictionary in format use by Zenodo
  database = Dict("trainingMacs"=>Matrix(X_df[train_idx, :]), 
                  "testMacs"=>Matrix(X_df_valid),
                  "trainingLabels"=>Matrix(y_df[train_idx, :]), 
                  "testLabels"=>Matrix(y_df_valid))
  database_wrap = Dict("database"=>database)

  #write database to file
  filename_strings = split(filename, "/")
  num_strings = size(filename_strings,1)
  if filename_strings[1] == "" filename_strings[1]=" " end
  dbname = filename_strings[end]
  #dbname_strings = split(dbname, ".")
  #dbname_strings[1] = string(dbname_strings[1], "_$(train_size).")
  dbname = replace(dbname, "."=>"_$(train_size).")
  newname = joinpath(filename_strings[1:end-1]..., dbname) 
  #newname = joinpath(filename_strings[1:end-1]..., string(dbname_strings...)) 
  if filename_strings[1] == " " newname=chop(newname, head=1, tail=0)  end

  MAT.matwrite(newname, database_wrap);


  #note that writing MAT to file creates a file with different variables, not a struct that Octave/Matlab algorithms are expecting
  #following lines are Octave/Matlab code to create struct file
#  database.trainingMacs   = trainingMacs;
#  database.testMacs       = testMacs;
#  database.trainingLabels = trainingLabels;
#  database.testLabels     = testLabels;
#  save -mat7-binary LTS01.mat database

  return true

end

#create parameters structure for ReadZenodoMATfiles
@with_kw struct ReadJHUcsvParams
    avgmeas::Bool    = true
    flag100::Bool    = true
    flagval::Float64 = -999.0
    verbose::Bool    = false
    datarep::Union{DataType,Nothing} = nothing 
    typesRegex::String="tdoa|rss|aoa"
    filtcolsRegex::Vector{String}=["\n"]
end

"""
    ReadJHUcsvFiles(filename::String; 
                    avgmeas::Bool=true, 
                    flag100::Bool=true, 
                    flagval::Float64=-999.0, 
                    verbose::Bool=false, 
                    datarep::Union{DataType,Nothing}=nothing,
                    typesRegex::String="tdoa|rss|aoa",
                    filtcolsRegex::Vector{String}=["\n"])
    
    Reads in JHU CSV file that is available at Zenodo.org under ID 6795580. 
    Output are training and testing dataframes. Measurements can be averaged ("avgmeas") in
    the training data -- there are often multiple measurements at given locations. Use flag100
    if wish to change minimum value to flagval. Note that setting flagval=-999.0 will 
    result in (min value in data -1) to replace "100" rather than passed value. 
    To transform the data, change 'datarep' to something like DTpowed, DTexponential, or DTpositive.

    To select passed features (train_data, test_data), set Regex variable to select appropriate
    columns -- passing filter. To subsequently remove any columns, use filtcoslRegex -- blocking 
    filter.

    Outputs tuple of dataframes: (train_data, train_locations, test_data, test_locations)

    Use Params keyword struct ReadZenodoMATParams to set and pass default values
"""
function ReadJHUcsvFiles(filename::String; 
    avgmeas::Bool=true, 
    flag100::Bool=true, 
    flagval::Float64=-999.0, 
    verbose::Bool=false, 
    datarep::Union{DataType,Nothing}=nothing,
    typesRegex::String="tdoa|rss|aoa",
    filtcolsRegex::Vector{String}=["\n"])
 
    #read in the file
    csv_data = DataFrames.DataFrame(CSV.File(filename));
    
    #some flags
    flag_noise_val = flagval
    flag_avg       = avgmeas; #if want to average measurement at some locations (train data)

    #filter and save data according to params
    X_df = csv_data[:,Regex(typesRegex)]
    for idx in filtcolsRegex
        X_df = select(X_df, Not(Regex(idx)));
    end
    y_df = csv_data[:,r"x|y|z"]
    y_df[!, :z] .= 0
    y_df[!, :floor] .= 0
    y_df[!, :bldg] .= 0

    #replace "100" artifact with minimum value or given value
    if flag100 && flag_noise_val==-999.0
      flag_noise_val = minimum(Matrix(X_df))-1;
      X_df = DataFrames.DataFrame(replace(Matrix(X_df), 100.0=> flag_noise_val), names(X_df));
    elseif flag100
      X_df = DataFrames.DataFrame(replace(Matrix(X_df), 100.0=> flag_noise_val), names(X_df));
    end

    #train/test split, seed for reproducibility 
    train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true, rng=1235);

    @debug "size Xdf, ydf: $(size(X_df)), $(size(y_df))"
    X_df_validation = X_df[test_idx, :];
    y_df_validation = y_df[test_idx, :];
    X_df = X_df[train_idx, :];
    y_df = y_df[train_idx, :];

    #average fingerprints at each locations or not? for kernel methods, generally a good idea
    if flag_avg
        temp = hcat(y_df, X_df)
        if verbose println("\nAveraging of training fingerprinting selected... \n") end
        #average measurements
        temp = DataFrames.groupby(temp, [:x, :y, :z, :floor, :bldg])
        temp = combine(temp, valuecols(temp) .=> mean)
        y_df = temp[!, [:x, :y, :z, :floor, :bldg]]
        X_df = DataFrames.select(temp, Regex(typesRegex))
        @debug size(y_df)
        @debug size(X_df)

        if false  #should not average test points
            if verbose println("\nAveraging of testing fingerprinting selected...!!! Why?!?! \n") end
            temp = hcat(y_df_validation, X_df_validation)
            #average measurements
            temp = DataFrames.groupby(temp, [:x, :y, :z, :floor, :bldg])
            temp = combine(temp, valuecols(temp) .=> mean)
            y_df_validation = temp[!, [:x, :y, :z, :floor, :bldg]]
            X_df_validation = DataFrames.select(temp, Regex(typesRegex))
            @debug size(y_df_validation)
            @debug size(X_df_validation)
        end
    end #end of measurement averaging

    #transform data based on passed representation
    if isa(datarep, DataType) 
      X_df = datarep(X_df)
      X_df_validation = datarep(X_df_validation)
    end
    
    return (X_df, y_df, X_df_validation, y_df_validation)

end #end of function

end # module
