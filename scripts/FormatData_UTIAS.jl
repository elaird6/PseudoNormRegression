#set of function to read in and process data associated with Utias UWB TDoA
#data.

@info "importing packages..."
using DataFrames, CSV, Statistics, MAT, Logging
using MLJ:partition

#only keep groups that have at least minNum measurements
function UTIASdf_grpTransform(df_grp::GroupedDataFrame{DataFrame}, minNum::Int=8)
  """ takes UTIAS grouped dataframe, that has been created when UTIAS data is 
  grouped based on [:pose_x, :pose_y, :pos_z], and transforms each group to a
  dataframe row with each d_(i,j) within row (versus a row for every
  d_(i,j)). Ignores groups less than a minumum number

  UTIASdf_grpTransform(df_grp, minNum)

  """
  #create dataframe row that will be returned
  df = DataFrames.DataFrame(x=Float64[],y=Float64[],z=Float64[], tdoa01=Float64[], tdoa12=Float64[],
                            tdoa23=Float64[], tdoa34=Float64[], tdoa45=Float64[], tdoa56=Float64[], 
                            tdoa67=Float64[], tdoa70=Float64[])

  for sdf in df_grp
    if nrow(sdf) > 7
      #create dataframe row that will be returned
      row = DataFrames.DataFrame(x=sdf[1,:pose_x], y=sdf[1,:pose_y], z=sdf[1,:pose_z], tdoa01=0.0, tdoa12=0.0,
                                 tdoa23=0.0, tdoa34=0.0, tdoa45=0.0, tdoa56=0.0, tdoa67=0.0, tdoa70=0.0)

      #group by sensor a and sensor b
      sdf_mean = combine(groupby(sdf, [:idA, :idB]), :tdoa_meas => mean => :tdoa_meas)

      for idx in 1:size(sdf_mean,1)
        tmpSym = Symbol("tdoa",convert(Int,sdf_mean[idx, :idA]), convert(Int,sdf_mean[idx, :idB]))
        try
          row[1,tmpSym]=sdf_mean[idx,:tdoa_meas]
        catch
          @warn "Unusual anchor pair: $tmpSym, not being saved"
        end
      end
      #push row onto dataframe
      append!(df, row)
    end #if statement
  end #sdf loop

  return df
end


#make sure logging is at right level
#change logging level back to standard
Logging.disable_logging(Logging.Debug);

constx = 4  #which subdirectory of flight database (1,2,3,4)
for constx in 1:4
  minNum = 8  #minimum number of measurements per position
  tdoav  = "tdoa2" #which version of tdoa (tdoa2 is centeralized, tdoa3 is decenteralized)
  #file write 
  file_write= "UTIf$(constx).mat"
  #filename base
  fn_base = datadir("exp_raw","UTIAS_UWB_TDOA_DATASET", "dataset", "flight-dataset","csv-data","const$constx")

  #using partial assignment method (contains("tdoa2")) get list of files we want
  fn_list = filter((contains(tdoav)),readdir(fn_base))

  #need to cycle through all the files to read in data
  UTIASdata = CSV.read(joinpath(fn_base,fn_list[1]), types=Union{Float64,Missing},delim=',',DataFrame)
  @info "reading $(fn_list[1]) which will be written to $file_write"
  for idx in fn_list[2:end]
    @info "reading $idx which will be written to $file_write"
    append!(UTIASdata, CSV.read(joinpath(fn_base,idx), delim=',', debug=false, DataFrame))
  end

    #flight data for files such as const1-trial1-tdoa2.csv 
    # csv cols  name               format
    #   1~4     tdoa               timestamp(ms), Anchor-ID i, Anchor-ID j, d_i,j (m)
    #   5~8     acceleration       timestamp(ms), acc x(G), acc y(G), acc z(G)
    #   9~12    gyroscope          timestamp(ms), gyo x(deg), gyo y(G), gyo z(G)
    #   13~14   Tof laser ranging  timestamp(ms), Tof (m)
    #   15~17   Optical flow       timestamp(ms), dpixel x, dpixel y
    #   18~19   barometer          timestamp(ms), barometer(asl)
    #   20~27   ground truth pose  timestamp(ms), x(m), y(m), z(m), qx, qy, qz, qw
    #

#remove missing anchor rows and missing position rows
  @info "removing rows with missing data..."
  subset!(UTIASdata, :idA => x-> .!ismissing.(x), :pose_x => x-> .!ismissing.(x))
  #round off positions so we can get more measurements per location
  @info "rounding off position to cm rather than mm..."
  myround(x) = round(x, digits=2)
  transform!(UTIASdata, :pose_x =>ByRow(myround) => :pose_x, :pose_y =>ByRow(myround) => :pose_y, :pose_z =>ByRow(myround) => :pose_z)
  #group measurements for given position
  @info "grouping measurements based on true position..."
  UTIASdata_gp = groupby(UTIASdata, [:pose_x, :pose_y, :pose_z])

  @info "core conversion from grouped measurements to zenodo standard format..."
  UTIASdata = UTIASdf_grpTransform(UTIASdata_gp, 8)
  #add columns to correspond to standard format

  #pull out tdoa, rss and aoa measurements
  X_df = UTIASdata[:,r"tdoa"]

  #get positions
  y_df = UTIASdata[:,r"x|y|z"]
  #add in floor and building (both set to 0)
  @info "adding zenodo standard floor and bldg"
  y_df[!, :floor] .= 0
  y_df[!, :bldg] .= 0

  #divide set between train and test
  @info "partitioning data"
  train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true);#, rng=1235);

  #create database dictionary
  @info "creating Matlab-like database"
  database = Dict(
                "trainingMacs"=>Matrix(X_df[train_idx, :]),
                "testMacs"=>Matrix(X_df[test_idx, :]),
                "trainingLabels"=>Matrix(y_df[train_idx, :]),
                "testLabels"=>Matrix(y_df[test_idx, :]))
  database_wrap = Dict("database"=>database)

  #write database to file
  @info "writing to $file_write"
  MAT.matwrite(datadir("exp_pro",file_write), database_wrap);
  @info "finished..."
end
