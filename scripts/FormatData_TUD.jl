#set of function to read in and process data associated with Utias UWB TDoA
#data.

@info "importing packages..."
using DataFrames, CSV, Statistics, MAT, Logging
using MLJ:partition

#make sure logging is at right level
#change logging level back to standard
Logging.disable_logging(Logging.Debug);

#file write 
file_write= "TUD.mat"
#filename base
fn_base = datadir("exp_raw", "TUDdataset", "UWBdataset", "tdoa")

#using partial assignment method (contains("tdoa2")) get list of files we want
fn_list = filter(contains("tdoa"), readdir(fn_base))
#remove "BAD" files from the list
filter!(!contains("BAD"), fn_list)

#need to cycle through all the files to read in data
TUDdata = CSV.read(joinpath(fn_base,fn_list[1]), types=Union{Float64,Missing},delim=',',DataFrame)
@info "reading $(fn_list[1]) which will be written to $file_write"
for idx in fn_list[2:end]
  @info "reading $idx which will be written to $file_write"
  append!(TUDdata, CSV.read(joinpath(fn_base,idx), delim=',', debug=false, DataFrame))
end
#columns in CSV file:
# timeTick, gyroX, gyroY, gyroZ, accX, accY, accZ, 
# tdoa01, tdoa02, tdoa03, tdoa04, tdoa05, tdoa06, tdoa07, tdoa12, tdoa13, tdoa14, tdoa15, 
# tdoa16, tdoa17, tdoa23, tdoa24, tdoa25, tdoa26, tdoa27, tdoa34, tdoa35, tdoa36, tdoa37, 
# tdoa45, tdoa46, tdoa47, tdoa56, tdoa57, tdoa67, 
# otX, otY, otZ  <-- ground truth

#fix ' ' that is in front of column names
namevec = [(x[1]==' ' ? chop(x,head=1,tail=0) : x) for x in names(TUDdata)]
rename!(TUDdata, namevec)

#remove rows with missing values
for cols in namevec
  subset!(TUDdata, Symbol(cols) => x-> .!ismissing.(x))
end
#cut down on number measurements, round to cm
@info "rounding off position to cm rather than mm..."
myround(x) = round(x, digits=2)
transform!(TUDdata, :otX =>ByRow(myround) => :otX, :otY=>ByRow(myround) => :otY, :otZ=>ByRow(myround) => :otZ)
#groupby location, then combine by averaging measurements
@info "grouping measurements based on rounded true position and then averaging measurements..."
TUDdata = combine(groupby(TUDdata, [:otX, :otY, :otZ]), Symbol.(namevec[1:end-3]) .=> mean .=> Symbol.(namevec[1:end-3]) )
#pull out tdoa, rss and aoa measurements
X_df = TUDdata[:,r"tdoa"]

#get positions
y_df = TUDdata[:,r"otX|otY|otZ"]
rename!(y_df, [:x, :y, :z])
#add in floor and building (both set to 0)
@info "adding zenodo standard floor and bldg"
y_df[!, :floor] .= 0
y_df[!, :bldg] .= 0

#divide set between train and test, rng sets seed for reproducibility
@info "partitioning data"
train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true, rng=1235);

#create database dictionary
@info "creating Matlab-like database"
database = Dict(
                "trainingMacs"=>Matrix(X_df[train_idx, :]),
                "testMacs"=>Matrix(X_df[test_idx, :]),
                "trainingLabels"=>Matrix(y_df[train_idx, :]),
                "testLabels"=>Matrix(y_df[test_idx, :]))
database_wrap = Dict("database"=>database)

#write database to file
@info "writing to $file_write in $(datadir("exp_pro"))"
MAT.matwrite(datadir("exp_pro",file_write), database_wrap);
@info "finished..."
