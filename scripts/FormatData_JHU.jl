using DrWatson
@quickactivate "PseudoNormRegression"

@info "importing packages"
using MAT, CSV, DataFrames
using MLJ:partition

#enter file name and location
file_read = "JHU_fingerprints.csv"

#cycle and create standard
file_dict = Dict("JHUall.mat"=>"tdoa|rss|aoa", 
                 "JHUtdoa.mat"=>"tdoa", 
                 "JHUrss.mat"=>"rss", 
                 "JHUaoa.mat"=>"aoa", 
                 "JHUtdoaRSS.mat"=>"tdoa|rss"
                )

#create pointer to file to read
data_file = datadir("exp_raw", file_read); 
#read in the file
zenodo_data = DataFrames.DataFrame(CSV.File(data_file));

#cycle over dictionary to create different datafiles
for dict_iter in file_dict
    #for file_write in keys(file_dict)
    file_write= dict_iter[1]
    @info "reading $file_read which will be written to $file_write"

    #pull out tdoa, rss and aoa measurements as needed
#    X_df=zenodo_data[:,r"tdoa|rss|aoa"]
    X_df=zenodo_data[:,Regex(dict_iter[2])]

    #get aoa columns, convert to radians !! DON'T USE!!!
    #value is already radians
    #for col=names(X_df[:,r"aoa"]) X_df[!, col]=angle.(X_df[:, col]); end
    #get positions
    y_df=zenodo_data[:,r"x|y"]
    #add in floor and building (both set to 0)
    @info "adding zenodo standard z axis, floor, and bldg"
    y_df[!, :z] .= 0
    y_df[!, :floor] .= 0
    y_df[!, :bldg] .= 0

    #divide set between train and test, keep repeatable with Random seed
    @info "partitioning data"
    @info "partitioning is seeded so repeatable"
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
    @info "writing to file"
    MAT.matwrite(datadir("exp_pro",file_write), database_wrap);
    @info "finished...$(dict_iter[1])"
    println( repeat("*",80) )
end
