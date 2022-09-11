@info "******************************************"
@info "using DrWatson and activating environment"
using DrWatson
@quickactivate "PseudoNormRegression"

@info "checking to see if proper subdirectories are in place"
cd(projectdir())
any("data" .== cd(readdir, projectdir())) ? nothing : mkdir("data")
cd(datadir())
any("exp_pro" .== cd(readdir, datadir())) ? nothing : mkdir("exp_pro")
any("exp_raw" .== cd(readdir, datadir())) ? nothing : mkdir("exp_raw")
any("sims" .== cd(readdir, datadir())) ? nothing : mkdir("sims")

@info "******************************************"
@info "now installing different data sets"
@info "******************************************"

@info "******************************************"
@info "now installing JHU data (Zenodo id 6795580) if desired"
@info "******************************************"
print("Do you want to install JHU data? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    @info "******************************************"
    @info "installing JHU data (Zenodo id 6795580)"
    @info "******************************************"
    @info "cd to data/exp_raw directory"
    cd(datadir("exp_raw"))
    @info "using curl to download JHU data"
    run(`curl -o JHU_fingerprints.csv https://zenodo.org/record/6795580/files/JHU_fingerprints.csv`)
    @info "done installing JHU data"
    @info "******************************************"
    @info "Format JHU data to be consistent with Zenodo 3968503 format"
    @info "******************************************"
    include(scriptsdir("FormatData_JHU.jl"))
    @info "done formating and saving JHU data"
else
    println("Not installing JHU Zenodo data (id 6795580)")
end

@info "******************************************"
@info "now installing Zenodoa data (id 3968503) if desired"
@info "******************************************"
print("Do you want to install Zenodo data (id 3968503)? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    @info "******************************************"
    @info "installing Zenodo data (id 3968503)"
    @info "******************************************"
    @info "cd to data/exp_raw directory"
    cd(datadir("exp_raw"))
    @info "using curl to download Zenodo data (id of 3968503)"
    run(`curl -o Zenodo_3968503.zip https://zenodo.org/record/3968503/files/Zenodo_3968503.zip`)
    @info "unzip Zenodo data"
    run(`unzip Zenodo_3968503.zip`)
    @info "done installing base Zenodo data"
    @info "moving databases in Zenodo_3968503/databases into data/exp_proc directory"
    file_list = filter!(s->occursin(r"[A-Z]{3}\d{1}\.mat",s), readdir(datadir("exp_raw","Zenodo_3968503","databases")))
    cp.(datadir.("exp_raw","Zenodo_3968503","databases",file_list), datadir.("exp_pro",file_list))
    @info "(did not move SIMxxx.mat databases as of limited value)"
    @info "done installing Zenodo data"
else
    println("Not installing Zenodo data (id 3968503)")
end

@info "******************************************"
@info "now installing TUD UWB data if desired"
@info "******************************************"
print("Do you want to install TUD UWB data? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    @info "******************************************"
    @info "installing TUD UWB data"
    @info "******************************************"
    @info "cd to data/exp_raw directory"
    cd(datadir("exp_raw"))
    @info "using curl to download TUD UWB data"
    run(`curl -o UWBdataset.zip https://data.4tu.nl/ndownloader/files/28631865`)
    @info "unzip TUD UWB data into /data/exp_raw/TUDdataset"
    run(`unzip -d TUDdataset UWBdataset.zip`)
    @info "cd to data/exp_raw/TUDdataset"
    cd(datadir("exp_raw","TUDdataset"))
    @info "renaming dataset directory (removing space)"
    run(`mv UWB\ dataset UWBdataset`)
    @info "done installing TUD UWB data"
    @info "******************************************"
    @info "Format TUD UWB data to be consistent with Zenodo 3968503 format"
    @info "******************************************"
    include(scriptsdir("FormatData_TUD.jl"))
    @info "done formating and saving TUD data"
else
    println("Not installing TUD UWB data'")
end

@info "******************************************"
@info "now installing UTIAS TDoA data if desired"
@info "******************************************"
print("Do you want to install UTIAS TDoA data (2GB download, 9GB uncompressed)? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    @info "******************************************"
    @info "installing UTIAS TDoA data"
    @info "******************************************"
    @info "cd to data/exp_raw directory"
    cd(datadir("exp_raw"))
    @info "using curl to download UTIAS TDoA data"
    run(`curl -o dataset.7z -L https://github.com/utiasDSL/util-uwb-dataset/releases/download/Dataset-v1.0/dataset.7z`)
    @info "unzip UTIAS UWB data (big file, almost 2Gb)"
    try
        run(`7zr x -oUTIAS_UWB_TDOA_DATASET dataset.7z`)
    catch
        @warn "p7zip is not installed, failed to unzip UTIAS UWB data"
    end
    @info "done installing UTIAS TDoA data"
    @info "******************************************"
    @info "Format UTIAS data to be consistent with Zenodo 3968503 format"
    @info "******************************************"
    include(scriptsdir("FormatData_UTIAS.jl"))
    @info "done formating and saving UTIAS data"
else
    println("Not installing UTIAS TDoA data")
end

@info "******************************************"
@info "now installing circle packing coordinates"
@info "******************************************"
print("Do you want to install ASCII files containing packing coordinates? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    @info "******************************************"
    @info "installing circle packing coordinates"
    @info "******************************************"
    @info "cd to data directory, if not a packing_coords directory, will make one"
    cd(datadir())
    any("packing_coords" .== cd(readdir, datadir())) ? nothing : mkdir("packing_coords")
    cd(datadir("packing_coords"))
    @info "using curl to download the various coordinate files, then will unzip them"
    for iter in collect(1:8)
        run(`curl -o crc_0.$(iter)00000000000_coords.tar.gz -L http://hydra.nat.uni-magdeburg.de/packing/crc_$(iter)00/txt/crc_0.$(iter)00000000000_coords.tar.gz`)
        run(`gunzip crc_0.$(iter)00000000000_coords.tar.gz`)
        run(`tar -xvf crc_0.$(iter)00000000000_coords.tar`)
        sleep(3)
    end
    cd(projectdir())
    @info "done installing packing coordinates text files"
else
    println("Not installing packing coordinates data")
end

@info "******************************************"
@info "now generating simulation data if desired"
@info "******************************************"
println("If choose 'n' to the following, you can always run\n
        'SimulationDataScript.jl' to generate data manually\n")
print("Do you want to generate simulated TDoA data? (y/n) [y]: ")
if any(readline() .== ["", "y"])
    include(scriptsdir("SimulationDataScript.jl"))
else
    println("Not running 'SimulationDataScript.jl'")
end


cd(projectdir())

@info "******************************************"
@info " done with the install script"
@info "******************************************"
