# PseudoNormRegression

This code base is based on the Julia Language (v1.7). Furthermore, it uses the [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) package to make it a reproducible scientific project.

> PseudoNormRegression

It is authored by Brent Laird.

This Julia repo utilizes standard and quasi-norm kernel-based fusion of heterogeneous data measurements followed by regession method of choice. Focus of this repo is the application of regression for indoor localization. It has been shown the use of the quasi-p-norm, p<1, as a similarity measure, results in improved performance over the standard p-norm. Additionally, the use of multiple kernels, one for each data measurement type --- e.g., time-difference of arrival, received signal strength, and angle of arrival for localization--- further improved performance in comparison to a single kernel.

The core of the repo leverages and builds on [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) extensively. There are two ML model constructs with one focused on single kernel and the second one focused on multi-kernel. Creation of kernels leverages [KernelFunctions.jl](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/). If utilize optimal spacing (circle packing) in sampling function, requires pulling files from packomania.com

# Clone project and download data
To (locally) reproduce this project, do the following:

1. Download/clone this code base. Notice that the measurement data is not included in the repo and will need to be downloaded independently (see step 3. below)..
2. Start Julia REPL and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```
   This will install all necessary packages for you to be able to run the scripts and everything should work out of the box, including correctly finding local paths. Note that if you receive a warning about the active manifest file is missing a Julia version or that dependencies may have been resolved with a different Julia version, there are two different actions. One is to use the same version of Julia as this repo (check Project.toml file) -- preferred to ensure reproducibility.  The other is to run Pkg.update() which will update all the packages -- exact reproducibility is no longer assured though general results should be similar. 

3. The measurement data used is available in a number of repositories. After cloning this repository, either manually step through the steps below  or run the **InstallData** script at the Julia prompt (assuming PsuedoNormRegression environment is active). To reiterate, this **InstallData** script will install all the data outlined below.  It will prompt at each dataset whether to download or skip. 
   ```
   $ cd path/to/this/project
   $ julia                                            <-- running in base env
   julia> using DrWatson
   julia> ]                                           <-- enter pkg mode
   (@v1.7) pkg> activate .                            <-- active repo env
     Activating project at `path/to/this/project`
   (PseudoNormRegression) pkg> <backspace>            <-- exit pkg mode
   julia> include(scriptsdir("InstallData.jl"))       <-- running in repo env
   ```

  - [JHU data](https://zenodo.org/record/6795580) [^1]
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ cd psuedonormregression/data/exp_raw
      $ curl -o JHU_fingerprints.csv https://zenodo.org/record/6795580/files/JHU_fingerprints.csv
      $ cd psuedonormregression
      $ julia
      julia> using DrWatson
      julia> @quickactivate "PseudoNormRegression"
      julia> include(scriptsdir("FormatData_JHU.jl")) 
      ```
  - [Zenodo_3968503 data](https://zenodo.org/record/3968503) [^2] 
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ cd psuedonormregression/data/exp_raw
      $ curl -o Zenodo_3968503.zip https://zenodo.org/record/3968503/files/Zenodo_3968503.zip
      $ unzip Zenodo_3968503.zip
      $ mv psuedonormregression/data/exp_raw/Zenodo_3968503/databases/\*.mat psuedonormregression/data/exp_pro 
      $ mv psuedonormregression/data/exp_pro/SIM*.mat pseudonormregression/data/sims/
      ```
  - [TUD data](https://data.4tu.nl/articles/dataset/Data_underlying_the_publication_A_Computationally_Efficient_Moving_Horizon_Estimator_for_UWB_Localization_on_Small_Quadrotors_/14827680) [^3] 
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ cd psuedonormregression/data/exp_raw/
      $ curl -o UWBdataset.zip https://data.4tu.nl/ndownloader/files/28631865
      $ unzip -d TUDdataset UWBdataset.zip
      $ mv TUDdataset/UWB\ dataset TUDdataset/UWBdataset
      $ cd psuedonormregression
      $ julia
      julia> using DrWatson
      julia> @quickactivate "PseudoNormRegression"
      julia> include(scriptsdir("FormatData_TUD.jl")) 
      ```
  - [UTIAS data](https://utiasdsl.github.io/util-uwb-dataset/) [^4]
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ cd psuedonormregression/data/exp_raw/
      $ curl -o dataset.7z -L https://github.com/utiasDSL/util-uwb-dataset/releases/download/Dataset-v1.0/dataset.7z
      $ 7zr x -oUTIAS_UWB_TDOA_DATASET dataset.7z 
      $ cd psuedonormregression
      $ julia
      julia> using DrWatson
      julia> @quickactivate "PseudoNormRegression"
      julia> include(scriptsdir("FormatData_UTIAS.jl")) 
      ```
  - [Circles in rectangles](http://www.packomania.com)
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ cd psuedonormregression/data/
      $ mkdir packing_coords/ 
      $ cd packing_coords/
      ```
      There are 8 files to pull.  Simply swap in a digit from 1 to 8 for the # symbol. There are three substitutions in the curl command!
      ```
      $ curl -o crc_0.#00000000000_coords.tar.gz -L http://hydra.nat.uni-magdeburg.de/packing/crc_#00/txt/crc_0.#00000000000_coords.tar.gz
      $ gunzip crc_0.#00000000000_coords.tar.gz
      $ tar -xvf crc_0.#00000000000_coords.tar
      ```

  - Simulation data: TDoA data,  based on JHU testbed, is generated with various error distributions (Laplace, Tukey-Lambda, Normal)
    - Open terminal, go to the root of the cloned package, then:
      ```
      $ julia
      julia> using DrWatson
      julia> @quickactivate "PseudoNormRegression"
      julia> include(scriptsdir("SimulationDataScript.jl") 
      ```
## Example Usage

Import packages and functions (copy/paste to julia REPL). Assumes DrWatson is
imported and PsuedoNormRegression environment is activated.

```
  #core functions - contains MLJ model constructs
  #and kernel functions
  using QuasinormRegression
  #utility sampling functions - if use sphere packing - requires files
  #from packomania.com.
  using Zenodo3968503Functions

  using MLJ
  using Printf, Logging, ProgressMeter, Dates
  using DataFrames, CSV
  using Plots
  pyplot();
  using LinearAlgebra
```
Read in data and format appropriately

```
  #files for input 
  base_name = "data/exp_raw/JHU_fingerprints"  #CSV file to load with measurements and associated parameters
  data_file = base_name*".csv"

  #load data
  D_orig = DataFrames.DataFrame(CSV.File(data_file));

  #pull out tdoa, rss and aoa measurements
  X_df=D_orig[:,r"tdoa|rss|aoa"]
  #get positions
  y_df=D_orig[:,r"x|y"]
  #y_tb=Tables.rowtable(y_df);

  #recover memory
  D_orig=0;
```


Set up some base regression models 

```
  base_models = [@load RidgeRegressor      pkg ="MLJLinearModels" verbosity=0; #1
                 @load LassoRegressor      pkg ="MLJLinearModels" verbosity=0; #2
                 @load ElasticNetRegressor pkg ="MLJLinearModels" verbosity=0; #3
                 @load KNNRegressor verbosity=0;                               #4
                 #@load KNeighborsRegressor pkg ="ScikitLearn"     verbosity=0;#5
                 ];
  mdl_number = 4;
```
### Single Kernel Regression
Construct, fit, and test Julia ML models.  For single kernel approach, all the
measurements are kernelized using one kernel. In the code below, "y_df" and
"X_df" are dataframes. The former is cartesian location information (x, y) and
the latter has a set of measurements associated with each location.  Data is
kernelized and then regressed to estimate location.

```
  t_start = now()
  #declare composite model including core model and kernel
  mlj_enr_x = SingleKernelRegressor(mdl= base_models[mdl_number]())
  mlj_enr_y = SingleKernelRegressor(mdl= base_models[mdl_number]());

  #set model λ_kern
  mlj_enr_x.λ_kern=mlj_enr_y.λ_kern=999.0  #999.0
  #set model p
  mlj_enr_x.p=mlj_enr_y.p=0.20

  #set lambda/gamma... can comment out and use default settings
  #set some default settings for various models
  if mdl_number == 1 || mdl_number == 2
      mlj_enr_x.mdl.lambda = 1e-1
      mlj_enr_y.mdl.lambda = 1e-1
  elseif mdl_number == 3
      mlj_enr_x.mdl.lambda = mlj_enr_y.mdl.lambda = 1e-1
      mlj_enr_x.mdl.gamma  = mlj_enr_y.mdl.gamma  = 1e-1
  elseif mdl_number == 4
      mlj_enr_x.mdl.K         = mlj_enr_y.mdl.K         = 1
      mlj_enr_x.mdl.leafsize  = mlj_enr_y.mdl.leafsize  = 10
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = :kdtree
  elseif mdl_number ==5
      mlj_enr_x.mdl.n_neighbors = mlj_enr_y.mdl.n_neighbors = 1           #default is 5
      mlj_enr_x.mdl.n_jobs = mlj_enr_y.mdl.n_jobs   = 6                   #default is 1
      mlj_enr_x.mdl.leaf_size = mlj_enr_y.mdl.leaf_size = 10              #default is 30
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = "kd_tree"       #default is auto
  else
      println("check model choice")
  end

  #re-split into test, train indices... getting indices for each row
  #train_idx, valid_idx, test_idx = partition(eachindex(y_df[:,1]), 0.1, 0.1, shuffle=true, rng=1235); #70:20:10 split, want 30% for testing
  train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true, rng=1235); #70:20:10 split

  #create machines
  opt_mc_x = machine(mlj_enr_x, X_df[train_idx,:], y_df[train_idx, :x])
  opt_mc_y = machine(mlj_enr_y, X_df[train_idx,:], y_df[train_idx, :y])

  #fit the machines
  MLJ.fit!(opt_mc_x, verbosity=1)
  MLJ.fit!(opt_mc_y, verbosity=1);

  #predict
  yhat_x = MLJ.predict(opt_mc_x, X_df[test_idx,:])
  yhat_y = MLJ.predict(opt_mc_y, X_df[test_idx,:]);

  #mean euclidean distance error
  current_mee   = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_std   = std(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_mee_x = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2)))
  current_mee_y = mean(sqrt.((yhat_y- y_df[test_idx, :y]).^(2)))

  #see results
  @printf("The avg error (%0.2fm) and std (%0.2fm) using model: %s against %s data\n", current_mee, current_std, typeof(mlj_enr_x.mdl), data_file)
  @printf("Avg error for x and y dimensions: %5.2f | %5.2f\n", current_mee_x, current_mee_y)
  @printf("Runtime: %s\n",(now()-t_start))
  @show mlj_enr_x.mdl
```

### Multi-Kernel Regression

Construct, fit, and test Julia ML models.  For multi-kernel approach, each set of 
measurements are kernelized independently. In the code below, "y_df" and
"X_df" are dataframes. The former is cartesian location information (x, y) and
the latter has a set of measurements {TDoA, RSS, AoA} associated with each location.  Data is
kernelized and then regressed to estimate location.

Note the example below is constructed as if there are two sets of measurements
(or feature types) hence two values of 'p', 'lambda_kern', and in the feature
count array (f_count) which declares the number of features for each
feature/measurement type.  

```
  t_start=now()

  #declare composite model including core model and kernel
  mlj_enr_x = MultipleKernelRegressor(mdl= base_models[mdl_number]())
  mlj_enr_y = MultipleKernelRegressor(mdl= base_models[mdl_number]());

  #set model p, λ_kern (use 999.0 flag), and feature count
  p        = [0.385 0.385]
  λ_kern   = [999.0 999.0]
  f_counts = [112 24]
  mlj_enr_x.p        = mlj_enr_y.p        = p
  mlj_enr_x.λ_kern   = mlj_enr_y.λ_kern   = λ_kern
  mlj_enr_x.f_counts = mlj_enr_y.f_counts = f_counts

  #set lambda/gamma... can comment out and use default settings
  #set some default settings for various models
  if mdl_number == 1 || mdl_number == 2
      mlj_enr_x.mdl.lambda = 1e-1
      mlj_enr_y.mdl.lambda = 1e-1
  elseif mdl_number == 3
      mlj_enr_x.mdl.lambda = mlj_enr_y.mdl.lambda = 1e-1
      mlj_enr_x.mdl.gamma  = mlj_enr_y.mdl.gamma  = 1e-1
  elseif mdl_number == 4
      mlj_enr_x.mdl.K         = mlj_enr_y.mdl.K           = 1
      mlj_enr_x.mdl.leafsize  = mlj_enr_y.mdl.leafsize    = 5
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm   = :kdtree
  elseif mdl_number == 5
      mlj_enr_x.mdl.n_neighbors = mlj_enr_y.mdl.n_neighbors = 1           #default is 5
      mlj_enr_x.mdl.n_jobs    = mlj_enr_y.mdl.n_jobs    = 6               #default is 1
      mlj_enr_x.mdl.leaf_size = mlj_enr_y.mdl.leaf_size = 10              #default is 30
      mlj_enr_x.mdl.algorithm = mlj_enr_y.mdl.algorithm = "kd_tree"       #default is auto
  else
      println("check model choice")
  end

  #re-split into test, train indices... getting indices for each row
  #train_idx, valid_idx, test_idx = partition(eachindex(y_df[:,1]), 0.6, 0.1, shuffle=true, rng=1235); #70:20:10 split, want 30% for testing
  train_idx, test_idx = partition(eachindex(y_df[:,1]), 0.7, shuffle=true)#, rng=1235); #70:20:10 split

  #create machines
  opt_mc_x = machine(mlj_enr_x, X_df[train_idx,:], y_df[train_idx, :x])
  opt_mc_y = machine(mlj_enr_y, X_df[train_idx,:], y_df[train_idx, :y])

  #fit the machines
  MLJ.fit!(opt_mc_x, verbosity=1)
  MLJ.fit!(opt_mc_y, verbosity=1);

  #predict
  yhat_x = MLJ.predict(opt_mc_x, X_df[test_idx,:])
  yhat_y = MLJ.predict(opt_mc_y, X_df[test_idx,:]);

  #mean euclidean distance error
  current_mee   = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_std   = std(sqrt.((yhat_x- y_df[test_idx, :x]).^(2) + (yhat_y - y_df[test_idx, :y]).^2))
  current_mee_x = mean(sqrt.((yhat_x- y_df[test_idx, :x]).^(2)))
  current_mee_y = mean(sqrt.((yhat_y- y_df[test_idx, :y]).^(2)))

  #see results
  @printf("The avg error (%0.2fm) and std (%0.2fm) using model: %s against %s data\n", current_mee, current_std, typeof(mlj_enr_x.mdl), data_file)
  @printf("Avg error for x and y dimensions: %5.2f | %5.2f\n", current_mee_x, current_mee_y)
  @printf("Runtime: %s\n",(now()-t_start))
  @show mlj_enr_x.mdl
```

### Optimal Training Spacing

To selectively chose a subset of measurement locations for training, there is a set of utility
functions to periodically choose from given measurements.  One approach is
sample base on given order of measurements.  Another approach via circle packing 
is to choose set number of measurements from locations with maximal
spacing.  Note that function assumes rectangular area when passing in
measurement locations.

First example is periodic sampling based on order of measurements:

```
  #periodic sampling based on order of sample locations
  train_idx = periodic_sampling(train_percentage, size(y_df)[1])
  _, test_idx = partition(setdiff(eachindex(y_df[:,1]), train_idx), 0.7, shuffle=true); #70:20:10 split want 30% for test, remove train_idx index

```

Second example is periodic sample based on optimal spacing.  First load
packing coordinates from set of files.  This is a one time operation.

```
  #loading of packing coordinates
  packing_df = DataFrames.DataFrame()
  file_location_packing = "./data/packing_coords/"
  #set of possible numbers that will be iterated/tested
  for train_percentage in  0.0025.*2.0.^(range(0,5,step=1))
      train_idx = Zenodo3968503Functions.periodic_sampling(train_percentage, size(y_df)[1])
      print(size(train_idx,1), " ")#," ", train_idx)
      
      idx = size(train_idx,1)
      #find file with number of circles tha match number of training points
      filename = filter!(s->occursin(r"crc"*string(idx)*"_0.6",s),readdir(file_location_packing))
      print(filename," ")
      temp_df = DataFrames.DataFrame(CSV.File(file_location_packing*filename[1], header=["grp", "x", "y"], delim=' ', ignorerepeated=true))
      temp_df[!, :grp]=temp_df[!, :grp].*0 .+ idx
      append!(packing_df, temp_df)
  end
  packing_df = DataFrames.groupby(packing_df, :grp);
```
Now that coordinates are loaded, choose optimal spacing of measurement locations given number of locations.  Here, number of locations is passed via percentage of total available measurement locations:

```
  #sampling based on optimized spacing (packing)
  train_idx = Zenodo3968503Functions.periodic_sampling(train_percentage, size(y_df)[1], y_df, packing_df, rand_float=1.0)
  _, test_idx = partition(setdiff(eachindex(y_df[:,1]), train_idx), 0.7, shuffle=true); #70:20:10 split want 30% for test, remove train_idx index

```

# Replicate Published Results
  The following assumes that Julia REPL is started, that DrWatson package is imported, and that the PsuedoNormRegression environment is activated. General processing time ranges from minutes to hours (on Intel® Core™ i7-9750H CPU @ 2.60GHz × 12) with exception of TUD dataset.
  - Run code for 2021 Asilomar testing and results [^1]. This code utilized
    pseudonorms to move non-linear, non-Gaussian features into a higher
    dimensional space via kernel trick. This code is focused on single and
    multiple kernel approaches. Technique provides strong results in
    improving regression results -- here for localization estimation. 
    - include(scriptsdir("LinearRegModule.jl")) <-- enter in "ParamsFile_2021asilomarSK.jl" 
    - include(scriptsdir("LinearRegModule.jl")) <-- enter in "ParamsFile_2021asilomarMK.jl"
  - Run code for 2022 Asilomar testing and results [^5]. This code is a
    departure form utilizing pseudonorm directly to address outliers.  Rather,
    the outliers are removed then standard localization algorithms are
    utilized.
    - include(scriptsdir("DenoiseModule.jl")) <-- enter in "ParamsFile_2022asilomarDenoise.jl"
    - include(scriptsdir("LinearRegModule.jl")) <-- enter in "ParamsFile_2022asilomar.jl"
    - include(scriptsdir("PaperFigureScript2022asilomar.jl"))  <-- generate figures and table
  - Run code for 2022 TBD journal (this is still in progress). This code is
    focused on the relationship between the choice of pseudonorms and the distribution of
    features (L2 -- Gaussian, L1 -- Laplacian, etc.).
    - include(scriptsdir("LinearRegModule.jl")) <-- enter in "ParamsFile_2022tbdJournal.jl"
    - include(scriptsdir("LinearRegModule.jl")) <-- enter in "ParamsFile_2022tbdJournal_TUDdata.jl"
      - Due to size of TUD dataset, processing time was around two weeks on Intel® Core™ i7-9750H CPU @ 2.60GHz × 12 
    - include(scriptsdir("LinearRegModuleLoop.jl")) <-- enter in "ParamsFile_2022tbdJournalSims.jl"
      - If you get 'scipy' not recognized, then do the following:
      ```
      julia> using Conda
      julia> Conda.add("scipy")
      ```
    - include(scriptsdir("PaperFiguresScriptTBDpaper.jl"))


# Errata
  - To speed up processing, set the number of Julia threads to utilize 75-90% of available cores. Its always good to leave one or two available! Two ways to set threads with the first method explicitly starting Julia with a given number of threads (see first line below).  The second sets the host system environment variable (here Linux) in .bashrc (see second line below).  For more complete information, see [Julia Multi-Threading Section](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading).
        ```
        $ julia --threads 9
        ```
        or
        ```
        export JULIA_NUM_THREADS = 9
        ```
  - If run into Matplotlib backend issue such as no GUI-compatible backend, please follow instructions at [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)

# References 
[^1]: B. Laird and T. Tran, "Quasi-norm Kernel-based Emitter Localization," 2021 55th Asilomar Conference on Signals, Systems, and Computers, 2021, pp. 534-538, doi: 10.1109/IEEECONF53345.2021.9723416.
[^2]: Torres-Sospedra, J.; Richter, P.; Moreira, A.; Mendoza-Silva, G.; Lohan, E.; Trilles, S.; Matey-Sanz, M. and Huerta, J. A Comprehensive and Reproducible Comparison of Clustering and Optimization Rules in Wi-Fi Fingerprinting IEEE Transactions on Mobile Computing, 2020  https://doi.org/10.1109/TMC.2020.3017176 
[^3]: S. Pfeiffer, C. d. Wagter and G. C. H. E. d. Croon, "A Computationally Efficient Moving Horizon Estimator for Ultra-Wideband Localization on Small Quadrotors," in IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 6725-6732, Oct. 2021, doi: 10.1109/LRA.2021.3095519.
[^4]: https://arxiv.org/pdf/2203.14471.pdf
[^5]: B. Laird and T. Tran, "Outlier Removal for Fingerprinting Localization Methods," Accepted for 2022 56th Asilomar Conference on Signals, Systems, and Computers

