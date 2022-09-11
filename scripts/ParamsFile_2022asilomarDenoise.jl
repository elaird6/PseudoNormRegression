#used with DenoiseModule.jl
#
#params file that is modified as needed to run different experiments
#there are dependencies in setting up params (mainly keyword structs)

using DrWatson
@quickactivate "PseudoNormRegression"


#explicit dictionary structure to avoid error creating dictionary with
#@strdict keyword method (forget to modify if add/remove parameters)

params = Dict(
  "dbpath" => datadir("exp_pro"),
  "db" => ["JHUtdoa.mat"],
  "prefix" => "DenoiseELN",  #prefix to append to "db" name once denoised
#  "λ" => 10.0.^collect(range(-4,stop=-2, step=1)),
  "λ" => 10.0.^collect(range(-4,stop=-2, step=1)),
#  "γ" => 10.0.^collect(range(-0,stop= 2, step=1))
  "γ" => 10.0.^collect(range(-1,stop= 1, step=1))
 )


###############################################################
## need to do some basic parameter compatibility checking here...
##


@info "Using this process: `denoise` on these files: $(params["db"])..."

param_combinations = dict_list(params)
