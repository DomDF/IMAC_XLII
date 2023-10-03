#=
  "Risk Optimal Material Test Planning using Value of Information Analysis"
   SEM IMAC XLII Conference, 2024
   
   Accompanying Julia code to set-up and run calculations

   Domenic Di Francesco, PhD, CEng (MIMechE)
   The Alan Turing Institute, University of Cambridge

   -_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_-
BS 7910 

When direct determination of fracture toughness is not possible, estimates may
be made using Charpy correlations as described in Annex J. Alternatively, generic
data (e.g. obtained from literature) may be used provided that they are
demonstrated to be fully representative of the material used in the structure
being assessed and that the fracture toughness chosen represents a lower bound
to the data.

7.1.7
If the data meet the criteria in 7.1.5.3 with the exception of scatter
limits based on three tests, the minimum of three equivalent (MOTE) may be
employed

3-5   => lowest
6-10  => second-lowest
11-15 => third-lowest

C.2.2
Charpy impact data might be available
from the original construction records, in which case the correlations given in
Annex J may be used. Alternatively, cut-out material may be used or a
conservative lower bound established from the literature for similar steel and
weldments adopted.

J.2.5
To avoid overestimating fracture toughness at the service temperature in
materials with potentially low upper shelf Charpy energy, Kmat (estimated in
accordance with J.2.1 and J.2.2) should not exceed the value given by
Equation (J.10).

Kmat(MPa.m^1/2) = 0.54 CVN(J) + 55

-_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_--_-_-_-_-_-_-_-_-
https://doi.org/10.1016/j.ijpvp.2018.10.016

Annex J (‘Use of Charpy V-notch impact tests to estimate fracture toughness’) describes a 
number of approaches to estimating a lower-bound value of Kmat for ferritic steels, depending 
on whether the Charpy tests were carried out on the lower shelf, in the lower transition, or 
on the upper shelf, and the temperature for which Kmat is required. The use of sub-size Charpy 
data (ie specimens with thickness <10 mm) is also described and a flowchart included to guide 
the first-time user in particular. The equations are largely taken from work undertaken in the 
SINTAP and FITNET projects [19].

=#


cd("/Users/ddifrancesco/Github/IMAC_XLII")
using Pkg; Pkg.activate(".") # create a new virtual environment

StanSample.set_cmdstan_home!("/Users/ddifrancesco/.cmdstan/cmdstan-2.33.0")
stan_files = filter(x -> contains(x, ".stan"), readdir())

ksi_to_MPa = 6.89476; in_to_m = 0.0254; ftlbs_to_J = 1.35581795

######################################################
#
# loading libraries
#
######################################################

# For describing probabilistic models
using Distributions, StanSample, MCMCChains, Random, LatinHypercubeSampling
# For describing and solving decision problem
using JuMP, Gurobi, DecisionProgramming, LinearAlgebra
# For working with data
using CSV, DataFrames, DataFramesMeta, UnicodePlots

######################################################
#
# defining functions
#
######################################################

function draw_lhs(dist, n::Int; reprod::Int = 240819)
  Random.seed!(reprod)
  samples = randomLHC(n + 2, 1) |>
      lhc -> scaleLHC(lhc, [(0, 1)]) |>
      lhc -> quantile(dist, lhc)[:,1] |>
      q -> filter(!∈((-Inf, Inf)), q) |>
      q -> [q[i] for i ∈ 1:length(q) if abs(q[i]) >= 10^-10]
  return samples
end

function get_LogNorm_params(μ::Float64, σ::Float64)
  log_sd = √(log(1 + σ^2 / μ^2))
  log_mean = log(μ) - 1/2 * log_sd^2
  return (log_μ = log_mean, log_σ = log_sd)
end

function run_Stan_model(data_dict::Dict, name::String)
  
  model_text = open(name * ".stan") do file
    read(file, String)
  end
  
  tmpdir = pwd() * "/tmp"; model = SampleModel(name, model_text, tmpdir)

  rc = stan_sample(model; 
       data = data_dict, 
       num_warmups = 10_000, num_samples = n_mcmc ÷ n_chains, num_threads = 1, 
       num_chains = n_chains, use_cpp_chains = true, save_warmup = false)

  if success(rc)
      posterior_df = read_samples(model, :mcmcchains) |> x -> DataFrame(x)
      diags_df = read_summary(model) |> x -> DataFrame(x)
  else
      @warn "sampling failed"
  end

  return(posterior_df, diags_df)

end

######################################################
#
# define a prior model for Kmat, based on a small number of measurements
#
######################################################

K_mat_true = Normal(150, 10)

n_meas = 3; meas_error = 5; n_mcmc = 1_000; n_chains = 4

K_mat_samples = Xoshiro(240819) |> 
  prng -> rand(prng, K_mat_true, 100)

Kmat_post_df, Kmat_diags_df = run_Stan_model(
  Dict("N" => n_meas, "test_data" => K_mat_samples[1:n_meas], "epsilon" => meas_error), 
  "Kmat")

UnicodePlots.histogram(Kmat_post_df.Kmat_pred, nbins = 20)

#=
Define the prior decision parameters
=#

a_ln = get_LogNorm_params(4.0, 2/3)

Y = 1.12  # geometry factor
a = LogNormal(a_ln.log_μ, a_ln.log_σ)   # crack size, mm
aᵢ = draw_lhs(a, n_mcmc, reprod = 240819)
σ = 30 * ones(n_mcmc)    # applied stress, MPa

Kᵢ = Y .* σ .* .√(π .* aᵢ)
UnicodePlots.histogram(Kᵢ)

function pr_fail(K_mat::Vector{Float64}, a::Vector{Float64}, σ::Vector{Float64}; 
                 repair::Bool = false, reduce::Bool = false)
  @assert length(K_mat) == length(a) == length(σ) "Inputs must be of equal length"
  
  if repair == true
    a = get_LogNorm_params(1.0, 2/3) |> 
          x -> LogNormal(x.log_μ, x.log_σ) |> 
          x -> draw_lhs(x, n_mcmc, reprod = 240819)
  end

  if reduce == true
    σ = 20 .* ones(length(σ))
  end
  
  Kᵢ = Y .* σ .* .√(π .* a)

  return sum(K_mat .< Kᵢ) / length(K_mat)
end

pr_fail(Kmat_post_df.Kmat_pred, aᵢ, σ)

site_visit = 0.005; cost_repair = 0.015; cost_reduce = 0.025

maint_opts = Dict("no_action" => 0, 
                  "repair" => cost_repair + site_visit, 
                  "reduce_operation" => cost_reduce, 
                  "repair_reduce" => cost_repair + cost_reduce + site_visit)

maint_states = keys(maint_opts) |> x -> collect(x)

β_states = ["fail", "survive"]; CoFs = [1, 0]

function solve_id(;K_mat::Vector{Float64} = Kmat_post_df.Kmat_pred)

  # Initialise influence diagram and add nodes
  SIM = InfluenceDiagram()

  add_node!(SIM, DecisionNode("maint", [], maint_states))
  add_node!(SIM, ChanceNode("β", ["maint"], β_states))
  add_node!(SIM, ValueNode("CoF", ["β"]))
  add_node!(SIM, ValueNode("C_maint", ["maint"]))

  generate_arcs!(SIM)

  # Populate structural reliability node(s) with PoFs
  β = ProbabilityMatrix(SIM, "β")
  
  β["no_action", :] = pr_fail(K_mat, aᵢ, σ) |> x -> [x, 1 - x]
  β["repair", :] = pr_fail(K_mat, aᵢ, σ, repair = true) |> x -> [x, 1 - x]
  β["reduce_operation", :] = pr_fail(K_mat, aᵢ, σ, reduce = true) |> x -> [x, 1 - x]
  β["repair_reduce", :] = pr_fail(K_mat, aᵢ, σ, repair = true, reduce = true) |> x -> [x, 1 - x]

  # Populate maintenance cost node(s)
  Cₘ = UtilityMatrix(SIM, "C_maint")
  for i ∈ maint_states
      Cₘ[i] = maint_opts[i]
  end

  # Populate failre cost node(s)
  Cᵣ = UtilityMatrix(SIM, "CoF")
  Cᵣ[:] = CoFs

  add_probabilities!(SIM, "β", β)
  add_utilities!(SIM, "C_maint", Cₘ)
  add_utilities!(SIM, "CoF", Cᵣ)

  generate_diagram!(SIM)

  # Define JuMP model with solver using all available threads
  SIM_model = Model(); set_silent(SIM_model)
  set_optimizer(SIM_model, Gurobi.Optimizer)
  set_optimizer_attribute(SIM_model, "threads", Threads.nthreads())

  # Define decision variables and expected utility for optimisation
  z = DecisionVariables(SIM_model, SIM)
  EC = expected_value(SIM_model, SIM, 
                      PathCompatibilityVariables(SIM_model, SIM, z))

  @objective(SIM_model, Min, EC)
  optimize!(SIM_model)

  # Extract a* and u* from the solution
  Z = DecisionStrategy(z)
  U_dist = UtilityDistribution(SIM, Z)
  
  # Return results as a dataframe
  opt_df = DataFrame(a_opt = maint_states[argmax(Z.Z_d[1])],
                     u_opt = LinearAlgebra.dot(U_dist.p, U_dist.u))

  return(opt_df)

end

prior_decision = solve_id()

VoPI_df = DataFrame(a_opt = String[], u_opt = Float64[], Kmat_meas = Float64[])
for Kmat ∈ Kmat_post_df.Kmat_pred
  append!(VoPI_df,
          @rtransform(solve_id(K_mat = Kmat * ones(n_mcmc)), :Kmat_meas = Kmat))
end

combine(groupby(VoPI_df, :a_opt), nrow => :count)

using Statistics
VoPI = prior_decision.u_opt[1] - (VoPI_df.u_opt |> Statistics.mean)

VoI_df = DataFrame(a_opt = String[], u_opt = Float64[], Kmat_meas = Float64[])
for i ∈ 1:length(Kmat_post_df.Kmat_pred)

  Kmat_I_df, Kmat_Idiags_df = run_Stan_model(
    Dict("N" => n_meas + 1, "test_data" => vcat(K_mat_samples[1:n_meas], Kmat_post_df.Kmat_pred[i]), "epsilon" => meas_error), 
    "Kmat")

  append!(VoI_df,
          @rtransform(solve_id(K_mat = Kmat_I_df.Kmat_pred), :Kmat_meas = Kmat_post_df.Kmat_pred[i]))
end

using Statistics
VoI = prior_decision.u_opt[1] - (VoI_df.u_opt |> Statistics.mean)

CSV.write("VoPI_df.csv", VoPI_df)
CSV.write("prior_decision.csv", prior_decision)

# A breech is an opening in a gun where bullets are loaded. 
# Cabot ESR, M68 gun tubes rotary forged breech

CVN_ft_lbs = [30, 31, 23, 24, 31, 32, 24, 25, 30, 29, 26, 25, 27, 22, 23, 23, 35, 36, 39, 38]
σY_ksi = [172, 171, 172, 172, 176, 176, 172, 174, 171, 175, 176, 175, 170, 170, 175, 177, 177, 176, 169, 167]
Kmat_ksi_in = [143, 131, 114, 120, 142, 140, 118, 119, 136, 146, 119, 117, 120, 117, 115, 113, 146, 143, 150, 151]

CVN_J = CVN_ft_lbs .* 1.35581795
Kmat_MPa_m = Kmat_ksi_in .* 6.89476 .* sqrt(0.0254)

0.54 .* CVN_J .+ 55
20 * ((12 .* .√CVN_J .-20) * (25/1))

######################################################################
#
# Charpy correlations
#
######################################################################

# Barson Rolfe upper-shelf correlation
function Barson_Rolfe(CVN_ft_lbs::Vector{Float64} = Float64.(CVN_ft_lbs), 
                      σY_ksi::Vector{Float64} = Float64.(σY_ksi))
  return 5 .* CVN_ft_lbs .* σY_ksi .- (σY_ksi.^2)./4 |> x -> .√x
end

Barson_Rolfe() - Kmat_ksi_in

Barson_Rolfe_model_text = open("Barson_Rolfe.stan") do file
    read(file, String)
end

BR_posterior_df, BR_diags_df = run_Stan_model(
  Dict("N" => length(CVN_ft_lbs), "CVN_ft_lbs" => CVN_ft_lbs, 
                 "sigmaY_ksi" => σY_ksi, "Kmat_ksi_in" => Kmat_ksi_in,
                 "sigmaY_ksi_pred" => 170, "CVN_ft_lbs_pred" => 30), 
  "Barson_Rolfe")

CSV.write("BR_posterior_df.csv", BR_posterior_df)

UnicodePlots.histogram(BR_posterior_df.sigma, nbins = 30)

# BS7910 upper-shelf correlation
function BS7910_US(CVN_J::Vector{Float64} = Float64.(CVN_J)) 
  return 0.54 .* CVN_J .+ 55
end

BS7910_post_df, BS7910_diags_df = run_Stan_model(
  Dict("N" => length(CVN_ft_lbs), "CVN_J" => CVN_J, 
                 "Kmat_MPa_m" => Kmat_MPa_m, "CVN_J_pred" => 30), 
  "BS7910_us")

UnicodePlots.histogram(BS7910_post_df.sigma, nbins = 30)

CSV.write("BS7910_posterior_df.csv", BS7910_post_df)


# go from Kmat to CVN_J
CVN_J_pred = Normal(30, 4) |> x -> draw_lhs(x, 100)

Barson_Rolfe_df = DataFrame(a_opt = String[], u_opt = Float64[], CVN_J_meas = Float64[])

for CVN_pp ∈ CVN_J_pred

  rc = stan_sample(Barson_Rolfe_model; 
                   data = Dict("N" => length(CVN_ft_lbs), "CVN_ft_lbs" => CVN_ft_lbs, 
                               "sigmaY_ksi" => σY_ksi, "Kmat_ksi_in" => Kmat_ksi_in,
                               "sigmaY_ksi_pred" => 170, "CVN_ft_lbs_pred" => CVN_pp / ftlbs_to_J), 
                   num_warmups = n_mcmc, num_samples = n_mcmc ÷ n_chains, num_threads = 1, 
                   num_chains = n_chains, use_cpp_chains = true, save_warmup = false)

  if success(rc)
    posterior_df = read_samples(Barson_Rolfe_model, :mcmcchains) |> x -> DataFrame(x)
  else
    @warn "sampling failed"
  end

  UnicodePlots.histogram(posterior_df.Kmat_pred_ksi .* ksi_to_MPa .* √in_to_m, nbins = 10) |> 
    display

  append!(Barson_Rolfe_df,
          @rtransform(solve_id(K_mat = posterior_df.Kmat_pred_ksi .* ksi_to_MPa .* √in_to_m), 
                      :CVN_J_meas = CVN_pp))
end



