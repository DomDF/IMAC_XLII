#=
  "Risk Optimal Material Test Planning using Value of Information Analysis"
   SEM IMAC XLII Conference, 2024
   
   Accompanying Julia code to set-up and run calculations

   Domenic Di Francesco, PhD, CEng (MIMechE)
   The Alan Turing Institute, University of Cambridge
   
=#

######################################################
#
# Loading libraries
#
######################################################

# For describing probabilistic models
using Distributions, Turing, Random, LatinHypercubeSampling, Copulas
# For describing and solving decision problem
using JuMP, HiGHS, Gurobi, DecisionProgramming, LinearAlgebra
# For working with data
using CSV, DataFrames, DataFramesMeta

Gurobi.Env()
