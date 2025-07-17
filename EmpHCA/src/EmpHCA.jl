module EmpHCA

using ConcreteStructs
using PyPlot; import StatsPlots
using DataFrames
using Random, Statistics, Turing, MCMCChains, Distributions
using Zygote
using LinearAlgebra
using NNlib: softmax

const MainColors = (; data = "#415a77", 
                    lcol = ["#E33128","k","#3788C1"],
                    GB = ["#FFD983","#DE3045","#e09f3e"])
export MainColors
const ModelNames = (;   MEmp = ["L < 1","L = 1","L > 1"], 
        MAllE1 = ["Random","N-Act","Emp-l","General"], 
        MAllE2 = ["Rand","Na","Na(G)-Emp(B)","Emp(G)-Na(B)","E-E","General"])
export ModelNames

include("Functions_for_emp.jl")
include("Functions_for_emp_K.jl")
include("Functions_for_gold.jl")
include("Functions_for_goldDataE1.jl")
include("Functions_for_goldDataE2.jl")
include("Functions_general.jl")
include("Functions_for_plotting.jl")
include("Functions_MCMC_RandEffects.jl")

end # module EmpHCA


