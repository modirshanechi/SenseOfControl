module EmpHCA

using ConcreteStructs

using PyPlot; 
import StatsPlots
import StatsBase: countmap, summarystats

using DataFrames
using CSV, JLD2, JSON3

using Random, Statistics, Distributions
using Turing, MCMCChains
using MCMCDiagnosticTools: mcse

using LinearAlgebra, Zygote
using NNlib: softmax

using JuMP
import Clarabel
import MathOptInterface as MOI


const MainColors = (; data = "#415a77", 
                    lcol = ["#E33128","k","#3788C1"],
                    GB = ["#FFD983","#DE3045","#e09f3e"])
export MainColors
const ModelNames = (; MEmp = ["L < 1","L = 1","L > 1"], 
                MAll = ["Random","N-Act","Emp-l","General"])
export ModelNames

include("Functions_empell.jl")
include("Functions_Klyemp.jl")

include("Functions_experiments.jl")
include("Functions_goldagent.jl")

include("Functions_TuringModels.jl")
include("Functions_bridgesampling.jl")
include("Functions_BIC.jl")
include("Functions_MCMC_RandEffects.jl")

include("Functions_general.jl")
include("Functions_plotting.jl")


end # module EmpHCA


