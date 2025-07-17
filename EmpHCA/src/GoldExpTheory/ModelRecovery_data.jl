################################################################################
# Code for performing the model recovery analysis for different ranges of l;
# The results will be saved in the end and can be plotted via 
#     src/GoldExpTheory/ModelRecovery_plot.jl
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random
using Turing, MCMCChains, Distributions
using DataFrames
using JLD2
using AdvancedMH

import StatsPlots


PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExpTheory/Figures/Recovery/"

# ------------------------------------------------------------------------------
# possible rooms
# ------------------------------------------------------------------------------
ifallroom = false
valid_rooms = [0,1,2,4,5,6,7,8,9] .+ 1
for K = [1] 
for β = [10., 2., 1.]
      seed = 2024; rng = MersenneTwister(seed)
      
      Ymax = 1; Xmax = 1
      Prooms, ΔState, ΔStateDict = gold_proom_sets()
      if ifallroom      
            N_rooms = length(Prooms); Ymax = 1; Xmax = 1
            nametag = ""
      else
            Prooms = Prooms[valid_rooms]
            N_rooms = length(Prooms); Ymax = 1; Xmax = 1
            nametag = "_chosenrooms"
      end

      # agent parameters
      Xs = gold_Room2X_basic(Prooms, ΔState, ΔStateDict, Xmax, Ymax, K);

      N_agent = 3 * 30
      m_true = Int64.(vcat([i .* ones(Int(N_agent / 3)) for i = 1:3]...))
      l_true = zeros(N_agent)
      γ_true = zeros(N_agent)
      A_agent = Vector{Vector{Int64}}([])
      for i = 1:N_agent
            if m_true[i] == 1
                  l_true[i] = rand(rng)
            elseif m_true[i] == 2
                  l_true[i] = 1
            elseif m_true[i] == 3
                  l_true[i] = 1 + 2*rand(rng)
            end
            γ_true[i] = 0.5 + 0.5 * rand(rng)
            @time push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, β; rng = rng))
      end


      # ------------------------------------------------------------------------------
      # Turing inference: param
      # ------------------------------------------------------------------------------
      chn_set = []
      m_hat  = zeros(N_agent)
      l_hat  = zeros(N_agent)
      β_hat = zeros(N_agent)
      γ_hat = zeros(N_agent)
      for i = 1:N_agent
            as = A_agent[i]
            # inference
            model = TuringGoldMSel(Xs, as, K=K); 
            g = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:m, :γ))
            chn = sample(model, g, 1500);
            push!(chn_set,chn)

            # read-out of parameters
            chn_df = DataFrame(chn)
            filter!(row -> row.iteration > 100, chn_df)
            chn_df = chn_df[1:10:size(chn_df)[1],:]
            m_hat[i] = findmax([sum(chn_df.m .== j) for j = 1:3])[2]
            if m_hat[i] == 1
                  l_hat[i] = mean(chn_df.l0)
            elseif m_hat[i] == 2
                  l_hat[i] = 1
            elseif m_hat[i] == 3
                  l_hat[i] = mean(chn_df.l2)
            end
            β_hat[i] = mean(chn_df.β)
            γ_hat[i] = mean(chn_df.γ)

            # print
            println("----------------------------")
            @show i
            @show m_true[i]
            @show m_hat[i]
            @show l_true[i]
            @show l_hat[i]
            @show γ_true[i]
            @show γ_hat[i]
      end

      save(Path_Save * "RecSingSub_B" * string(β) * "_K" * string(K) * nametag * ".jld2",
            "A_agent", A_agent, "chn_set", chn_set,
            "m_true", m_true, "l_true", l_true, "γ_true", γ_true, "β", β, "K", K,
            "m_hat",  m_hat,  "l_hat",  l_hat,  "γ_hat",  γ_hat,  "β_hat", β_hat)
end
end