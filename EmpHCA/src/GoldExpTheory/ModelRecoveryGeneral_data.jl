################################################################################
# Code for performing the model recovery analysis for the 4 main models;
# The results will be saved in the end and can be plotted via 
#     src/GoldExpTheory/ModelRecoveryGeneral_plot.jl
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

Path_Save = "src/GoldExpTheory/Figures/Recovery/"

# ------------------------------------------------------------------------------
# possible rooms
# ------------------------------------------------------------------------------
ifallroom = true
valid_rooms = [0,1,2,4,5,6,7,8,9] .+ 1
for K = [1] 
for β = [2., 1.]
      seed = 2024; rng = MersenneTwister(seed)
      
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
      Xinds = vcat([[[i,j] for j = (1:N_rooms)[(1:N_rooms) .!= i]] 
                                                for i = 1:N_rooms]...)
      Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax, K)
      Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]

      model_set = ["Random","N-Act","Emp-l","General"]
      N_agent = 4 * 30
      m_true = Int64.(vcat([i .* ones(Int(N_agent / 4)) for i = 1:4]...))
      l_true = zeros(N_agent)
      γ_true = zeros(N_agent)
      θ_true = zeros(N_agent, N_rooms)
      A_agent = Vector{Vector{Int64}}([])
      for i = 1:N_agent
            ml_temp = rand(rng, 1:3)
            if ml_temp == 1
                  l_true[i] = rand(rng)
            elseif ml_temp == 2
                  l_true[i] = 1
            elseif ml_temp == 3
                  l_true[i] = 1 + 2*rand(rng)
            end
            γ_true[i] = 0.5 + 0.5 * rand(rng)
            θ_true[i,:] = randn(N_rooms) .* β

            if m_true[i] == 1       # random: β = 0
                  @time push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, 0; rng = rng))
            elseif m_true[i] == 2   # N-act
                  @time push!(A_agent, gold_simulate_Nact(Nas, β; rng = rng))
            elseif m_true[i] == 3   # Emp-l
                  @time push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, β; rng = rng))
            elseif m_true[i] == 4   # General
                  @time push!(A_agent, gold_simulate_General(Xinds, θ_true[i,:]; rng = rng))
            end
      end


      # ------------------------------------------------------------------------------
      # Turing inference: param
      # ------------------------------------------------------------------------------
      chn_set = []
      m_hat  = zeros(N_agent)
      l_hat  = zeros(N_agent)
      β_hat  = zeros(N_agent)
      βa_hat = zeros(N_agent)
      γ_hat  = zeros(N_agent)
      θ_hat  = zeros(N_agent,N_rooms)
      for i = 1:N_agent
            as = A_agent[i]
            # inference
            model = TuringGoldBasicInfvsEmplvsNa(Xs, Xinds, Nas, as; 
                                                N_rooms=N_rooms, K=K)
            g = Gibbs(HMC(0.01, 50, :θ, :l, :β, :βa, :βθ), MH(:m, :γ))
            chn = sample(model, g, MCMCThreads(), 1100, 3)
            # StatsPlots.plot(chn; legend=true)
            push!(chn_set,chn)

            # read-out of parameters
            chn_df = DataFrame(chn)
            θs = Array(chn_df[:,["θ[" * string(i) * "]" for i = 1:N_rooms]]); θs[:,1] .= 0
            filter!(row -> row.iteration > 100, chn_df)
            chn_df = chn_df[1:10:size(chn_df)[1],:]
            m_hat[i]  = findmax([sum(chn_df.m .== j) for j = 1:4])[2]
            l_hat[i]  = mean(chn_df.l)
            β_hat[i]  = mean(chn_df.β)
            βa_hat[i] = mean(chn_df.βa)
            γ_hat[i]  = mean(chn_df.γ)
            θ_hat[i,:] = mean(θs, dims = 1)[:]

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

      save(Path_Save * "GeneralRecSingSub_B" * string(β) * "_K" * string(K) * nametag * ".jld2",
            "A_agent", A_agent, "chn_set", chn_set,
            "m_true", m_true, "l_true", l_true, "γ_true", γ_true, "β", β, "K", K,
            "m_hat",  m_hat,  "l_hat",  l_hat,  "γ_hat",  γ_hat,  "β_hat", β_hat,
            "βa_hat", βa_hat, "θ_hat", θ_hat, "θ_true", θ_true)
end
end