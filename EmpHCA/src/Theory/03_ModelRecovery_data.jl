################################################################################
# Code for performing the model recovery analysis for different ranges of l;
# The results will be saved in the end and can be plotted via
#     src/Theory/04_ModelRecovery_plot.jl
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

SavePath = "src/Theory/Figures/Recovery/"

# ----------------------------------------------------------------------
# inference information
# ----------------------------------------------------------------------
n_chains = 1
chain_length = 1500
burn_in_length = 500
thinning = 1

K = 1

# ------------------------------------------------------------------------------
# looping over all settings
# ------------------------------------------------------------------------------
for ifallroom = [true, false]
      for β = [10., 2., 1.]
            rng = Xoshiro(2024)
            
            # ------------------------------------------------------------------
            # room selection
            # ------------------------------------------------------------------
            Prooms, ΔState, ΔStateDict = gold_proom_sets()
            if ifallroom      
                  N_rooms = length(Prooms); Ymax = 1; Xmax = 1
                  nametag = ""
            else
                  Prooms = Prooms[valid_rooms_E23]
                  N_rooms = length(Prooms); Ymax = 1; Xmax = 1
                  nametag = "_chosenrooms"
            end

            # ------------------------------------------------------------------
            # simulation
            # ------------------------------------------------------------------
            # agent parameters
            Xinds = vcat([[[i,j] for j = (1:N_rooms)[(1:N_rooms) .!= i]] 
                                                      for i = 1:N_rooms]...)
            Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax, K)

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
                  push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, β; 
                                                                        rng = rng))
            end


            # ------------------------------------------------------------------
            # Turing inference: param
            # ------------------------------------------------------------------
            m_hat  = zeros(N_agent)
            l_hat  = zeros(N_agent)
            logl_hat  = zeros(N_agent)
            for i = 1:N_agent
                  as = A_agent[i]
                  # model inference
                  # ------------------------------------------------------------
                  # unconstrained l-inference
                  # ------------------------------------------------------------
                  FITresultsEmpl = EmpL_fit(rng, Xinds, as,
                              Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                              n_chains, chain_length, burn_in_length, thinning)

                  # ------------------------------------------------------------
                  # model selection: l0, l1, l2
                  # ------------------------------------------------------------
                  MSresultsEmp = Emp_model_selection(rng, Xinds, as,
                              Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                              n_chains, chain_length, burn_in_length, thinning)


                  # read-out of the model
                  chn_df = MSresultsEmp.chnemp_df
                  m_hat[i] = findmax([sum(chn_df.m .== j) for j = 1:3])[2]  

                  # read-out of parameters
                  chnemp_df = FITresultsEmpl.chnemp_df
                  l_hat[i] = mean(exp.(chnemp_df.logl))
                  logl_hat[i] = mean(chnemp_df.logl)

                  # print
                  println("----------------------------")
                  @show i
                  @show m_true[i]
                  @show m_hat[i]
                  @show l_true[i]
                  @show l_hat[i]
                  @show log(l_true[i])
                  @show logl_hat[i]
            end

            save(SavePath * "RecSingSub_B" * 
                  string(β) * "_K" * string(K) * nametag * ".jld2",
                  "A_agent", A_agent,
                  "m_true", m_true, "l_true", l_true, "γ_true", γ_true, "β", β, "K", K,
                  "m_hat",  m_hat,  "l_hat",  l_hat,  "logl_hat", logl_hat)
      end
end