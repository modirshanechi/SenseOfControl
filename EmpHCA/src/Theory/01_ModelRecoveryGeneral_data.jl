################################################################################
# Code for performing the model recovery analysis for the 4 main models;
# The results will be saved in the end and can be plotted via
#     src/Theory/02_ModelRecoveryGeneral_plot.jl
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
                  i_exp = 1
            else
                  Prooms = Prooms[valid_rooms_E23]
                  N_rooms = length(Prooms); Ymax = 1; Xmax = 1
                  nametag = "_chosenrooms"
                  i_exp = 2
            end

            # ------------------------------------------------------------------
            # simulation
            # ------------------------------------------------------------------
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
                        push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, 0; rng = rng))
                  elseif m_true[i] == 2   # N-act
                        push!(A_agent, gold_simulate_Nact(Nas, β; rng = rng))
                  elseif m_true[i] == 3   # Emp-l
                        push!(A_agent, gold_simulate(Xs, l_true[i], γ_true[i], K, β; rng = rng))
                  elseif m_true[i] == 4   # General
                        push!(A_agent, gold_simulate_General(Xinds, θ_true[i,:]; rng = rng))
                  end
            end


            # ------------------------------------------------------------------
            # Turing inference: param
            # ------------------------------------------------------------------
            m_hat  = zeros(N_agent)
            logL_mat = zeros(N_agent, 4)
            for i = 1:N_agent
                  as = A_agent[i]

                  # ------------------------------------------------------------
                  # BradleyTerry inference
                  # ------------------------------------------------------------
                  FITresultsBD = BradTerry_fit(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                  # ------------------------------------------------------------
                  # unconstrained l-inference
                  # ------------------------------------------------------------
                  FITresultsEmpl = EmpL_fit(rng, Xinds, as,
                              Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                              n_chains, chain_length, burn_in_length, thinning)

                  # ------------------------------------------------------------
                  # a-inference
                  # ------------------------------------------------------------
                  FITresultsNa = Na_fit(rng, Xinds, as,
                              Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                              n_chains, chain_length, burn_in_length, thinning)

                  # ------------------------------------------------------------
                  # BIC
                  # ------------------------------------------------------------
                  infdata = Dict("FITresultsBD" => FITresultsBD, 
                                 "FITresultsEmpl" => FITresultsEmpl,
                                 "FITresultsNa" => FITresultsNa)
                  
                  logL_mat[i,:] = [bic_evaluation(m, infdata, Xinds, as,
                                    Prooms, ΔState, ΔStateDict, N_rooms, 
                                    Ymax, Xmax, i_exp).BIC
                                                for m = modelspecs_BIC]

                  # read-out of parameters
                  m_hat[i]  = findmax(logL_mat[i,:])[2]

                  # print
                  println("----------------------------")
                  @show i
                  @show m_true[i]
                  @show m_hat[i]
                  @show logL_mat[i,:]
            end

            save(SavePath * "GeneralRecSingSub_B" * string(β) * "_K" * string(K) * nametag * ".jld2",
                  "A_agent", A_agent, 
                  "θ_true", θ_true, "m_true", m_true, "l_true", l_true, "γ_true", γ_true, "β", β, "K", K,
                  "m_hat",  m_hat
                  )
      end
end