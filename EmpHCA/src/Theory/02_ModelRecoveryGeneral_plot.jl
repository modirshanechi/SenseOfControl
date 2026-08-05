################################################################################
# Plotting the results of the model recovery analysis for the 4 main models;
# See src/Theory/01_ModelRecoveryGeneral_data.jl
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

SavePath = "src/Theory/Figures/Recovery/"

# ------------------------------------------------------------------------------
# all-rooms vs. chosen-rooms toggle
# ------------------------------------------------------------------------------
for ifallroom = [true, false]
      if ifallroom      
            nametag = ""
      else
            nametag = "_chosenrooms"
      end

      K = 1
      for β = [10., 2., 1.]
            temp = load(SavePath * "GeneralRecSingSub_B" * string(β) * "_K" * string(K) * nametag * ".jld2")
            A_agent = temp["A_agent"]
            m_true = temp["m_true"]
            m_hat = temp["m_hat"]

            # ------------------------------------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------------------------------------
            Legends = ["Random","N-Act","Emp-l","General"]; N_model = length(Legends)
            ConfMat = zeros(N_model,N_model)
            for i = 1:N_model
                  for j = 1:N_model
                        ConfMat[i,j] = sum(m_hat[m_true .== i] .== j)
                  end
                  ConfMat[i,:] .= ConfMat[i,:] ./ sum(ConfMat[i,:])
            end

            fig = figure(figsize=(6,5))
            Y = ConfMat
            ax = subplot(1,1,1)
            cp = ax.imshow(Y,vmin=0,vmax=1.0,cmap="Blues")
            for i = 1:N_model
                  for j = 1:N_model
                        ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                                    horizontalalignment="center")
                  end
            end
            fig.colorbar(cp, ax=ax)
            ax.set_xticks(0:(N_model-1)); ax.set_xticklabels(Legends)
            ax.set_yticks(0:(N_model-1)); ax.set_yticklabels(Legends)
            ax.set_ylabel("True model"); ax.set_xlabel("Recovered model")
            ax.set_title("Recovery rate with β = " * string(β) * ", K = " * string(K))

            tight_layout()
            savefig(SavePath * "GeneralRecovery_singlesub_beta_" * string(β) *
                              "_K_" * string(K) * nametag * ".png")
            savefig(SavePath * "GeneralRecovery_singlesub_beta_" * string(β) *
                              "_K_" * string(K) * nametag * ".pdf")
            savefig(SavePath * "GeneralRecovery_singlesub_beta_" * string(β) *
                              "_K_" * string(K) * nametag * ".svg")
      end
end