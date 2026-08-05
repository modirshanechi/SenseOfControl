################################################################################
# Code for evaluating marginal likelihood (model evidence) via bridge
# sampling, and using it for hierarchical (random-effects) model selection
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random
using Turing, MCMCChains, Distributions
using DataFrames
using CSV
using JLD2
using AdvancedMH

import StatsPlots

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

thinning = 5

Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

# ----------------------------------------------------------------------
# registry of fitted models: how to find each model's chain inside the
# per-subject jld2 dict and how to rebuild the Turing model object that
# bridge sampling needs.
# "Random" is a closed-form baseline evaluated directly from the
# trial data. Order matches ModelNames.MAll (Random, Na, Emp-l, General)
# ----------------------------------------------------------------------
model_specs = [
    (name = "Random", key = nothing,               field = nothing,
        build = nothing),
    (name = "Na",     key = "FITresultsNa",        field = :chnA,
        build = (Xinds, Xs, Nas, as) -> TuringGoldNa(Nas, as)),
    (name = "Empl",   key = "FITresultsEmpl",      field = :chnemp,
        build = (Xinds, Xs, Nas, as) -> TuringGoldEmpl(Xs, as; K = 1)),
    (name = "BD",     key = "FITresultsBD",        field = :chnBD,
        build = (Xinds, Xs, Nas, as) -> TuringGoldBradTerry(Xinds, as; N_rooms = N_rooms)),
]
N_modelAll = 4

# ----------------------------------------------------------------------
# marginal log-likelihood of one subject under one model_specs entry, via
# bridge sampling on its already-fitted chain
# ----------------------------------------------------------------------
function eval_marginal_logp(rng, infdata, spec, Xinds, Xs, Nas, as; thinning = 1)
    if spec.key === nothing
        return random_model_logp(Xinds, as)
    end

    result = infdata[spec.key]

    chn = getproperty(result, spec.field)
    chn = chn[1:thinning:size(chn, 1), :, :]
    mdl = spec.build(Xinds, Xs, Nas, as)

    return bridgesampling(rng, chn, mdl; tol = 1e-3).logml
end

# ----------------------------------------------------------------------
# subject x model matrix of bridge-sampling log evidences, loading each
# subject's fitted chains from the per-subject jld2 (optionally restricted
# to gb_idx, 1 = Gold, 2 = Bomb, for the gold/bomb split experiment)
# ----------------------------------------------------------------------
function marginal_logp_matrix(expspec, selDF, subjectIDs,
                                Prooms, ΔState, ΔStateDict, Xmax, Ymax;
                                gb_idx = nothing, thinning = thinning)
    logml_mat = fill(NaN, length(subjectIDs), length(model_specs))
    for i = eachindex(subjectIDs)
        @show i
        i_sub = subjectIDs[i]
        rng = Xoshiro(2024 + i_sub)

        infdata = load(expspec.fig_path * "MCMC_sub" * string(i_sub) * ".jld2")
        if gb_idx !== nothing
            for k = keys(infdata)
                infdata[k] = infdata[k][gb_idx]
            end
        end

        df = selDF[selDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        if gb_idx !== nothing
            df = df[df.Gtrials .== (gb_idx == 1 ? 1 : 0), :]
        end

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        as = df.action .+ 1

        Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax, 1)
        Nas = [[size(Prooms[x[1]])[1], size(Prooms[x[2]])[1]] for x = Xinds]

        for j = eachindex(model_specs)
            @show model_specs[j].name
            logml_mat[i, j] = eval_marginal_logp(rng, infdata, model_specs[j],
                                                    Xinds, Xs, Nas, as; thinning)
        end
    end
    return logml_mat
end

# ----------------------------------------------------------------------
# loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    @show i_exp
    # ----------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec)
    selDF = data.selDF; subjectIDs = data.subjectIDs

    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

    # ----------------------------------------------------------------------
    # evaluating marginal likelihoods and model selection
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        L_matrixAll = marginal_logp_matrix(expspec, selDF, subjectIDs,
                        Prooms, ΔState, ΔStateDict, Xmax, Ymax; thinning)
    else
        # gold/bomb split: run the same (All, Emp) selection separately for
        # each condition (gb_idx 1 = Gold, 2 = Bomb)
        logml_matG = marginal_logp_matrix(expspec, selDF, subjectIDs,
                        Prooms, ΔState, ΔStateDict, Xmax, Ymax; gb_idx = 1, thinning)
        logml_matB = marginal_logp_matrix(expspec, selDF, subjectIDs,
                        Prooms, ΔState, ΔStateDict, Xmax, Ymax; gb_idx = 2, thinning)
        L_matrixAll = [logml_matG, logml_matB]
    end

    save(expspec.fig_path * "BridgeSampLmatrix.jld2", "L_matrixAll", L_matrixAll)
end
