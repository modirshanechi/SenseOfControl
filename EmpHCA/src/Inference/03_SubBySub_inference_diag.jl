################################################################################
# Code for convergence diagnosis of non-hierarchical inference
################################################################################
using PyPlot
using EmpHCA
using DataFrames
using Random

using LinearAlgebra

using Turing, MCMCChains, Distributions
using CSV
using JLD2
using AdvancedMH

import StatsPlots

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ----------------------------------------------------------------------
# helper for rhat/ess scatter coloring
# ----------------------------------------------------------------------
diagcolors = ["green", "orange", "orange", "red"]
diaglabels = ["Rhat<1.01 & ESS>1'000", "Rhat>1.01 & ESS>1'000",
              "Rhat<1.01 & ESS<1'000", "Rhat>1.01 & ESS<1'000"]
diagcat(r, e) = r <= 1.01 && e >= 1000 ? 1 :
                r >  1.01 && e >= 1000 ? 2 :
                r <= 1.01 && e <  1000 ? 3 : 4

# ----------------------------------------------------------------------
# reduce a vector of per-subject diagnosis tables to per-subject
# max Rhat / min ESS; shared by extract_rhat_ess and extract_rhat_ess_cv
# ----------------------------------------------------------------------
function diagnosis_rhat_ess(diags; capr = 2.)
    maxrhats = [findmax(v.rhat[isnan.(v.rhat) .== 0])[1] for v = diags]
    maxrhats[maxrhats .> capr] .= capr

    miness = [findmin(v.ess_min[isnan.(v.ess_min) .== 0])[1] for v = diags]

    return maxrhats, miness
end

# ----------------------------------------------------------------------
# per-subject max Rhat / min ESS for one result `key` (e.g. "MSresultsEmp")
# and one `field` of that result holding the MCMC chains (e.g. :chnEmp).
# When the experiment has a gold/bomb split, each infdata[i][key] is a 2-vector
# [gold, bomb]; pass gb_idx = 1 or 2 to select which half to read.
# ----------------------------------------------------------------------
function extract_rhat_ess(infdata, key, field, discvar; gb_idx = nothing,
                            capr = 2.)
    results = [gb_idx === nothing ? d[key] : d[key][gb_idx] for d = infdata]
    chns = [getproperty(r, field) for r = results];
    diags = Vector{DataFrame}([])
    for i = eachindex(chns)
        @show (i, length(chns))
        push!(diags,
            MCMC_convergence_diagnostics(chns[i];  discrete_parameters=discvar))
    end
    return diagnosis_rhat_ess(diags; capr = capr)
end


# ----------------------------------------------------------------------
# per-subject max Rhat / min ESS for the cross-validated Empl fit saved
# by 02_SubBySub_CVinference.jl (key "data_save", field
# :FITresultsEmpl_CV, a 2-vector [set1, set2]); pass cv_idx = 1 or 2 to
# select the split half. When the experiment has a gold/bomb split, the
# split is the outer index: cvinfdata[i]["data_save"] is [gold, bomb],
# each holding its own FITresultsEmpl_CV pair.
# ----------------------------------------------------------------------
function extract_rhat_ess_cv(cvinfdata, cv_idx; gb_idx = nothing,
                            capr = 2.)
    results = [gb_idx === nothing ? d["data_save"] : d["data_save"][gb_idx]
                    for d = cvinfdata]
    chns = [getproperty(r.FITresultsEmpl_CV[cv_idx], :chnemp) for r = results];
    diags = Vector{DataFrame}([])
    for i = eachindex(chns)
        @show (i, length(chns))
        push!(diags,
            MCMC_convergence_diagnostics(chns[i];  
                        discrete_parameters=Dict{Symbol,Int}()))
    end
    return diagnosis_rhat_ess(diags; capr = capr)
end

# ----------------------------------------------------------------------
# scatter plot of max Rhat vs min ESS, colored by convergence category
# ----------------------------------------------------------------------
function plot_diag_scatter(maxrhats, miness, fig_path, name)
    figure(figsize = (5,5))
    ax = subplot(1,1,1)
    cats = diagcat.(maxrhats, miness)
    for k = 1:4
            pct = round(100 * count(cats .== k) / length(cats), digits = 1)
            ax.scatter(maxrhats[cats .== k], miness[cats .== k],
                            color = diagcolors[k], alpha = 0.5,
                            label = "$(diaglabels[k]) ($pct%)")
    end
    ax.set_title(name)
    ax.set_xlabel("max Rhat"); ax.set_ylabel("min ESS")
    ax.legend()

    tight_layout()
    savefig(fig_path * "MCMCDiag/InferenceDiagnostics" * name * "Scatter.pdf")

    return cats
end

# ----------------------------------------------------------------------
# loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    # ----------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec); 
    subjectIDs = data.subjectIDs

    # ----------------------------------------------------------------------
    # load inference data
    # ----------------------------------------------------------------------
    infdata = [load(expspec.fig_path * "MCMC_sub" *
                    string(i_sub) * ".jld2") for i_sub = subjectIDs]
    
    # ----------------------------------------------------------------------
    # rhat vs ess diagnostics for the four inference objects:
    # (dict key, diagnosis field, dict label, filename suffix)
    # ----------------------------------------------------------------------
    diagspecs = [
        (key = "MSresultsEmp",   field = :chnemp, label = "MSEmp", suffix = "MSEmp", discvar = Dict(:m => 3)),
        (key = "FITresultsBD",   field = :chnBD,  label = "BD",    suffix = "BD",    discvar = Dict{Symbol,Int}()),
        (key = "FITresultsEmpl", field = :chnemp, label = "Empl",  suffix = "Empl",  discvar = Dict{Symbol,Int}()),
        (key = "FITresultsNa",   field = :chnA,   label = "Na",    suffix = "Na",    discvar = Dict{Symbol,Int}()),
    ]

    cats = Dict{String, Vector{Int}}()
    for spec = diagspecs
        key, field, label, filesuffix, discvar = 
                    spec.key, spec.field, spec.label, spec.suffix, spec.discvar
        if expspec.has_gb_split
            for (gb_idx, gb_name) = [(1, "Gold"), (2, "Bomb")]
                maxrhats, miness =
                        extract_rhat_ess(infdata, key, field, discvar; gb_idx = gb_idx)
                cats[label * gb_name] = plot_diag_scatter(maxrhats, miness,
                                            expspec.fig_path, filesuffix * gb_name)
            end
        else
            maxrhats, miness = extract_rhat_ess(infdata, key, field, discvar)
            cats[label] = plot_diag_scatter(maxrhats, miness,
                                            expspec.fig_path, filesuffix)
        end
    end

    # ----------------------------------------------------------------------
    # rhat vs ess diagnostics for the cross-validated Empl fit (2 splits)
    # ----------------------------------------------------------------------
    cvinfdata = [load(expspec.fig_path * "MCMC_sub" *
                    string(i_sub) * "_CV.jld2") for i_sub = subjectIDs]
    
    cvdiagspecs = [
        (cv_idx = 1, label = "EmplCVset1", suffix = "EmplCVset1"),
        (cv_idx = 2, label = "EmplCVset2", suffix = "EmplCVset2"),
    ]

    for spec = cvdiagspecs
        if expspec.has_gb_split
            for (gb_idx, gb_name) = [(1, "Gold"), (2, "Bomb")]
                maxrhats, miness =
                        extract_rhat_ess_cv(cvinfdata, spec.cv_idx; gb_idx = gb_idx)
                cats[spec.label * gb_name] = plot_diag_scatter(maxrhats, miness,
                                            expspec.fig_path, spec.suffix * gb_name)
            end
        else
            maxrhats, miness = extract_rhat_ess_cv(cvinfdata, spec.cv_idx)
            cats[spec.label] = plot_diag_scatter(maxrhats, miness,
                                            expspec.fig_path, spec.suffix)
        end
    end
end
