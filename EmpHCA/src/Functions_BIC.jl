# ----------------------------------------------------------------------
# General model specifications for BIC evaluation
#   key: results for parameter inference used for the initialization
#   field: the particular field of "key" that refers to the MCMC dataframe
#   fit_function: function for fitting the maximum likelihood estimate
#
# modelID = 1 (Random) has no MLE fit — its log-likelihood is closed-form via
# random_model_logp and contributes 0 free parameters to the BIC penalty.
# ----------------------------------------------------------------------
const modelspecs_BIC = [
        (modelID = 1, key = nothing,          field = nothing,   
            modelparams = Symbol[],  label = "Random", fit_function = nothing),
        (modelID = 2, key = "FITresultsNa",   field = :chnA_df,  
            modelparams = [:βa],     label = "Na",     fit_function = Na_fit_MLE),
        (modelID = 3, key = "FITresultsEmpl", field = :chnemp_df, 
            modelparams = [:logl, :γ, :β], label = "Empl",  fit_function = EmpL_fit_MLE),
        (modelID = 4, key = "FITresultsBD",   field = :chnBD_df,
            modelparams = vcat(:βθ, [Symbol("θ[$i]") for i = 1:length(PaperRoomOrder)]),
            label = "BD",     fit_function = BradTerry_fit_MLE),
    ]
export modelspecs_BIC


# ----------------------------------------------------------------------
# number of parameters
# ----------------------------------------------------------------------
# Fixed (not data-derived) number of free parameters used for the BIC penalty
# of each model. BD's count depends on the experiment because the number of
# rooms entering the Bradley-Terry fit differs across experiments.
function n_params_for_BIC(label, i_exp)
    if label == "Na"
        return 1
    elseif label == "Empl"
        return 2
    elseif label == "BD"
        return i_exp == 1 ? 11 : 8
    else
        error("No fixed BIC parameter count defined for label $label")
    end
end

# ----------------------------------------------------------------------
# per-subject BIC evaluation
# ----------------------------------------------------------------------
function bic_evaluation(mspecs, infdata_sub, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax, i_exp)
    if mspecs.label == "Random"
        return (; BIC = random_model_logp(Xinds, as), MLEres=nothing)
    end
    # reading the MCMC inference data
    infdata_temp = infdata_sub[mspecs.key][mspecs.field]

    # considering expected posterior as the parameter initialization
    if mspecs.label == "BD"
        θ_init = [mean(infdata_temp[:, Symbol("θ[$i]")]) for i = 1:N_rooms]
        init_params = Dict(:βθ => mean(infdata_temp[:,:βθ]), :θ => θ_init)
    else
        init_params = Dict([m => mean(infdata_temp[:,m])
                                for m = mspecs.modelparams])
    end

    # MLE optimization
    MLEres = mspecs.fit_function(Xinds, as, init_params,
                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)
    n_params = n_params_for_BIC(mspecs.label, i_exp)

    # BIC computation
    BIC = MLEres.maxloglikelihood - n_params / 2 * log(length(Xinds))

    return (; BIC = BIC, MLEres=MLEres)
end
export bic_evaluation

# ----------------------------------------------------------------------
# subject x model BIC evaluation
# -> when the experiment has a gold/bomb split, pass gb_idx = 1 or 2 to select
#    which half of each per-subject result (and of the trial data) to use.
# ----------------------------------------------------------------------
function bic_logp_matrix(selDF, subjectIDs, infdata,
                            Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                            modelspecs_BIC, i_exp; gb_idx = nothing)
    L_matrixAll = fill(NaN, length(subjectIDs), length(modelspecs_BIC))
    for i = eachindex(subjectIDs)
        @show i
        i_sub = subjectIDs[i]
        
        # selecting subject i_sub's choices
        df = selDF[selDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        if gb_idx !== nothing
            df = df[df.Gtrials .== (gb_idx == 1 ? 1 : 0), :]
        end
        Xinds = [[df.room1[k], df.room2[k]] .+ 1 for k = 1:size(df)[1]]
        as = df.action .+ 1

        # selecting subject i_sub's MCMC data
        infdata_sub = gb_idx === nothing ? infdata[i] :
                        Dict(k => v[gb_idx] for (k,v) = infdata[i])

        # BIC computation for all models
        for j = eachindex(modelspecs_BIC)
            L_matrixAll[i,j] = bic_evaluation(modelspecs_BIC[j], infdata_sub, Xinds, as,
                                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax, i_exp).BIC
        end
    end
    return L_matrixAll
end
export bic_logp_matrix