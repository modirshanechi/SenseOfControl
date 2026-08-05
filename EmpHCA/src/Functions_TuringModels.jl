################################################################################
# Code for general functions of models in Turing.jl and their inference
################################################################################

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: Emp-l fitting and selection
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# fitting l unconstrained
@model function TuringGoldEmpl(Xs, as; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # parameters
    # l ~ truncated(Normal(0.0, 10), 0, Inf)
    logl ~ Normal(0.0, 2)
    l = exp(logl)

    γ ~ truncated(Normal(0.5, 2.), 0, 1.0)
    
    β ~ truncated(Normal(0.0, 10), 0, Inf)
    agent = GoldAgent(Vector{typeof(β)}(undef, N_a))
    # Samples
    for t = 1:N_trial
          if K > 1
                emp_model = emplK(l,γ,K)
          else
                emp_model = emplK(l,1.,1)
          end
          gold_pa1!(Xs[t], emp_model, β, agent)
          as[t] ~ Categorical(agent.pa)
    end
end;
export TuringGoldEmpl

# fitting l and m (= range of l) together
@model function TuringGoldMselEmpl(Xs, as; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # model
    m ~ Categorical(3)
    # parameters
    logl0 ~ truncated(Normal(0., 2.), -Inf, 0)
    logl2 ~ truncated(Normal(0., 2.),  0, Inf)
    l0 = exp(logl0)
    l2 = exp(logl2)
    γ ~ truncated(Normal(0.5, 2.), 0, 1.0)
    
    β ~ truncated(Normal(0.0, 10), 0, Inf)
    agent = GoldAgent(Vector{typeof(β)}(undef, N_a))

    # Samples
    for t = 1:N_trial
          if m == 1
                if K > 1
                      emp_model = emplK(l0,γ,K)
                else
                      emp_model = emplK(l0,1.,1)
                end
          elseif m == 2
                if K > 1
                      emp_model = emplK(1.,γ,K)
                else
                      emp_model = emplK(1.,1.,1)
                end
          elseif m == 3
                if K > 1
                      emp_model = emplK(l2,γ,K)
                else
                      emp_model = emplK(l2,1.,1)
                end
          end
          gold_pa1!(Xs[t], emp_model, β, agent)
          as[t] ~ Categorical(agent.pa)
    end
end;
export TuringGoldMselEmpl

# fitting l conditioned on m (= range of l)
@model function TuringGoldEmplCondM(Xs, as, m; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # parameters
    logl0 ~ truncated(Normal(0., 2.), -Inf, 0)
    logl2 ~ truncated(Normal(0., 2.),  0, Inf)
    l0 = exp(logl0)
    l2 = exp(logl2)

    γ ~ truncated(Normal(0.5, 2.), 0, 1.0)
    
    β ~ truncated(Normal(0.0, 10), 0, Inf)
    agent = GoldAgent(Vector{typeof(β)}(undef, N_a))

    # Samples
    for t = 1:N_trial
          if m == 1
                if K > 1
                      emp_model = emplK(l0,γ,K)
                else
                      emp_model = emplK(l0,1.,1)
                end
          elseif m == 2
                if K > 1
                      emp_model = emplK(1.,γ,K)
                else
                      emp_model = emplK(1.,1.,1)
                end
          elseif m == 3
                if K > 1
                      emp_model = emplK(l2,γ,K)
                else
                      emp_model = emplK(l2,1.,1)
                end
          end
          gold_pa1!(Xs[t], emp_model, β, agent)
          as[t] ~ Categorical(agent.pa)
    end
end;
export TuringGoldEmplCondM

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: Inferring preferences of Bradley-Terry model
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBradTerry(Xs, as; N_rooms = 12)
    # Get observation length.
    N_trial = length(Xs)
    
    # parameters
    βθ ~ truncated(Normal(0.0, 10), 0, Inf)
    θ  ~ MvNormal(zeros(N_rooms), 1.0 * I)
    
    # Samples
    for t = 1:N_trial
        i1, i2 = Xs[t]
        if i1 == 1
            v1 = 0
        else
            v1 = βθ * θ[i1]
        end
        if i2 == 1
            v2 = 0
        else
            v2 = βθ * θ[i2]
        end
        
        v = [v1, v2]
        pa = softmax(v)
        as[t] ~ Categorical(pa)
    end
end;
export TuringGoldBradTerry

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: General model selection
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBradTerryvsEmplvsNa(Xs, Xinds, Nas, as; 
                                            N_rooms = 12, K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)
    
    # model
    m ~ Categorical(4)
    
    # Na parameters
    βa ~ truncated(Normal(0.0, 10), 0, Inf)
    
    # Emp l parameters
    # l ~ truncated(Normal(0.0, 5.), 0, Inf)
    logl ~ Normal(0.0, 2)
    l = exp(logl)
    
    γ ~ truncated(Normal(0.5, 2.), 0, 1.0)
    β ~ truncated(Normal(0.0, 10), 0, Inf)

    # unstructured parameters
    βθ ~ truncated(Normal(0.0, 10), 0, Inf)
    θ ~ MvNormal(zeros(N_rooms), 1. * I)

    agent = GoldAgent(Vector{typeof(β)}(undef, N_a))
    # Samples
    for t = 1:N_trial
        if m == 1   # Random model
            as[t] ~ Categorical(ones(N_a) ./ N_a)
        elseif m == 2   # Na model
            gold_pa1_Na!(Nas[t], βa, agent)
            as[t] ~ Categorical(agent.pa)
        elseif m == 3   # Emp-l model
            if K > 1
                emp_model = emplK(l,γ,K)
            else
                emp_model = emplK(l,1.,1)
            end
            gold_pa1!(Xs[t], emp_model, β, agent)
            as[t] ~ Categorical(agent.pa)
        elseif m == 4   # General model
            i1, i2 = Xinds[t]
            if i1 == 1
                v1 = 0
            else
                v1 = βθ * θ[i1]
            end
            if i2 == 1
                v2 = 0
            else
                v2 = βθ * θ[i2]
            end
            pa = softmax([v1, v2])
            as[t] ~ Categorical(pa)
        end
    end
end;
export TuringGoldBradTerryvsEmplvsNa


@model function TuringGoldNa(Nas, as; N_a = 2)
    # Get observation length.
    N_trial = length(Nas)
    
    # Na parameters
    βa ~ truncated(Normal(0.0, 10), 0, Inf)
    agent = GoldAgent(Vector{typeof(βa)}(undef, N_a))
    # Samples
    for t = 1:N_trial
        gold_pa1_Na!(Nas[t], βa, agent)
        as[t] ~ Categorical(agent.pa)
    end
end;
export TuringGoldNa

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Random model log-p
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function random_model_logp(Xinds, as)
    return sum(- log.(length.(Xinds)))
end
export random_model_logp


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# MCMC convergence diagnostics
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# export MCMC_convergence_diagnostics
# Replace each discrete parameter p (with support 1, ..., M_max) by its
# indicator variables I_i = (p == i), i = 1, ..., M_max, so that Rhat/ESS are
# computed on binary indicators rather than on the raw (unordered) category
# codes. `discrete_parameters` maps each discrete parameter name to its M_max.
function expand_discrete_indicators(pchn::Chains, discrete_parameters::Dict{Symbol,Int})
    isempty(discrete_parameters) && return pchn, Symbol[]

    disc_names = collect(keys(discrete_parameters))
    cont_names = filter(p -> !(p in disc_names), names(pchn))

    n_iter, _, n_chains = size(pchn)

    indicator_names = Symbol[]
    indicator_cols = Matrix{Float64}[]
    for p in disc_names
                pvals = Array(pchn[p])   # n_iter x n_chains
        for i in 1:discrete_parameters[p]
            push!(indicator_names, Symbol(p, "_eq_", i))
            push!(indicator_cols, Float64.(pvals .== i))
        end
    end

    indicator_data = Array{Float64}(undef, n_iter, length(indicator_cols), n_chains)
    for (j, col) in enumerate(indicator_cols)
        indicator_data[:, j, :] .= col
    end
    indicator_chn = Chains(indicator_data, indicator_names; iterations = range(pchn))

    isempty(cont_names) && return indicator_chn, indicator_names
    return hcat(pchn[cont_names], indicator_chn), indicator_names
end
export expand_discrete_indicators
function MCMC_convergence_diagnostics(
    chn::Chains;
    max_rhat = 1.01,
    min_ess_per_chain = 100,
    discrete_parameters::Dict{Symbol,Int} = Dict{Symbol,Int}(), # e.g., Dict(:m => 4)
)
    pchn = Chains(chn, :parameters)
    n_chains = size(pchn, 3)
    min_ess = min_ess_per_chain * n_chains

    pchn, discrete_names = expand_discrete_indicators(pchn, discrete_parameters)
    vals = Array(pchn)

    df = DataFrame(summarystats(pchn))
    select!(df, :parameters, :ess_bulk, :ess_tail, :rhat)
    rename!(df, :parameters => :parameter)

    df.ess_min = min.(df.ess_bulk, df.ess_tail)

    df.constant = [
        let x = vec(@view vals[:, j, :])
            all(y -> isequal(y, first(x)), x)
        end
        for j in axes(vals, 2)
    ]

    df.discrete = in.(df.parameter, Ref(discrete_names))

    df.rhat_pass = isfinite.(df.rhat) .& (df.rhat .< max_rhat)
    df.bulk_ess_pass = isfinite.(df.ess_bulk) .& (df.ess_bulk .>= min_ess)
    df.tail_ess_pass = isfinite.(df.ess_tail) .& (df.ess_tail .>= min_ess)

    df.pass = [
        row.constant ||
        (
            row.rhat_pass &&
            row.bulk_ess_pass &&
            (
                row.tail_ess_pass ||
                (row.discrete && isnan(row.ess_tail))
            )
        )
        for row in eachrow(df)
    ]

    df.status = [
        row.constant ? "constant; Rhat/ESS not diagnostic" :
        row.pass && row.discrete && isnan(row.ess_tail) ?
            "pass; tail ESS undefined for discrete variable" :
        row.pass ? "pass" :
        "fail"
        for row in eachrow(df)
    ]

    return df
end
export MCMC_convergence_diagnostics

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# General model selection: random, Na, Emp-l, and General
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function general_model_selection(rng, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)
    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, 
                                            ΔStateDict, Xmax, Ymax,1)
    Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
    modelAll = TuringGoldBradTerryvsEmplvsNa(Xs, Xinds, Nas, as; N_rooms = N_rooms, K = 1)
    gAll = Gibbs(HMC(HMC_param1, HMC_param2, :θ, :logl, :β, :βa, :βθ), 
                    MH(:m, :γ))
    chnAll = sample(rng, modelAll, gAll, MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)
    
    chnAll_df = DataFrame(chnAll)
    chnAll_df = chnAll_df[1:thinning:size(chnAll_df)[1],:]
    return (; chnAll_df=chnAll_df, chnAll=chnAll)
end
export general_model_selection


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Emp-l model selection: l0, l1, l2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Emp_model_selection(rng, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)
    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, 
                                            ΔStateDict, Xmax, Ymax,1)
    modelemp = TuringGoldMselEmpl(Xs, as, K=1); 
    gemp = Gibbs(HMC(HMC_param1, HMC_param2, :logl0, :logl2, :β), MH(:m, :γ))
    chnemp = sample(rng, modelemp, gemp,  MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)
    
    chnemp_df = DataFrame(chnemp)
    chnemp_df = chnemp_df[1:thinning:size(chnemp_df)[1],:]
    return (; chnemp_df=chnemp_df, chnemp=chnemp)
end
export Emp_model_selection

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Na fitting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Na_fit(rng, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)

    Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
    modelA = TuringGoldNa(Nas, as)
    gA = HMC(HMC_param1, HMC_param2, :βa)
    chnA = sample(rng, modelA, gA,  MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)

    chnA_df = DataFrame(chnA)
    chnA_df = chnA_df[1:thinning:size(chnA_df)[1],:]
    return (; chnA_df=chnA_df, chnA=chnA)
end
export Na_fit

function Na_fit_MLE(Xinds, as, init_params::Dict{Symbol,<:Real},
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                    param_change_tol = 1e-6)
    Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
    modelA = TuringGoldNa(Nas, as)
    mle_result = maximum_likelihood(modelA; initial_params = [init_params[:βa]])

    mle_params = Dict(:βa => mle_result.values[:βa])
    n_params_changed =
        count(p -> abs(mle_params[p] - init_params[p]) >
                param_change_tol, keys(init_params))

    return (; mle_result=mle_result,
                mle_params=mle_params,
                maxloglikelihood=mle_result.lp,
                n_params_changed=n_params_changed)
end
export Na_fit_MLE

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Unconstrained l fitting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function EmpL_fit(rng, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)

    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, 
                                            ΔStateDict, Xmax, Ymax,1)
    modelemp = TuringGoldEmpl(Xs, as; K = 1)
    gemp = Gibbs(HMC(HMC_param1, HMC_param2, :logl, :β), MH(:γ))
    chnemp = sample(rng, modelemp, gemp,  MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)
    
    chnemp_df = DataFrame(chnemp)
    chnemp_df = chnemp_df[1:thinning:size(chnemp_df)[1],:]
    return (; chnemp_df=chnemp_df, chnemp=chnemp)
end
export EmpL_fit

function EmpL_fit_MLE(Xinds, as, init_params::Dict{Symbol,<:Real},
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                    param_change_tol = 1e-6)
    param_names = [:logl, :γ, :β]

    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState,
                                            ΔStateDict, Xmax, Ymax,1)
    modelemp = TuringGoldEmpl(Xs, as; K = 1)
    init_vec = [init_params[p] for p in param_names]
    mle_result = maximum_likelihood(modelemp; initial_params = init_vec)

    mle_params = Dict(p => mle_result.values[p] for p in param_names)
    n_params_changed = 
        count(p -> abs(mle_params[p] - init_params[p]) > 
                param_change_tol,param_names)

    return (; mle_result=mle_result,
                mle_params=mle_params,
                maxloglikelihood=mle_result.lp,
                n_params_changed=n_params_changed)
end
export EmpL_fit_MLE


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# m-conditioned l fitting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function EmpL_mcond_fit(rng, m, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)

    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, 
                                            ΔStateDict, Xmax, Ymax,1)
    modelemp = TuringGoldEmplCondM(Xs, as, m; K = 1)
    gemp = Gibbs(HMC(HMC_param1, HMC_param2, :logl0, :logl2, :β), MH(:γ))
    chnemp = sample(rng, modelemp, gemp,  MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)

    chnemp_df = DataFrame(chnemp)
    chnemp_df = chnemp_df[1:thinning:size(chnemp_df)[1],:]
    return (; chnemp_df=chnemp_df, chnemp=chnemp)
end
export EmpL_mcond_fit


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Bradley-Terry fitting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BradTerry_fit(rng, Xinds, as,
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                    n_chains, chain_length, burn_in_length, thinning;
                    HMC_param1 = 0.01, HMC_param2 = 50)

    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, 
                                            ΔStateDict, Xmax, Ymax,1)
    modelBD = TuringGoldBradTerry(Xinds, as; N_rooms = N_rooms)
    gBD = HMC(HMC_param1, HMC_param2, :θ, :βθ)
    chnBD = sample(rng, modelBD, gBD,  MCMCSerial(), 
                    chain_length, n_chains; discard_initial = burn_in_length)

    chnBD_df = DataFrame(chnBD)
    chnBD_df = chnBD_df[1:thinning:size(chnBD_df)[1],:]
    return (; chnBD_df=chnBD_df, chnBD=chnBD)
end
export BradTerry_fit

function BradTerry_fit_MLE(Xinds, as, init_params::Dict{Symbol,<:Any},
                    Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                    param_change_tol = 1e-6)
    # init_params must have keys :βθ (a scalar) and :θ (a vector of length N_rooms),
    # e.g. init_params = Dict(:βθ => 1.0, :θ => zeros(N_rooms)).
    # βθ is fixed at its initial value; only θ is optimized.
    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState,
                                            ΔStateDict, Xmax, Ymax,1)
    modelBD = TuringGoldBradTerry(Xinds, as; N_rooms = N_rooms)
    modelBD = Turing.fix(modelBD, (; βθ = init_params[:βθ]))
    init_vec = init_params[:θ]
    mle_result = maximum_likelihood(modelBD; initial_params = init_vec)

    mle_vals = mle_result.values.array
    mle_params = Dict(:βθ => init_params[:βθ], :θ => mle_vals)
    n_params_changed = count(abs.(mle_vals .- init_vec) .> param_change_tol)

    return (; mle_result=mle_result,
                mle_params=mle_params,
                maxloglikelihood=mle_result.lp,
                n_params_changed=n_params_changed)
end
export BradTerry_fit_MLE



