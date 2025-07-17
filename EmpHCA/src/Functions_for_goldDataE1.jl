# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function ExcIndicatorExp1!(df;
                        room1preference = 0.7,
                        selection_timeout = 10,
                        attention_check_fail = 2,
                        skip_survey = 1,
                        straightlining = 0.8,
                        zigzagging = 0.8)
    goodsubs_task = (df.room1preference .>= room1preference)
    goodsubs_task .&= (df.selection_timeout .< selection_timeout)
    df.task_outliers = (1 .- goodsubs_task) .== 1
    
    goodsubs_survey = (df.attention_check_fail .< attention_check_fail)
    goodsubs_survey .&= (df.skip_survey .< skip_survey)
    goodsubs_survey .&= (df.straightlining .< straightlining)
    goodsubs_survey .&= (df.zigzagging .< zigzagging)
    df.survey_outliers = (1 .- goodsubs_survey) .== 1

    df.outliers = (df.task_outliers .+ df.survey_outliers) .> 0
end
export ExcIndicatorExp1!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Consistency calculation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Xinds = pairs of room indices
# as_inds = chosen room
function choice_consistency(Xinds, as_inds; ifpassinds = false, ifraw = false)
    inds = [[x[2],x[1]] ∈ Xinds for x = Xinds]
    as_inds = as_inds[inds]; Xinds = Xinds[inds]; 
    # consistency
    consistency_indeces = zero(as_inds)
    for t = eachindex(as_inds)
            x = Xinds[t]
            a1 = as_inds[t]
            a2 = as_inds[[[x[2],x[1]] == x2 for x2 = Xinds]][1]
            consistency_indeces[t] = a1 == a2
    end
    consistency_index = mean(consistency_indeces)
    
    # correction
    if !ifraw
        consistency_index = (consistency_index + 1) / 2
    end
    if ifpassinds
        return consistency_index, inds
    else
        return consistency_index
    end
end
function choice_consistency(Xinds1, as_inds1, 
                            Xinds2, as_inds2; ifpassinds = false)
    # choosing repeated pairs
    inds11 = [[x[2],x[1]] ∈ Xinds1 for x = Xinds1]
    as_inds1 = as_inds1[inds11]; Xinds1 = Xinds1[inds11]; 
    inds22 = [[x[2],x[1]] ∈ Xinds2 for x = Xinds2]
    as_inds2 = as_inds2[inds22]; Xinds2 = Xinds2[inds22]; 

    # choosing common pairs
    inds1 = [[x[1],x[2]] ∈ Xinds2 for x = Xinds1]
    as_inds1 = as_inds1[inds1]; Xinds1 = Xinds1[inds1]; 
    inds2 = [[x[1],x[2]] ∈ Xinds1 for x = Xinds2]
    as_inds2 = as_inds2[inds2]; Xinds2 = Xinds2[inds2]; 
    
    
    if length(as_inds1) != length(as_inds2)
        error("Something wrong")
    end

    # consistency
    consistency_indeces = zero(as_inds1) .* 1.0
    for t = eachindex(as_inds1)
            x = Xinds1[t]
            a1s = [as_inds1[t],
                   as_inds1[[[x[2],x[1]] == x2 for x2 = Xinds1]][1]]
            a2s = [as_inds2[[[x[1],x[2]] == x2 for x2 = Xinds2]][1],
                   as_inds2[[[x[2],x[1]] == x2 for x2 = Xinds2]][1]]
            consistency_indeces[t] = mean([mean(a1s .== a) for a = a2s])
    end
    consistency_index = mean(consistency_indeces)
    if ifpassinds
        return consistency_index, inds1, inds2
    else
        return consistency_index
    end
end
export choice_consistency


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: Emp l fitting and selection
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# fitting L unconstrained
@model function TuringGold(Xs, as; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # parameters
    l ~ truncated(Normal(0.0, 10), 0, Inf)
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
export TuringGold

# fitting l and m (= range of l) together
@model function TuringGoldMSel(Xs, as; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # model
    m ~ Categorical(3)
    # parameters
    l0 ~ truncated(Normal(0.5, 1.0), 0, 1)
    l2 ~ truncated(Normal(1, 2.), 1, Inf)
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
export TuringGoldMSel

# fitting l conditioned on m (= range of l)
@model function TuringGoldMSelCondM(Xs, as, m; K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)

    # parameters
    l0 ~ truncated(Normal(0.5, 1.0), 0, 1)
    l2 ~ truncated(Normal(1, 2.), 1, Inf)
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
export TuringGoldMSelCondM

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: Inferring preferences of Bradley-Terry model
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBasicInf(Xs, as; N_rooms = 12)
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
export TuringGoldBasicInf

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: General model selection
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBasicInfvsEmplvsNa(Xs, Xinds, Nas, as; 
                                            N_rooms = 12, K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)
    
    # model
    m ~ Categorical(4)
    
    # Na parameters
    βa ~ truncated(Normal(0.0, 10), 0, Inf)
    
    # Emp l parameters
    l ~ truncated(Normal(0.0, 5.), 0, Inf)
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
export TuringGoldBasicInfvsEmplvsNa


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
