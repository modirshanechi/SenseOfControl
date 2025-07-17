# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function ExcIndicatorExp2!(df;
                        room1preference = 0.75,
                        selection_timeout_gold = 5,
                        selection_timeout_bomb = 5,
                        attention_check_fail = 2,
                        skip_survey = 1,
                        straightlining = 0.8,
                        zigzagging = 0.8)
    goodsubs_task   = (df.room1preference_gold .>= room1preference) .&
                      (df.room1preference_bomb .>= room1preference)
    goodsubs_task .&= (df.selection_timeout_gold .< selection_timeout_gold)
    goodsubs_task .&= (df.selection_timeout_bomb .< selection_timeout_bomb)
    df.task_outliers = (1 .- goodsubs_task) .== 1
    
    goodsubs_survey = (df.attention_check_fail .< attention_check_fail)
    goodsubs_survey .&= (df.skip_survey .< skip_survey)
    goodsubs_survey .&= (df.straightlining .< straightlining)
    goodsubs_survey .&= (df.zigzagging .< zigzagging)
    df.survey_outliers = (1 .- goodsubs_survey) .== 1

    df.outliers = (df.task_outliers .+ df.survey_outliers) .> 0
end
export ExcIndicatorExp2!



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: General model selection jointly over the two experiments
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBasicInfvsEmplvsNa_E2(Xs, Xinds, Nas, as, Ginds; 
                                            N_rooms = 12, K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)
    
    # model
    m ~ Categorical(6)
    
    # Na parameters (m=2)
    βaG2 ~ truncated(Normal(0.0, 10), 0, Inf)
    βaB2 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Na-Empl parameters (m=3)
    βaG3 ~ truncated(Normal(0.0, 10), 0, Inf)
    lB3 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γB3 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βB3 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Empl-Na parameters (m=4)
    lG4 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γG4 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βG4 ~ truncated(Normal(0.0, 10), 0, Inf)
    βaB4 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Empl-Emp-l parameters (m = 5)
    lG5 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γG5 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βG5 ~ truncated(Normal(0.0, 10), 0, Inf)
    lB5 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γB5 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βB5 ~ truncated(Normal(0.0, 10), 0, Inf)

    # unstructured parameters (m = 6)
    βθG ~ truncated(Normal(0.0, 10), 0, Inf)
    θG ~ MvNormal(zeros(N_rooms), 1. * I)
    βθB ~ truncated(Normal(0.0, 10), 0, Inf)
    θB ~ MvNormal(zeros(N_rooms), 1. * I)

    agent = GoldAgent(Vector{typeof(βaG2)}(undef, N_a))
    # Samples
    for t = 1:N_trial
        if m == 1   # random model
            as[t] ~ Categorical(ones(N_a) ./ N_a)
        elseif m == 2   # Na model
            if Ginds[t] == 1    # if it's a gold trial
                gold_pa1_Na!(Nas[t], βaG2, agent)
            else                # if it's a bomb trial
                gold_pa1_Na!(Nas[t], βaB2, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 3   # Na-Empl
            if Ginds[t] == 1    # if it's a gold trial
                gold_pa1_Na!(Nas[t], βaG3, agent)
            else                # if it's a bomb trial
                if K > 1
                    emp_model = emplK(lB3,γB3,K)
                else
                    emp_model = emplK(lB3,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βB3, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 4   # Empl-Na
            if Ginds[t] == 1    # if it's a gold trial
                if K > 1
                    emp_model = emplK(lG4,γG4,K)
                else
                    emp_model = emplK(lG4,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βG4, agent)
            else                # if it's a bomb trial
                gold_pa1_Na!(Nas[t], βaB4, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 5   # Emp-Emp
            if Ginds[t] == 1    # if it's a gold trial
                if K > 1
                    emp_model = emplK(lG5,γG5,K)
                else
                    emp_model = emplK(lG5,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βG5, agent)
            else                # if it's a bomb trial
                if K > 1
                    emp_model = emplK(lB5,γB5,K)
                else
                    emp_model = emplK(lB5,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βB5, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 6
            i1, i2 = Xinds[t]
            if i1 == 1
                v1 = 0
            else
                if Ginds[t] == 1    # if it's a gold trial
                    v1 = βθG * θG[i1]
                else    # if it's a bomb trial
                    v1 = βθB * θB[i1]
                end
            end
            if i2 == 1
                v2 = 0
            else
                if Ginds[t] == 1    # if it's a gold trial
                    v2 = βθG * θG[i2]
                else    # if it's a bomb trial
                    v2 = βθB * θB[i2]
                end
            end
            pa = softmax([v1, v2])
            as[t] ~ Categorical(pa)
        end
    end
end;
export TuringGoldBasicInfvsEmplvsNa_E2

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Turing Inference: Double-l inference
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@model function TuringGoldBasicConditionalInf_E2(Xs,  Nas, as, Ginds, m; 
                                                        K = 1, N_a = 2)
    # Get observation length.
    N_trial = length(Xs)
    
    # Na parameters (m=2)
    βaG2 ~ truncated(Normal(0.0, 10), 0, Inf)
    βaB2 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Na-Empl parameters (m=3)
    βaG3 ~ truncated(Normal(0.0, 10), 0, Inf)
    lB3 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γB3 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βB3 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Empl-Na parameters (m=4)
    lG4 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γG4 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βG4 ~ truncated(Normal(0.0, 10), 0, Inf)
    βaB4 ~ truncated(Normal(0.0, 10), 0, Inf)

    # Empl-Emp-l parameters (m = 5)
    lG5 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γG5 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βG5 ~ truncated(Normal(0.0, 10), 0, Inf)
    lB5 ~ truncated(Normal(0.0, 5.), 0, Inf)
    γB5 ~ truncated(Normal(0.5, 2.), 0, 1.0)
    βB5 ~ truncated(Normal(0.0, 10), 0, Inf)

    agent = GoldAgent(Vector{typeof(βaG2)}(undef, N_a))
    # Samples
    for t = 1:N_trial
        if m == 2   # Na model
            if Ginds[t] == 1    # if it's a gold trial
                gold_pa1_Na!(Nas[t], βaG2, agent)
            else                # if it's a bomb trial
                gold_pa1_Na!(Nas[t], βaB2, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 3   # Na-Empl
            if Ginds[t] == 1    # if it's a gold trial
                gold_pa1_Na!(Nas[t], βaG3, agent)
            else                # if it's a bomb trial
                if K > 1
                    emp_model = emplK(lB3,γB3,K)
                else
                    emp_model = emplK(lB3,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βB3, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 4   # Empl-Na
            if Ginds[t] == 1    # if it's a gold trial
                if K > 1
                    emp_model = emplK(lG4,γG4,K)
                else
                    emp_model = emplK(lG4,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βG4, agent)
            else                # if it's a bomb trial
                gold_pa1_Na!(Nas[t], βaB4, agent)
            end
            as[t] ~ Categorical(agent.pa)
        elseif m == 5   # Emp-Emp
            if Ginds[t] == 1    # if it's a gold trial
                if K > 1
                    emp_model = emplK(lG5,γG5,K)
                else
                    emp_model = emplK(lG5,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βG5, agent)
            else                # if it's a bomb trial
                if K > 1
                    emp_model = emplK(lB5,γB5,K)
                else
                    emp_model = emplK(lB5,1.,1)
                end
                gold_pa1!(Xs[t], emp_model, βB5, agent)
            end
            as[t] ~ Categorical(agent.pa)
        end
    end
end;
export TuringGoldBasicConditionalInf_E2
