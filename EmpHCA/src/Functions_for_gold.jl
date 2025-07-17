# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Gold collector agent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct GoldAgent
    pa
end
GoldAgent() = GoldAgent([0.5,0.5])
export GoldAgent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Action probabilities
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function gold_pa1(X,emp_model, β)
    emp_vals = β .* [p2empCat!(R.p, R.sloc, emp_model, R.Q, R.V) for R = X]
    softmax(emp_vals)
end
export gold_pa1
function gold_pa1!(X,emp_model, β, agent)
    for i = eachindex(agent.pa)
        agent.pa[i] = β * p2empCat!(X[i].p, X[i].sloc, emp_model, X[i].Q, X[i].V)
    end
    softmax!(agent.pa)
end
export gold_pa1!
function softmax!(x)
    # Step 1: Subtract the maximum value for numerical stability
    max_val = maximum(x)
    for i in eachindex(x)
        x[i] = exp(x[i] - max_val)
    end
    
    # Step 2: Compute the sum of exponentials
    sum_exp = 0.0
    for val in x
        sum_exp += val
    end

    # Step 3: Normalize each element
    for i in eachindex(x)
        x[i] /= sum_exp
    end
end
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Generate room features
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The order of the room in Figure S1 of figure is different from the code
# PaperRoomOrder[i] maps the index of the room i of the paper to that of the code  
const PaperRoomOrder = [
    2, 3, 10, 9,    # rooms in the main text
    1,              # perfect room
    5, 6, 7, 8,     # other rooms in E2
    4, 12, 11       # only in E1
]
export PaperRoomOrder

function gold_proom_sets()
    ΔState = vcat([(i, j) for i = -1:1, j = -1:1]...)
    ΔStateDict = 
        Dict(element => index for (index, element) in enumerate(ΔState))

    Prooms = Vector{Matrix{T} where T}([])
    # ------------------------------------
    # Room 1
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(8,9)
    temp[1,ΔStateDict[(-1,0)]] = 1
    temp[2,ΔStateDict[(0,-1)]] = 1
    temp[3,ΔStateDict[(0,+1)]] = 1
    temp[4,ΔStateDict[(+1,0)]] = 1
    temp[5,ΔStateDict[(-1,-1)]] = 1
    temp[6,ΔStateDict[(+1,-1)]] = 1
    temp[7,ΔStateDict[(+1,+1)]] = 1
    temp[8,ΔStateDict[(-1,+1)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 2
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[(-1,0)]] = 1
    temp[2,ΔStateDict[(0,-1)]] = 1
    temp[3,ΔStateDict[(0,+1)]] = 1
    temp[4,ΔStateDict[(+1,0)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 3
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[(-1,+1)]] = 1
    temp[1,ΔStateDict[(0,+1)]] = 1
    temp[1,ΔStateDict[(+1,+1)]] = 1
    temp[2,ΔStateDict[(-1,0)]] = 1
    temp[3,ΔStateDict[(+1,0)]] = 1
    temp[4,ΔStateDict[(-1,-1)]] = 1
    temp[4,ΔStateDict[(0,-1)]] = 1
    temp[4,ΔStateDict[(+1,-1)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 4
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[1,ΔStateDict[(-1,+1)]] = 1
    temp[2,ΔStateDict[(-1, 0)]] = 1
    temp[3,ΔStateDict[(+1, 0)]] = 1
    temp[4,ΔStateDict[( 0,-1)]] = 1
    temp[4,ΔStateDict[(+1,-1)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 5
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[1,ΔStateDict[(-1, 0)]] = 1
    temp[2,ΔStateDict[( 0,+1)]] = 1
    temp[2,ΔStateDict[(+1, 0)]] = 1
    temp[3,ΔStateDict[( 0,+1)]] = 1
    temp[4,ΔStateDict[( 0,+1)]] = 1
    temp[4,ΔStateDict[(+1, 0)]] = 1
    temp[4,ΔStateDict[(-1, 0)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 6
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[1,ΔStateDict[(+1,+1)]] = 1
    temp[2,ΔStateDict[(-1, 0)]] = 1
    temp[2,ΔStateDict[(-1,+1)]] = 1
    temp[3,ΔStateDict[(+1, 0)]] = 1
    temp[3,ΔStateDict[(+1,-1)]] = 1
    temp[4,ΔStateDict[( 0,-1)]] = 1
    temp[4,ΔStateDict[(-1,-1)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 7
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[1,ΔStateDict[(-1, 0)]] = 1
    temp[1,ΔStateDict[(+1, 0)]] = 1
    temp[2,ΔStateDict[(-1, 0)]] = 1
    temp[3,ΔStateDict[(+1, 0)]] = 1
    temp[4,ΔStateDict[( 0,-1)]] = 1
    temp[4,ΔStateDict[(-1, 0)]] = 1
    temp[4,ΔStateDict[(+1, 0)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 8
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(3,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[2,ΔStateDict[(-1, 0)]] = 1
    temp[3,ΔStateDict[(+1, 0)]] = 1
    for i = 1:3
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 9
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(2,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[1,ΔStateDict[( 0,-1)]] = 1
    temp[1,ΔStateDict[(-1, 0)]] = 1
    temp[1,ΔStateDict[(+1, 0)]] = 1
    temp[2,ΔStateDict[(+1,+1)]] = 1
    temp[2,ΔStateDict[(-1,-1)]] = 1
    temp[2,ΔStateDict[(-1,+1)]] = 1
    temp[2,ΔStateDict[(+1,-1)]] = 1
    for i = 1:2
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 10
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(2,9)
    temp[1,ΔStateDict[(+1, 0)]] = 1
    temp[2,ΔStateDict[(-1, 0)]] = 1
    for i = 1:2
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 11
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(2,9)
    temp[1,ΔStateDict[( 0,+1)]] = 1
    temp[2,ΔStateDict[( 0,-1)]] = 1
    for i = 1:2
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    # ------------------------------------
    # Room 12
    # ------------------------------------
    # p[a,Δs'] = p(Δs'|a)
    temp = zeros(4,9)
    temp[1,ΔStateDict[(-1,-1)]] = 1
    temp[2,ΔStateDict[(+1,-1)]] = 1
    temp[3,ΔStateDict[(+1,+1)]] = 1
    temp[4,ΔStateDict[(-1,+1)]] = 1
    for i = 1:4
        temp[i,:] .= temp[i,:] ./ sum(temp[i,:])
    end
    push!(Prooms,temp)

    return Prooms, ΔState, ΔStateDict
end
export gold_proom_sets

# Making an environment of size (2Xmax+1) × (2Ymax+1) from Proom
function gold_env_setup(Proom, ΔState, ΔStateDict, Xmax, Ymax)
    StateS = vcat([(i, j) for i = (-Xmax):Xmax, j = (-Ymax):Ymax]...)
    StateSDict = 
        Dict(element => index for (index, element) in enumerate(StateS))
    N_a = size(Proom)[1]
    N_s = length(StateS)

    Penv = zeros(N_a, N_s, N_s)
    for a  = 1:N_a
    for s  = 1:N_s      
        for sp = 1:N_s
                ΔS = (StateS[sp] .- StateS[s])
                if (ΔS ∈ ΔState)
                    Penv[a,sp,s] = Proom[a,ΔStateDict[ΔS]]
                end
        end
        if sum(Penv[a,:,s]) == 0
                Penv[a,s,s] = 1.
        else
                Penv[a,:,s] .= Penv[a,:,s] ./ sum(Penv[a,:,s])
        end
    end
    end

    return Penv, StateS, StateSDict, N_s
end
export gold_env_setup

# Making a pairwise comparison X between all combinations of rooms in Prooms
# without repetition
function gold_Room2X_basic(Prooms, ΔState, ΔStateDict, Xmax, Ymax,K)
    Xs = Vector{Vector{NamedTuple{(:p, :sloc, :Q, :V), 
            Tuple{Array{T1,3}, Int, Array{T2,4}, Array{T3,3}}}} where {T1, T2, T3}}([])
    for ip1 = eachindex(Prooms)
    for ip2 = eachindex(Prooms)
        if ip1 != ip2
            p1, dict1 = gold_env_setup(Prooms[ip1], ΔState, ΔStateDict, 
                                    Xmax, Ymax)[[1,3]]
            p2, dict2 = gold_env_setup(Prooms[ip2], ΔState, ΔStateDict, 
                                    Xmax, Ymax)[[1,3]]
            N_a, N_s, _ = size(p1)
            Q1 = zeros(N_a,N_s,K,N_s); V1 = zeros(N_s,K,N_s);
            N_a, N_s, _ = size(p2)
            Q2 = zeros(N_a,N_s,K,N_s); V2 = zeros(N_s,K,N_s);
            X1 = (;p = p1, sloc = dict1[(0,0)], Q=Q1, V=V1)
            X2 = (;p = p2, sloc = dict2[(0,0)], Q=Q2, V=V2)
            push!(Xs, [X1,X2])
        end
    end
    end
    Xs
end
export gold_Room2X_basic

function gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,K)
    Xs = Vector{Vector{NamedTuple{(:p, :sloc, :Q, :V), 
            Tuple{Array{T1,3}, Int, Array{T2,4}, Array{T3,3}}}} where {T1, T2, T3}}([])
    for i = eachindex(Xinds)
        ip1, ip2 = Xinds[i]
        p1, dict1 = gold_env_setup(Prooms[ip1], ΔState, ΔStateDict, 
                                Xmax, Ymax)[[1,3]]
        p2, dict2 = gold_env_setup(Prooms[ip2], ΔState, ΔStateDict, 
                                Xmax, Ymax)[[1,3]]
        N_a, N_s, _ = size(p1)
        Q1 = zeros(N_a,N_s,K,N_s); V1 = zeros(N_s,K,N_s);
        N_a, N_s, _ = size(p2)
        Q2 = zeros(N_a,N_s,K,N_s); V2 = zeros(N_s,K,N_s);
        X1 = (;p = p1, sloc = dict1[(0,0)], Q=Q1, V=V1)
        X2 = (;p = p2, sloc = dict2[(0,0)], Q=Q2, V=V2)
        push!(Xs, [X1,X2])
    end
    Xs
end
export gold_Room2X_indexbased

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Simulator
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Xs[t] = (Xs[t][1] , Xs[t][2])
# Xs[t][i] = (Proom, ΔState, ΔStateDict, Xmax, Ymax)
function gold_simulate(Xs, l, γ, K, β; 
                        agent = GoldAgent(), 
                        rng = Random.default_rng())
    N_trial = length(Xs)

    # empowerment model 
    emp_model = emplK(l,γ,K)
    # simulation
    as = Vector{Int64}([])

    for t = 1:N_trial
        gold_pa1!(Xs[t], emp_model, β, agent)
        push!(as, rand(rng, Categorical(agent.pa)))
    end

    return as
end
export gold_simulate


function gold_pa1_Na!(Na, β, agent)
    for i = eachindex(agent.pa)
        agent.pa[i] = β * Na[i]
    end
    softmax!(agent.pa)
end
function gold_simulate_Nact(Nas, β;
                        agent = GoldAgent(), 
                        rng = Random.default_rng())
    N_trial = length(Nas)
    # simulation
    as = Vector{Int64}([])
    for t = 1:N_trial
        gold_pa1_Na!(Nas[t], β, agent)
        push!(as, rand(rng, Categorical(agent.pa)))
    end
    return as
end
export gold_simulate_Nact

function gold_simulate_General(Xinds, θ;
                        rng = Random.default_rng())
    N_trial = length(Xinds)
    # simulation
    as = Vector{Int64}([])
    for t = 1:N_trial
        i1, i2 = Xinds[t]
        if i1 == 1
            v1 = 0
        else
            v1 = θ[i1]
        end
        if i2 == 1
            v2 = 0
        else
            v2 = θ[i2]
        end
        v = [v1, v2]
        push!(as, rand(rng, Categorical(softmax(v))))
    end
    return as
end
export gold_simulate_General

