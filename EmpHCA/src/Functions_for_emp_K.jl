# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for computing categorical empowerments
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# computing categorical empowerment of degree l from transition probabilities
# p[a,s',s] = p(s' | a,s)
@concrete struct emplK
    l
    γ
    K
end
function emplK(;l=2, γ = 0.9, K = 1)
    emplK(l, γ, K)
end
export emplK
function p2empCat(p, emptype::emplK)
    N_a, N_s, _ = size(p)
    Q = zeros(N_a,N_s,emptype.K,N_s);
    V = zeros(N_s,emptype.K,N_s);
    value_comp_Kl!(p, Q, V; γ = emptype.γ);
    V_0 = V[:,1,:]

    sum(V_0 .^ emptype.l, dims = 2)[:]
end
function p2empCat(p, s, emptype::emplK)
    N_a, N_s, _ = size(p)
    Q = zeros(N_a,N_s,emptype.K,N_s);
    V = zeros(N_s,emptype.K,N_s);
    value_comp_Kl!(p, Q, V; γ = emptype.γ);
    V_0 = V[s,1,:]

    sum(V_0 .^ emptype.l)
end
export p2empCat

function p2empCat!(p, emptype::emplK, Q, V)
    value_comp_Kl!(p, Q, V; γ = emptype.γ);
    V_0 = @view V[:,1,:]
    sum(V_0 .^ emptype.l, dims = 2)[:]
end
function p2empCat!(p, s, emptype::emplK, Q, V)
    value_comp_Kl!(p, Q, V; γ = emptype.γ);
    V_0 = @view V[s,1,:]
    sum(v^emptype.l for v in V_0)
end
export p2empCat!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for value computation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# p[a,s',s] = p(s' | a,s)
# Q[a,s ,t] = Q_{t-1}(a,s) for t ∈ [1,...,K]
# V[s,t] = V_{t-1}(s) for t ∈ [1,...,K]
function value_comp_Kl_sg!(p, Q, V, sg; γ = 0.9)
    N_a, N_s, K = size(Q)
    if size(Q,3) != size(V,2)
        error("Q and V must have the same size for K")
    end
    Q .= 0.; V .= 0.;
    # for the case of t = K
    Q[:,:,K] .= @views p[:,sg,:]
    if K > 1
        Q[:,sg,K] .= 0
    end
    for s = 1:N_s
        V[s,K] = @views (findmax(Q[:,s,K])[1])
    end
    # for t < K if K > 1
    if K > 1
        for t = (K-1):(-1):2
            for s = 1:N_s
            if s ≠ sg
                for a = 1:N_a
                    Q[a,s,t] = @views (p[a,sg,s] + γ * (dot(p[a,:,s], V[:,t+1])))
                end
                V[s,t] = @views (findmax(Q[:,s,t])[1])
            end
            end
        end
        t = 1
        for s = 1:N_s
            for a = 1:N_a
                Q[a,s,t] = @views (p[a,sg,s] + γ * (dot(p[a,:,s], V[:,t+1])))
            end
            V[s,t] = @views (findmax(Q[:,s,t])[1])
        end
    end
end
export value_comp_Kl_sg!

# p[a,s',s] = p(s' | a,s)
# Q[a,s ,t,sg] = Q_{t-1}(a,s|sg) for t ∈ [1,...,K]
# V[s,t,sg] = V_{t-1}(s|sg) for t ∈ [1,...,K]
function value_comp_Kl!(p, Q, V; γ = 0.9)
    N_s = size(Q, 2)
    if size(Q,3) != size(V,2)
        error("Q and V must have the same size for K")
    end
    for sg = 1:N_s
        V_sg = @view V[:,:,sg]
        Q_sg = @view Q[:,:,:,sg]
        value_comp_Kl_sg!(p, Q_sg, V_sg, sg; γ = γ)
    end
end
export value_comp_Kl!