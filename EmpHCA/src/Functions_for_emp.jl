# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for computing categorical empowerments
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# computing categorical empowerment of degree l from transition probabilities
# p[a,s',s] = p(s' | a,s)
@concrete struct empl
    l
end
function empl(;l=2)
    empl(l)
end
export empl
function p2empCat(p, emptype::empl)
    N_s = size(p)[2]
    p_ss = zeros(N_s,N_s)
    for i = 1:N_s
          for j = 1:N_s
                p_ss[i,j] = findmax(p[:,j,i])[1]
          end
    end
    sum(p_ss .^ emptype.l, dims = 2)[:]
end
export p2empCat

function pas2empCat(pas, emptype::empl)
    N_s = size(pas)[2]
    p_s = zeros(N_s)
    for i = 1:N_s
        p_s[i] = findmax(pas[:,i])[1]
    end
    sum(p_s .^ emptype.l)
end
export pas2empCat

function p2empCat(p, s, emptype::empl)
    pas2empCat(pas, emptype)
end

# computing categorical Klyubin-empowerment from transition probabilities
@concrete struct empKly
    thresh
    max_iter
end
function empKly(;thresh=1e-10, max_iter=100)
    empKly(thresh, max_iter)
end
export empKly
function p2empCat(p, emptype::empKly)
    N_s = size(p)[2]
    emp = zeros(N_s)
    for i = 1:N_s
          r,c = blahut_arimoto(p[:,:,i]; thresh=emptype.thresh, 
                            max_iter=emptype.max_iter, pass_all=false)
          emp[i] = c[end]
    end
    return emp
end
function p2empCat(p, s, emptype::empKly)
    r,c = blahut_arimoto(p[:,:,s]; thresh=emptype.thresh, 
                    max_iter=emptype.max_iter, pass_all=false)
    
    return c[end]
end
export p2empCat

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for Dirichlet-based empowerment
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# computing categorical empowerment version 1: Emp = Emp(E[p])
function α2empDir1(α, emptype)
    p2empCat(α2pDirExp(α), emptype)
end
export α2empDir1

# computing categorical empowerment version 2: Emp = E[Emp(p)]
function α2empDir2(α, emptype; N_samp = 10000, pass_samps = false)
    p_set = Vector{Array{Float64}}([])
    emp_set = Vector{Vector{Float64}}([])
    for i = 1:N_samp
        p_samp = α2pDirSamp(α)
        push!(p_set, p_samp)
        push!(emp_set, p2empCat(p_samp,emptype))
    end
    if pass_samps
        return mean(emp_set), emp_set, p_set
    else
        return mean(emp_set)
    end
end
export α2empDir2

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for expected Dirichlet-based empowerment for state-action pairs
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function p2exp_emp(p, emp; ifnormal=false)
    N_a = size(p)[1]; N_s = size(p)[2]
    exp_emp = zeros(N_a, N_s)
    for j = 1:N_s
        for i = 1:N_a
            exp_emp[i,j] = sum(p[i,:,j] .* emp)
        end
        if ifnormal
            exp_emp[:,j] .-= findmax(exp_emp[:,j])[1]
        end
    end
    return exp_emp
end
export p2exp_emp
function α2exp_emp(α, emp; ifnormal=false)
    p = α2pDirExp(α)
    p2exp_emp(p, emp; ifnormal=ifnormal)
end
export α2exp_emp


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for empowerment gain
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function α2empgain1(α::Array{Float64,3}, s::Int64, a::Int64, sp::Int64, emptype)
    emp_old = α2empDir1(α, emptype)
    α_new = deepcopy(α)
    α_new[a,sp,s] += 1
    emp_new = α2empDir1(α_new, emptype)
    return sum(emp_new .- emp_old)
end
function α2empgain1(α::Array{Float64,3}, s::Int64, a::Int64, emptype;
                    pass_all = false)
    N_s = size(α)[2]
    emp_set = zeros(N_s)
    for sp = 1:N_s
        emp_set[sp] = α2empgain1(α, s, a, sp, emptype)
    end
    p_sp = α2pDirExp(α)[a,:,s]
    if pass_all
        return sum(p_sp .* emp_set), emp_set
    else
        return sum(p_sp .* emp_set)
    end
end
function α2empgain1(α::Array{Float64,3}, s::Int64, emptype;
                    pass_all = false)
    N_a = size(α)[1]
    emp_set = zeros(N_a)
    for a = 1:N_a
        emp_set[a] = α2empgain1(α, s, a, emptype)
    end
    
    if pass_all
        return findmax(emp_set)[1], emp_set
    else
        return findmax(emp_set)[1]
    end
end
function α2empgain1(α::Array{Float64,3}, emptype)
    N_s = size(α)[2]
    emp_set = zeros(N_s)
    for s = 1:N_s
        emp_set[s] = α2empgain1(α, s, emptype)
    end
    return emp_set
end
export α2empgain1

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for information gain
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function α2infgain(α::Array{Float64,3}, s::Int64, a::Int64, sp::Int64, l)
    p_old = α2pDirExp(α)
    α_new = deepcopy(α)
    α_new[a,sp,s] += 1
    p_new = α2pDirExp(α_new)
    return sum((abs.(p_new .- p_old)).^l)
end
function α2infgain(α::Array{Float64,3}, s::Int64, a::Int64, l;
                    pass_all = false)
    N_s = size(α)[2]
    inf_set = zeros(N_s)
    for sp = 1:N_s
        inf_set[sp] = α2infgain(α, s, a, sp, l)
    end
    p_sp = α2pDirExp(α)[a,:,s]
    if pass_all
        return sum(p_sp .* inf_set), inf_set
    else
        return sum(p_sp .* inf_set)
    end
end
function α2infgain(α::Array{Float64,3}, s::Int64, l;
                    pass_all = false)
    N_a = size(α)[1]
    inf_set = zeros(N_a)
    for a = 1:N_a
        inf_set[a] = α2infgain(α, s, a, l)
    end
    
    if pass_all
        return findmax(inf_set)[1], inf_set
    else
        return findmax(inf_set)[1]
    end
end
function α2infgain(α::Array{Float64,3}, l)
    N_s = size(α)[2]
    inf_set = zeros(N_s)
    for s = 1:N_s
        inf_set[s] = α2infgain(α, s, l)
    end
    return inf_set
end
export α2infgain

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Functions for Blahut-Arimoto algorithm
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function cat_capacity(P_YX::Array{Float64,2},
                       r::Array{Float64,1},q::Array{Float64,2})
    N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets fo y (= number of states)
    logr = zeros(N)
    for n = 1:N
        inds = (P_YX[n,:] .!= 0)
        temp1 = P_YX[n,inds]
        temp2 = q[inds, n]
        logr[n] = sum(temp1 .* log.( temp2 ./ r[n] ) )
    end
    c = sum(r .* logr)
end
export cat_capacity

function cat_capacity(P_YX::Array{Float64,2},r::Array{Float64,1})
    N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets fo y (= number of states)

    P_Y = (r' * P_YX)[:]
    H_Y = - sum(P_Y[P_Y .!= 0] .* log.(P_Y[P_Y .!= 0]))
    H_YX = zeros(N)
    for n = 1:N
        P_YX_n = P_YX[n,:]
        H_YX[n] = - sum(P_YX_n[P_YX_n .!= 0] .* log.(P_YX_n[P_YX_n .!= 0]))
    end
    H_YX = sum(r .* H_YX)
    c = H_Y - H_YX
end
export cat_capacity


function blahut_arimoto(P_YX; thresh = 1e-10, max_iter = 100,
                                   pass_all = false)
    N = size(P_YX)[1]   # size of alphabets fo x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets fo y (= number of states)

    r = [ones(N) ./ N]  # Initializiation of r (= prior policy)
    c = [0.]

    qi = zeros(M,N)
    ri = zeros(N)
    for i = 1:max_iter
        for m = 1:M
            qi[m,:] = r[end] .* P_YX[:,m]
            if sum(qi[m,:]) == 0
                qi[m,:] = ones(N) ./ N
            else
                qi[m,:] .= qi[m,:] ./ sum(qi[m,:])
            end
        end

        for n=1:N
            inds = (qi[:, n] .!= 0)
            ri[n] = prod( qi[inds, n] .^ P_YX[n,inds])
        end
        ri[:] = ri[:] ./ sum(ri)

        tolerance = sum((ri - r[end]).^2)
        if pass_all
            push!(r, ri)
            push!(c, cat_capacity(P_YX,ri,qi))
        else
            r[1] = ri
            c[1] = cat_capacity(P_YX,ri,qi)
        end
        if tolerance < thresh
            break
        end
    end
    return r,c
end
export blahut_arimoto
