# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for computing categorical empowerments
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# computing categorical Klyubin-empowerment from transition probabilities
@concrete struct empKly
    thresh
    max_iter
end
function empKly(;thresh=1e-16, max_iter=500)
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
# Functions for Blahut-Arimoto algorithm
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function cat_capacity(P_YX::Array{Float64,2},
                       r::Array{Float64,1},q::Array{Float64,2})
    N = size(P_YX)[1]   # size of alphabets of x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets of y (= number of states)
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
    N = size(P_YX)[1]   # size of alphabets of x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets of y (= number of states)

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
    N = size(P_YX)[1]   # size of alphabets of x (= number of actions)
    M = size(P_YX)[2]   # size of alphabets of y (= number of states)

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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Functions for H_max
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# computing categorical Hmax from transition probabilities
@concrete struct Hmax
    tol
end
function Hmax(;tol=0.0)
    Hmax(tol)
end
export Hmax
function p2empCat(p, s, emptype::Hmax)
    return max_output_entropy(p[:,:,s]; 
                    tol= emptype.tol, pass_all=false)
end
export p2empCat

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Functions for the maximum entropy optimization problem
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function max_output_entropy(P_YX::AbstractMatrix{<:Real}; 
                            tol::Float64 = 0.0, pass_all=false)
    P = Matrix{Float64}(P_YX)
    N, M = size(P)

    if any(P .< -tol)
        error("P_YX has negative entries below tolerance.")
    end

    row_sums = vec(sum(P, dims = 2))
    if maximum(abs.(row_sums .- 1.0)) > tol
        error("Each row of P_YX must sum to 1.")
    end

    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, p_x[1:N] >= 0)
    @variable(model, p_y[1:M] >= 0)
    @variable(model, t[1:M])

    @constraint(model, sum(p_x) == 1)
    @constraint(model, [j = 1:M], p_y[j] == sum(P[i, j] * p_x[i] for i in 1:N))
    @constraint(model, [j = 1:M], [t[j], p_y[j], 1.0] in MOI.ExponentialCone())

    @objective(model, Max, sum(t))

    optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL
        error("Solver did not report MOI.OPTIMAL. Status: $status")
    end

    H_max_value = objective_value(model)
    p_x_opt = value.(p_x)
    p_y_opt = value.(p_y)

    if pass_all
        return (; H_max_value = H_max_value,
                p_x = p_x_opt, p_y = p_y_opt,
                status = status, model = model)
    else
        return H_max_value
    end
end