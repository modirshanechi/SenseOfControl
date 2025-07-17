# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# pvalue converting
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_pval_string(pval)
    if pval < 1e-4
        return "< 1e-4"
    elseif pval < 1e-3
        return "< 1e-3"
    else
        return string(round(pval, digits = 3))
    end
end
export Func_pval_string


function Func_logBF_string(logBF)
    return string(round(logBF, digits = 2))
end
export Func_logBF_string


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Converting numbers to 2-digit strings (e.g., 1 -> "01" and 21 -> "21")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Num2digitsStr(x)
    if x > 9
        return(string(x))
    else
        return("0" * string(x))
    end
end 
export Num2digitsStr

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Moving Average
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_movmean(x,k;ZeroPadding = false)
    n = length(x)
    y = zeros(n)
    if ZeroPadding
        x = append!(zeros(k),x)
        x = append!(x,zeros(k))
        inds = 0:(2*k)
        for i = 1:n
            y[i] = mean(x[i .+ inds])
        end
    else
        for i = 1:n
            if i <= k
                ind_beg = 1
            else
                ind_beg = i - k
            end
            if n <= i+k
                ind_end = n
            else
                ind_end = i + k
            end
            y[i] = mean(x[ind_beg:ind_end])
        end
    end
    return y
end
export Func_movmean

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# FDR control
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function FDR_control_pval(p_values;FDR=0.1)
    temp = sort(p_values)
    inds = Array(1:length(temp))
    pval_thresh = inds .* FDR / length(temp)
    if sum(temp .< pval_thresh) > 0
        pval_thresh = pval_thresh[findmax(inds[temp .< pval_thresh])[1]]
    else
        pval_thresh = 0
    end
    R0 = p_values .< pval_thresh
    if sum(R0)>0
        argR0 = inds[R0]
    else
        argR0 = []
    end
    return (; R0=R0, argR0=argR0, pval_thresh=pval_thresh)
end
export FDR_control_pval


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Rank function
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Rank(X)
    N = length(X); R = zeros(N)
    for n = 1:N
        R[n] = (N+1)/2 + sum(sign.(-X .+ X[n]))/2
    end
    return R
end
export Rank


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_OneSampleTTest(Y::Array{Float64,1})
    N = length(Y)

    σ_0 = sqrt(mean(Y.^2))
    K_0 = 1; P_0 = Normal(0,σ_0);

    μ_1 = mean(Y); σ_1 = sqrt(mean((Y .- μ_1).^2))
    K_1 = 2; P_1 = Normal(μ_1,σ_1)

    log_p0 = sum(logpdf.(P_0,Y)) - (K_0 * log(N) / 2)
    log_p1 = sum(logpdf.(P_1,Y)) - (K_1 * log(N) / 2)

    logBF = log_p1 - log_p0
end
function BIC_OneSampleTTest(Y1::Array{Float64,1},Y2::Array{Float64,1})
    BIC_OneSampleTTest(Y1 .- Y2)
end
export BIC_OneSampleTTest

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_EqualVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1})
    N_X = length(X)
    N_Y = length(Y)
    N = N_X + N_Y
    
    μ_0 = mean(vcat(X,Y));
    σ_0 = sqrt(mean((vcat(X,Y) .- μ_0).^2));
    K_XY0 = 2; K_X0 = 0; K_Y0 = 0;
    P_X0 = Normal(μ_0,σ_0); P_Y0 = Normal(μ_0,σ_0);

    μ_X1 = mean(X); μ_Y1 = mean(Y);
    σ_1 = sqrt(mean(vcat((X .- μ_X1), (Y .- μ_Y1)).^2));
    K_XY1 = 1; K_X1 = 1; K_Y1 = 1;
    P_X1 = Normal(μ_X1,σ_1); P_Y1 = Normal(μ_Y1,σ_1);

    log_p0 = sum(logpdf.(P_X0,X)) + sum(logpdf.(P_Y0,Y)) -
            (K_X0 * log(N_X) / 2) - (K_Y0 * log(N_Y) / 2) -
            (K_XY0 * log(N_Y + N_X) / 2)

    log_p1 = sum(logpdf.(P_X1,X)) + sum(logpdf.(P_Y1,Y)) -
            (K_X1 * log(N_X) / 2) - (K_Y1 * log(N_Y) / 2) -
            (K_XY1 * log(N_Y + N_X) / 2)

    logBF = log_p1 - log_p0
end
export BIC_EqualVarianceTTest
BIC_UnequalVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1}) = 
    BIC_EqualVarianceTTest(X::Array{Float64,1},Y::Array{Float64,1})
export BIC_UnequalVarianceTTest
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function BIC_CorrelationTest(X::Array{Float64,1},Y::Array{Float64,1})
    if length(X) != length(Y)
        error("X must have the same dimension as Y.")
    end
    N = length(Y)

    #X_0 = ones(N); Y_hat_0 = X_0 * inv(X_0' * X_0) * X_0' * Y;
    Y_hat_0 = mean(Y); Δ_0 = Y .- Y_hat_0
    σ_0 = sqrt(mean(Δ_0.^2))
    K_0 = 2; P_0 = Normal(0,σ_0);

    X_1 = hcat(ones(N),X); Y_hat_1 = X_1 * inv(X_1' * X_1) * X_1' * Y;
    Δ_1 = Y .- Y_hat_1; σ_1 = sqrt(mean(Δ_1.^2))
    K_1 = 3; P_1 = Normal(0,σ_1)

    log_p0 = sum(logpdf.(P_0,Δ_0)) - (K_0 * log(N) / 2)
    log_p1 = sum(logpdf.(P_1,Δ_1)) - (K_1 * log(N) / 2)

    logBF = log_p1 - log_p0
end
export BIC_CorrelationTest

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Effect size
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_Cohen_d(X1,X2)
    μ1 = mean(X1);   μ2 = mean(X2)
    Δμ = μ1 - μ2
    
    n1 = length(X1); n2 = mean(X2)
    σ1 = std(X1);    σ2 = std(X2)
    σ = sqrt( ( (n1 - 1) * σ1^2 + (n2 - 1) * σ2^2 ) / (n1 + n2 - 2)  )

    d = Δμ / σ
    return d
end
function Func_Cohen_d(X)
    μ = mean(X); σ = std(X);
    return μ / σ
end
export Func_Cohen_d

