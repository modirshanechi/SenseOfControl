################################################################################
# Bridge sampling for marginal-likelihood estimation of Turing.jl models.
#
# Main implementation adapted from:
#   https://github.com/sqwayer/BridgeSampling.jl
#
# Bridge estimator and error-measure formulas based on:
#   Gronau et al. (2017), A Tutorial on Bridge Sampling
#   https://doi.org/10.1016/j.jmp.2017.09.005
#
# Cross-checked against:
#   https://github.com/quentingronau/bridgesampling
#
# Differences from the reference implementations include numerically stable
# bridge-function evaluation and a chain-aware error estimate for Equation 17,
# whose posterior-side term is delegated to MCMCDiagnosticTools.jl' Monte Carlo
# standard error estimator to account for MCMC autocorrelation.
#
# Translation and implementation of this code was assisted by Claude Sonnet 5 
# Chat GPT 5.6 Sol, under close supervision of the corresponding author; all 
# operations were tracked back and checked.
################################################################################

# --- extract posterior samples as a (n_params x n_samples) matrix of values ---

# univariate parameter: MCMCChains stores it as one (n_iter x n_chains) array
bs_flatten(V::AbstractArray) = vec(V)

# multivariate parameter (e.g. a vector-valued prior): MCMCChains stores it as
# one (n_iter x n_chains) array per component; recombine into per-draw vectors
function bs_flatten(V::Tuple) # multivariate parameter: one array per component
    S = hcat([vec(v) for v in V]...)   # (n_draws x n_components)
    return [S[i, :] for i in 1:size(S, 1)]  # one vector of components per draw
end

# build the (n_params x n_draws) sample matrix, one row per named latent variable
function bs_samples(chn::Chains, mdl)
    pnames = collect(DynamicPPL.syms(DynamicPPL.VarInfo(mdl)))  # names of latent params (excludes observed data)
    params = MCMCChains.get_params(chn)
    cols = [bs_flatten(params[pn]) for pn in pnames]  # one flattened draw-vector per parameter
    samples = Matrix{Any}(undef, length(pnames), length(cols[1]))
    for (i, c) in enumerate(cols)
        samples[i, :] = c
    end
    return samples, pnames
end

# --- unnormalized log posterior (log joint) as a function of a parameter vector ---
function bs_log_posterior(mdl, pnames)
    vi0 = DynamicPPL.VarInfo(mdl)
    vns = [@varname($pn) for pn in pnames]  # VarName for each parameter symbol
    dists = [DynamicPPL.getdist(vi0, vn) for vn in vns]     # each parameter's prior distribution
    function logpost(vals)
        vi = vi0
        for (vn, val) in zip(vns, vals)
            vi = DynamicPPL.setindex!!(vi, val, vn)  # write the candidate values into the VarInfo
        end
        return DynamicPPL.logjoint(mdl, vi)  # log prior + log likelihood at these values
    end
    return logpost, dists
end

# --- transform to/from unconstrained space using each parameter's prior bijector ---
bs_link(x, dist) = Bijectors.link(dist, x)        # constrained -> unconstrained
bs_invlink(x, dist) = Bijectors.invlink(dist, x)   # unconstrained -> constrained
# log|d(constrained)/d(unconstrained)| evaluated at the unconstrained value y,
# needed to turn the (constrained-space) log posterior into a log density in
# unconstrained space, where the Gaussian proposal below lives
bs_logjac(y, dist) = Bijectors.logabsdetjac(Bijectors.inverse(Bijectors.bijector(dist)), y)

# apply f (bs_link or bs_invlink) elementwise to every parameter of every sample
function bs_transform_matrix(f, samples::AbstractMatrix, dists)
    nd, ns = size(samples)
    out = Matrix{Any}(undef, nd, ns)
    for j in 1:ns, i in 1:nd
        out[i, j] = f(samples[i, j], dists[i])
    end
    return out
end

# --- widen/narrow: flatten multivariate parameters into a plain real matrix ---

# turn the (n_params x n_draws) matrix (entries can be scalars or vectors) into
# a plain (n_dims x n_draws) real matrix suitable for fitting an MvNormal, plus
# the index range each parameter occupies in the widened rows
function bs_widen(samples::AbstractMatrix)
    nd, ns = size(samples)
    lens = [length(samples[i, 1]) for i in 1:nd]  # dimensionality of each parameter
    ranges = Vector{UnitRange{Int}}(undef, nd)
    o = 0
    for i in 1:nd
        ranges[i] = (o+1):(o+lens[i])
        o += lens[i]
    end
    wide = zeros(o, ns)
    for j in 1:ns, i in 1:nd
        wide[ranges[i], j] .= vcat(samples[i, j])  # vcat: works for both scalars and vectors
    end
    return wide, ranges
end

# inverse of bs_widen: collapse the widened rows back into one entry per
# parameter (scalar if 1-D, vector otherwise), given the ranges from bs_widen
function bs_narrow(wide::AbstractMatrix, ranges)
    nd = length(ranges)
    ns = size(wide, 2)
    samples = Matrix{Any}(undef, nd, ns)
    for j in 1:ns, i in 1:nd
        samples[i, j] = length(ranges[i]) == 1 ? wide[first(ranges[i]), j] : wide[ranges[i], j]
    end
    return samples
end

# --- Gaussian proposal fitted on the (unconstrained, widened) samples ---
bs_proposal(wide::AbstractMatrix) = MvNormal(vec(mean(wide, dims=2)), cov(wide'))

# unnormalized log posterior of the unconstrained variable = log_posterior(constrained) + log|Jacobian|
function bs_pdf_samples(compact, dists, logpost)
    n = size(compact, 2)
    p = zeros(n)
    for i in 1:n
        x = compact[:, i]  # constrained-space parameter vector for draw i
        # sum of per-parameter log-Jacobian terms (change of variables to unconstrained space)
        logdetJ = sum(bs_logjac(bs_link(x[k], dists[k]), dists[k]) for k in eachindex(dists))
        p[i] = logpost(x) + logdetJ
    end
    return p
end

# --- iterative bridge sampling estimator ---
#
# The general bridge sampling identity (eq. 11) says that for ANY bridge
# function h with suitable support,
#     p(y) = E_g[ q1(theta2) h(theta2) ] / E_f[ q2(theta1) h(theta1) ]
# where q1(theta) = prior(theta) * likelihood(theta | y) is the unnormalized
# posterior, q2 = g is the proposal density, theta2 ~ g (n2 draws) and
# theta1 ~ posterior f (n1 draws).
#
# The *optimal* bridge function (Meng & Wong, 1996; eq. 13 here) is
#     h_opt(theta) ∝ 1 / ( s1 * q1(theta) + s2 * p(y) * g(theta) ),
#     s1 = n1/(n1+n2),  s2 = n2/(n1+n2),
# which depends on the unknown p(y) itself. Substituting h_opt into the
# identity gives the self-consistent (fixed-point) update, eq. 15:

# Bridge sampling aims to solve the fixed-point of the recursive update (eq. 15):
#
#   p(y)^(t+1) =  [ (1/n2) sum_{i=1}^{n2} l2_i / (s1*l2_i + s2*p(y)^(t)) ]
#              /  [ (1/n1) sum_{j=1}^{n1} 1    / (s1*l1_j + s2*p(y)^(t)) ]
# where l2_i = q1(theta2_i)/g(theta2_i) for the n2 PROPOSAL draws theta2_i,
# and   l1_j = q1(theta1_j)/g(theta1_j) for the n1 POSTERIOR draws theta1_j 
# with
#   - q1(theta) = prior(theta) * likelihood(theta | y)
#   - q2 = g
#   - theta2 ~ g (n2 draws)
#   - theta1 ~ posterior f (n1 draws)
#
# `l1`, `l2` below are log(l1_j)/log(l2_i); everything is recentred by
# `lstar` purely for numerical stability, and `r = p(y)/exp(lstar)` is what's
# actually iterated (so `logml = log(r) + lstar` recovers eq. 15's p(y)).
function bs_iterate(l1, l2, n1, n2; tol=1e-10, maxiter=1000, numprec = 1e100)
    lstar = median(l1)   # recentering constant for numerical stability
    r = 0.0              # current marginal-likelihood *ratio* estimate (r = exp(logml - lstar))
    s1 = n1 / (n1 + n2)
    s2 = n2 / (n1 + n2)
    logml = -Inf
    el1 = clamp.(exp.(l1 .- lstar), 0.0, numprec)      # l1_j, j = 1..n1 (posterior draws)
    el2 = clamp.(exp.(l2 .- lstar), 1/numprec, numprec)   # l2_i, i = 1..n2 (proposal draws)
    for i in 1:maxiter
        numterm = sum(el2 ./ (s1 .* el2 .+ s2 * r)) / n2    # eq. 15 numerator: sum over all n2 proposal draws
        denomterm = sum(1 ./ (s1 .* el1 .+ s2 * r)) / n1    # eq. 15 denominator: sum over all n1 posterior draws
        rnew = numterm / denomterm
        logmlnew = log(rnew) + lstar
        # WARNING
        # eps = abs((logmlnew - logml) / logmlnew)   # relative change since last iteration
        eps = abs((rnew - r) / rnew)   # relative change since last iteration
        logml, r = logmlnew, rnew
        eps < tol && return logml, i   # converged
    end
    @warn "Bridge sampling: maximum number of iterations ($maxiter) reached before convergence under tolerance $tol"
    return logml, maxiter
end


# Approximate percentage coefficient of variation of the bridge-sampling
# estimate:
#
#   error = 100 * sqrt(RE²).
#
# The proposal contribution is estimated from the ordinary variance of the
# independently generated proposal draws. The posterior contribution is
# estimated from the Monte Carlo standard error of the mean of f2
#   --> implementing eq. 17.
#
# `nchains` specifies the number of independent posterior MCMC chains.
# The entries of p1 and g1 must be grouped by chain:
#
#   chain 1 draws, chain 2 draws, ..., chain nchains draws.
#
# MCMCDiagnosticTools expects the posterior values in a
#
#   draws × chains
#
# matrix, so the grouped vector of f2 values is reshaped before calculating
# its Monte Carlo standard error.
function bs_error_percent(logml, p1, g1, p2, g2, n1, n2; nchains=1)
    length(p1) == n1 && length(g1) == n1 ||
        throw(DimensionMismatch(
            "p1 and g1 must each contain n1 values"
        ))

    length(p2) == n2 && length(g2) == n2 ||
        throw(DimensionMismatch(
            "p2 and g2 must each contain n2 values"
        ))

    nchains >= 1 ||
        throw(ArgumentError(
            "nchains must be positive"
        ))

    n1 % nchains == 0 ||
        throw(DimensionMismatch(
            "each posterior chain must contribute the same number of draws"
        ))

    s1 = n1 / (n1 + n2)
    s2 = n2 / (n1 + n2)

    # Algebraically stable bridge-function values:
    #
    #   f1 = p_post / (s1*p_post + s2*g)
    #   f2 = g      / (s1*p_post + s2*g).
    #
    # Cancelling a common exponential factor avoids explicitly evaluating
    # potentially enormous posterior and proposal densities.
    f1 = 1 ./ (s1 .+ s2 .* exp.(logml .+ g2 .- p2))
    f2 = 1 ./ (s1 .* exp.(p1 .- logml .- g1) .+ s2)

    mean_f1 = mean(f1)
    mean_f2 = mean(f2)

    isfinite(mean_f1) && mean_f1 > 0 ||
        throw(ErrorException(
            "the mean of f1 must be finite and positive"
        ))

    isfinite(mean_f2) && mean_f2 > 0 ||
        throw(ErrorException(
            "the mean of f2 must be finite and positive"
        ))

    # Proposal draws are independent, so
    #
    #   Var(mean(f1)) = Var(f1)/n2.
    proposal_term =
        var(f1) / (n2 * mean_f1^2)

    # The vector is grouped by chain. Since Julia uses column-major storage,
    # reshaping it produces a matrix whose columns correspond to chains.
    draws_per_chain = n1 ÷ nchains

    f2_by_chain =
        reshape(f2, draws_per_chain, nchains)

    # MCSE(mean(f2)) estimates
    #
    #   sqrt(Var(mean(f2))).
    #
    # `kind=mean` requests the ordinary, untransformed MCSE of the mean. This
    # is the quantity needed by the bridge-sampling relative-error formula,
    # rather than a rank-normalized bulk or tail diagnostic.
    mean_f2_mcse =
        mcse(f2_by_chain; kind=mean)

    isfinite(mean_f2_mcse) && mean_f2_mcse >= 0 ||
        throw(ErrorException(
            "failed to estimate the Monte Carlo error of mean(f2)"
        ))

    # Since
    #
    #   MCSE(mean(f2))² ≈ Var(mean(f2)),
    #
    # the posterior relative-error contribution is
    #
    #   Var(mean(f2))/E(f2)².
    posterior_term =
        mean_f2_mcse^2 / mean_f2^2

    re2 = proposal_term + posterior_term

    isfinite(re2) && re2 >= 0 ||
        throw(DomainError(
            re2,
            "estimated relative mean-squared error must be finite and nonnegative",
        ))

    return 100 * sqrt(re2)
end

"""
    bridgesampling(rng::AbstractRNG, chn::Chains, mdl; tol=1e-10, maxiter=1000)

Estimate the log marginal likelihood of a Turing model `mdl` (as passed to
`sample`) from its posterior samples `chn`, using bridge sampling as described
in:

    Gronau, et al. (2017). A tutorial on bridge sampling. 
        Journal of Mathematical Psychology
        https://doi.org/10.1016/j.jmp.2017.09.005

Returns `(; logml, error, niter)`, where `logml` is the estimated log
marginal likelihood and `error` is its estimated relative error in percent.
Equation numbers referenced throughout this file are from the published
version of the tutorial above (verified against its PMC full text,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5699790/, and cross-checked against
the reference R implementation, https://github.com/quentingronau/bridgesampling).

`rng` is used to draw samples from the fitted proposal distribution; pass a
seeded RNG for reproducible results.
"""
function bridgesampling(rng::AbstractRNG, chn::Chains, mdl; 
                        tol=1e-10, maxiter=1000, maxerr = 1.)
    samples, pnames = bs_samples(chn, mdl)
    logpost, dists = bs_log_posterior(mdl, pnames)

    # Split each posterior chain into two non-overlapping subsets:
    #
    #   smp1: draws used on the posterior side of the bridge identity;
    #   smp2: draws used to fit the Gaussian proposal.
    #
    # The split is performed separately within each chain. Because `bs_flatten`
    # groups all iterations from chain 1, then chain 2, and so forth, the
    # resulting smp1 values also remain grouped by chain. This ordering allows
    # `bs_error_percent` to reshape f2 into a draws × chains matrix for
    # MCMCDiagnosticTools.
    niter_chains = size(chn, 1)
    nchains = size(chn, 3)

    posterior_iterations = 1:2:niter_chains
    fit_iterations = 2:2:niter_chains

    posterior_indices = Int[]
    fit_indices = Int[]

    for chain in 1:nchains
        offset = (chain - 1) * niter_chains

        append!(
            posterior_indices,
            offset .+ posterior_iterations,
        )

        append!(
            fit_indices,
            offset .+ fit_iterations,
        )
    end

    smp1 = samples[:, posterior_indices]
    smp2 = samples[:, fit_indices]

    n1 = size(smp1, 2)
    n2 = size(smp2, 2)

    # map both halves to unconstrained space and fit the proposal on smp2
    wide1, ranges = bs_widen(bs_transform_matrix(bs_link, smp1, dists))
    wide2, _ = bs_widen(bs_transform_matrix(bs_link, smp2, dists))
    prop_dist = bs_proposal(wide2)

    # draw fresh samples from the proposal, and map them back to constrained
    # space so the model's log posterior can be evaluated on them
    prop_wide = rand(rng, prop_dist, n2)
    prop_compact = bs_transform_matrix(bs_invlink, bs_narrow(prop_wide, ranges), dists)

    # for both sets of draws:
    #   p = log posterior (unconstrained-space)
    #   g = log proposal density
    p1 = bs_pdf_samples(smp1, dists, logpost)
    g1 = [logpdf(prop_dist, wide1[:, i]) for i in 1:n1]
    p2 = bs_pdf_samples(prop_compact, dists, logpost)
    g2 = [logpdf(prop_dist, prop_wide[:, i]) for i in 1:n2]

    logml, niter = bs_iterate(p1 .- g1, p2 .- g2, n1, n2; tol=tol, maxiter=maxiter)
    # Compute the approximate percentage coefficient of variation. `nchains`
    # lets `bs_error_percent` reshape the posterior bridge-function values
    # into a draws x chains matrix and estimate their Monte Carlo standard
    # error without conflating separate MCMC chains.
    err = bs_error_percent(logml, p1, g1, p2, g2, n1,n2; nchains=nchains)
    if err > maxerr
        println("Warning: bridge sampling relative error ($(err)%) exceeds maxerr ($maxerr%)")
    end
    return (; logml=logml, error=err, niter=niter)
end
export bridgesampling
