# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# functions for MCMC based on Klaas Stephan paper
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Prior_MCMC_sampler(rng::AbstractRNG,ϵ::Float64;n_r=4,n_M=60,H0=false,r=-1)
        if H0
                r = ones(n_r) ./ n_r
        elseif r == -1
                P_r = Dirichlet(n_r, ϵ)
                r = rand(rng,P_r)
        end
        M = rand(rng,Categorical(r),n_M)
        return r,M
end
Prior_MCMC_sampler(ϵ::Float64;kwargs...) = Prior_MCMC_sampler(Random.default_rng(),ϵ;kwargs...)
export Prior_MCMC_sampler


function Base_Chain_MCMC_sampler(rng::AbstractRNG,r,M;
                                N_scale = 10, ϵ = 1.0, N_change = 1,
                                counts::Union{Nothing,AbstractVector{<:Integer}} = nothing)
        M_new = copy(M)

        # Refresh only a subset of coordinates.
        selected = randperm(rng, length(M))[1:N_change]

        # r is constant across these draws, so build the sampler once.
        dist_r = Categorical(r)
        for i in selected
        M_new[i] = rand(rng, dist_r)
        end

        # counts, if supplied, must equal [count(==(k), M) for k in eachindex(r)]
        # exactly (e.g. maintained incrementally by the caller) -- passing it in
        # only skips the recount, it does not change the resulting α/r_new.
        counts_used = counts === nothing ?
                [count(==(k), M) for k in eachindex(r)] : counts
        α = ϵ .+ counts_used ./ N_scale
        r_new = rand(rng, Dirichlet(α))

        return r_new, M_new, selected
end
Base_Chain_MCMC_sampler(r,M;kwargs...) = Base_Chain_MCMC_sampler(Random.default_rng(),r,M;kwargs...)
export Base_Chain_MCMC_sampler

# Stable log(exp(a) + exp(b)).
_logaddexp(a, b) =
    a == -Inf ? b :
    b == -Inf ? a :
    max(a, b) + log1p(exp(-abs(a - b)))


function Base_Chain_MCMC_logpdf(r1, M1, r2, M2;
                                N_scale = 10, ϵ = 1.0, N_change = 1,
                                counts1::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    # Coordinates whose labels actually changed.
    changed = M1 .!= M2
    d = count(changed)

    # Impossible to change more coordinates than were selected.
    d > N_change && return -Inf

    # Every changed coordinate had to be selected and draw M2[i] from r1.
    log_Prob_m = sum(
        log(r1[Int(M2[i])])
        for i in eachindex(M1) if changed[i];
        init = 0.0,
    )

    # These selected coordinates redrew their original labels.
    n_redrawn = N_change - d

    if n_redrawn > 0
        # log_coeff[j+1] is the log-sum of probabilities for choosing
        # exactly j unchanged coordinates among those processed so far.
        log_coeff = fill(-Inf, n_redrawn + 1)
        log_coeff[1] = 0.0  # log(1): choose zero coordinates

        # going over unchanged coordinates one by one
        for i in eachindex(M1)
            changed[i] && continue

            # If coordinate i was selected but did not change,
            # it must have drawn M1[i] from Categorical(r1).
            log_p_same = log(r1[Int(M1[i])])

            # Update backwards so coordinate i cannot be selected twice.
            for j in n_redrawn:-1:1
                log_coeff[j + 1] = _logaddexp(
                    log_coeff[j + 1],          # do not select i
                    log_coeff[j] + log_p_same, # select i
                )
            end
        end

        log_Prob_m += log_coeff[n_redrawn + 1]
    end

    # Proposal density for r2 conditional on M1. counts1, if supplied, must
    # equal [count(==(k), M1) for k in eachindex(r1)] exactly.
    counts_used = counts1 === nothing ?
            [count(==(k), M1) for k in eachindex(r1)] : counts1
    α = ϵ .+ counts_used ./ N_scale

    log_Prob_r = logpdf(Dirichlet(α), r2)

    # Omits -log(binomial(N, N_change)), which cancels in the MH ratio.
    return log_Prob_m + log_Prob_r
end
export Base_Chain_MCMC_logpdf


function logL_MCMC(r,M,L_matrix;P_r=Dirichlet(3, 1.),
                    counts::Union{Nothing,AbstractVector{<:Integer}} = nothing)
        Sub_Num = length(M)
        log_prior = log(pdf(P_r,r))
        # counts, if supplied, must equal [count(==(i),M) for i=1:length(r)] exactly.
        counts_used = counts === nothing ?
                [count(==(i),M) for i = 1:length(r)] : counts
        log_M_r = 0.0
        for i in eachindex(r)
                log_M_r += counts_used[i] * log(r[i])
        end
        log_Y_M = 0.0
        for i in 1:Sub_Num
                log_Y_M += L_matrix[i,Int(M[i])]
        end
        return log_prior + log_M_r + log_Y_M
end
export logL_MCMC

function MCMC_accept_prob(r1,M1,r2,M2,L_matrix;N_scale=10,ϵ=1.,
                                        P_r=Dirichlet(3, 1),N_change=1,
                                        counts1::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                        counts2::Union{Nothing,AbstractVector{<:Integer}} = nothing)
        log_π1 = logL_MCMC(r1,M1,L_matrix;P_r=P_r,counts=counts1)
        log_π2 = logL_MCMC(r2,M2,L_matrix;P_r=P_r,counts=counts2)
        log_ϕ12 = Base_Chain_MCMC_logpdf(r1,M1,r2,M2;
                                        N_scale=N_scale,ϵ=ϵ,N_change=N_change,counts1=counts1)
        log_ϕ21 = Base_Chain_MCMC_logpdf(r2,M2,r1,M1;
                                        N_scale=N_scale,ϵ=ϵ,N_change=N_change,counts1=counts2)
        log_a12 = log_π2 + log_ϕ21 - log_π1 - log_ϕ12
        return min(1,exp(log_a12))
end
export MCMC_accept_prob

# Chains run in parallel via Threads.@threads; launch Julia with multiple
# threads (e.g. `julia --threads=4`) to actually get a speedup.
function MCMC_BMS(rng::AbstractRNG,L_matrix;
                        N_Sampling = Int(1e5), N_Chains = 5, α = 1.,
                        N_scale=1., ϵ=1., N_change=1, uniform_initial=false,
                        M_capture_iters::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                        M_preview_subjects::Int = 0,
                        ifverbose = false)
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        # M_capture_iters: absolute (1:N_Sampling) sample indices at which to
        # store M (in addition to R/π, which are always stored in full --
        # they are cheap, only M is the large array). `nothing` captures every
        # iteration, reproducing the original full-storage behavior exactly.
        capture_iters = M_capture_iters === nothing ?
                collect(1:N_Sampling) : collect(M_capture_iters)
        N_capture = length(capture_iters)
        capture_pos = zeros(Int, N_Sampling)
        for (j, idx) in enumerate(capture_iters)
                capture_pos[idx] = j
        end
        # Full-length (all N_Sampling iterations), but limited to the first
        # few subjects -- used only for trace-plot diagnostics so the M-trace
        # doesn't need the full Sub_Num-wide array to be kept.
        n_prev = min(M_preview_subjects, Sub_Num)

        R_matrix = zeros(N_Sampling, N_model, N_Chains);
        M_capture_matrix = zeros(N_capture, Sub_Num, N_Chains);
        M_preview = zeros(N_Sampling, n_prev, N_Chains);
        π_matrix = zeros(N_Sampling, N_Chains);

        P_r = Dirichlet(N_model, α)

        M0_star = zeros(size(L_matrix)[1])
        for i = 1:Sub_Num
                M0_star[i] = findmax(L_matrix[i,:])[2]
        end
        r0_star = [count(==(i),M0_star) for i = 1:N_model] ./ Sub_Num

        # Each chain gets its own independent Xoshiro stream (base_seed + i_chain,
        # following the same seed-offset-by-index convention used for per-subject
        # RNGs in 01_SubBySub_inference.jl), so chains can run concurrently under
        # Threads.@threads without racing on a shared rng's mutable state. Drawing
        # base_seed is the only draw taken from the caller's rng here -- it happens
        # before the threaded loop starts, so it's inherently race-free.
        base_seed = rand(rng, UInt64)
        print_lock = ReentrantLock()

        Threads.@threads for i_chain = 1:N_Chains
                rng_chain = Xoshiro(base_seed + i_chain)
                if uniform_initial
                        r0,M0 = Prior_MCMC_sampler(rng_chain,1.;n_r=N_model,n_M=Sub_Num)
                else
                        r0,M0 = Base_Chain_MCMC_sampler(rng_chain,r0_star,M0_star;
                                        N_scale=N_scale,ϵ=ϵ,N_change=N_change)
                end
                # Rolling scratch state for the current sample, carried across
                # iterations in plain variables instead of re-slicing R_matrix/
                # M_matrix each step (each slice used to be a fresh allocation).
                # M is kept as Int (exact, lossless vs. the Float64 M0/M_matrix
                # values, since model labels are small integers) to avoid
                # re-deriving Int(...) on every access.
                r1 = copy(r0)
                M1 = Int.(M0)
                counts1 = [count(==(k), M1) for k = 1:N_model]
                π1 = logL_MCMC(r1,M1,L_matrix;P_r=P_r,counts=counts1)

                R_matrix[1,:,i_chain] = r1
                if capture_pos[1] != 0
                        M_capture_matrix[capture_pos[1],:,i_chain] = M1
                end
                if n_prev > 0
                        M_preview[1,:,i_chain] = @view M1[1:n_prev]
                end
                π_matrix[1,i_chain] = π1

                for i_sample = 2:N_Sampling
                        r2,M2,selected = Base_Chain_MCMC_sampler(rng_chain,r1,M1;
                                        N_scale=N_scale,ϵ=ϵ,N_change=N_change,
                                        counts=counts1)
                        # Incremental histogram update for the proposed M2:
                        # only the `selected` coordinates can differ from M1,
                        # so decrement/increment those bins instead of a full
                        # recount -- exactly equal to a fresh recount of M2.
                        counts2 = copy(counts1)
                        for i in selected
                                counts2[M1[i]] -= 1
                                counts2[M2[i]] += 1
                        end
                        a12 = MCMC_accept_prob(r1,M1,r2,M2,L_matrix;
                                        N_scale=N_scale,ϵ=ϵ,P_r=P_r,N_change=N_change,
                                        counts1=counts1,counts2=counts2)
                        if rand(rng_chain) < a12
                                r1 = r2
                                M1 = M2
                                counts1 = counts2
                                π1 = logL_MCMC(r1,M1,L_matrix;P_r=P_r,counts=counts1)
                        end
                        # else: r1, M1, counts1, π1 all carry forward unchanged,
                        # matching the original rejection branch exactly.

                        R_matrix[i_sample,:,i_chain] = r1
                        if capture_pos[i_sample] != 0
                                M_capture_matrix[capture_pos[i_sample],:,i_chain] = M1
                        end
                        if n_prev > 0
                                M_preview[i_sample,:,i_chain] = @view M1[1:n_prev]
                        end
                        π_matrix[i_sample,i_chain] = π1
                end
                if ifverbose 
                        lock(print_lock) do
                                println("chain $i_chain / $N_Chains done")
                        end
                end
        end
        return R_matrix, M_capture_matrix, π_matrix, M_preview, capture_iters
end
MCMC_BMS(L_matrix; kwargs...) = MCMC_BMS(Random.default_rng(),L_matrix;kwargs...)
export MCMC_BMS


function Sampling_H0_H1_rOnly(rng::AbstractRNG,L_matrix,α; N_Sampling = Int(1e5))
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        logP0 = MarginalProb_XgivenR(L_matrix, ones(N_model) ./ N_model)[1]

        R1_matrix = zeros(N_Sampling, N_model);
        π1_matrix = zeros(N_Sampling);
        for i_sample = 1:N_Sampling
                r1,M1 = Prior_MCMC_sampler(rng,α;n_r=N_model,n_M=Sub_Num)
                R1_matrix[i_sample,:] = r1
                π1_matrix[i_sample] = MarginalProb_XgivenR(L_matrix,r1)[1]
        end
        π1_max = findmax(π1_matrix)[1]
        w = exp.(π1_matrix .- π1_max)
        logP1 = π1_max + log(sum(w)) - log(N_Sampling)

        # Monte Carlo error of logP1: delta method on the i.i.d.-prior-sample
        # average defining logP1, i.e. SE(log mean(w)) ≈ SE(mean(w))/mean(w),
        # the coefficient of variation of w scaled by 1/sqrt(N). This is the
        # no-autocorrelation, no-bridge special case of the relative-error
        # estimate in bs_error_percent (Functions_bridgesampling.jl, eq. 17
        # of Gronau et al. 2017): only the i.i.d.-proposal term applies here.
        d_logP1 = std(w) / (mean(w) * sqrt(N_Sampling))

        BOR = 1 / (1 + exp(logP1 - logP0))
        # delta method: d(BOR)/d(logP1) = -BOR*(1-BOR); logP0 is exact (no MC error)
        d_BOR = BOR * (1 - BOR) * d_logP1

        return (; BOR=BOR, d_BOR=d_BOR, logP0=logP0, logP1=logP1, d_logP1=d_logP1,
                  R1_matrix=R1_matrix, π1_matrix=π1_matrix)
end
Sampling_H0_H1_rOnly(L_matrix,α;kwargs...) = Sampling_H0_H1_rOnly(Random.default_rng(),L_matrix,α;kwargs...)
export Sampling_H0_H1_rOnly

function MarginalProb_XgivenR(L_matrix,r)
        Sub_Num = size(L_matrix)[1]
        L_max_vec = [findmax(L_matrix[n,:])[1] for n = 1:Sub_Num]
        L_xr_vec = [(L_max_vec[n] +
                    log(sum(r .* exp.(L_matrix[n,:] .- L_max_vec[n]))))
                    for n = 1:Sub_Num]
        return sum(L_xr_vec), L_xr_vec
end
export MarginalProb_XgivenR


function Sampling_H0_H1(rng::AbstractRNG,L_matrix,α; N_Sampling = Int(1e5))
        N_model = size(L_matrix)[2]
        Sub_Num = size(L_matrix)[1]

        R0_matrix = zeros(N_Sampling, N_model);
        M0_matrix = zeros(N_Sampling, Sub_Num);
        π0_matrix = zeros(N_Sampling);

        R1_matrix = zeros(N_Sampling, N_model);
        M1_matrix = zeros(N_Sampling, Sub_Num);
        π1_matrix = zeros(N_Sampling);

        for i_sample = 1:N_Sampling
                r0,M0 = Prior_MCMC_sampler(rng,α;n_r=N_model,n_M=Sub_Num,H0=true)
                R0_matrix[i_sample,:] = r0
                M0_matrix[i_sample,:] = M0
                π0_matrix[i_sample] = sum([L_matrix[i,Int64(M0[i])] for i=1:Sub_Num])

                r1,M1 = Prior_MCMC_sampler(rng,α;n_r=N_model,n_M=Sub_Num,H0=false)
                R1_matrix[i_sample,:] = r1
                M1_matrix[i_sample,:] = M1
                π1_matrix[i_sample] = sum([L_matrix[i,Int64(M1[i])] for i=1:Sub_Num])
        end
        π0_max = findmax(π0_matrix)[1]
        logP0 = π0_max + log(sum(exp.(π0_matrix .- π0_max))) - log(N_Sampling)
        π1_max = findmax(π1_matrix)[1]
        logP1 = π1_max + log(sum(exp.(π1_matrix .- π1_max))) - log(N_Sampling)

        BOR = 1 / (1 + exp(logP1 - logP0))

        return BOR, logP0, logP1, R0_matrix, M0_matrix, π0_matrix,
               R1_matrix, M1_matrix, π1_matrix
end
Sampling_H0_H1(L_matrix,α;kwargs...) = Sampling_H0_H1(Random.default_rng(),L_matrix,α;kwargs...)
export Sampling_H0_H1


function MCMC_BMS_Statistics(R_matrix_samples,M_matrix_samples,BOR)
        R_samples_all = zeros(size(R_matrix_samples)[1]*size(R_matrix_samples)[3],
                              size(R_matrix_samples)[2]);
        for i = 1:size(R_matrix_samples)[2]
                R_samples_all[:,i] = R_matrix_samples[:,i,:][:]
        end
        exp_r = mean(R_samples_all, dims = 1)[:]
        d_exp_r = std(R_samples_all, dims = 1)[:]

        Best_model_samples = zeros(size(R_samples_all));
        for i = 1:size(R_samples_all)[1]
                Best_model_samples[i,findmax(R_samples_all[i,:])[2]] = 1
        end
        xp = mean(Best_model_samples, dims = 1)[:]

        pxp = xp .* (1 - BOR) .+ ones(length(xp)) ./ length(xp) .* BOR

        M_samples_all = zeros(size(M_matrix_samples)[1]*size(M_matrix_samples)[3],
                              size(M_matrix_samples)[2]);
        for i = 1:size(M_matrix_samples)[2]
                M_samples_all[:,i] = M_matrix_samples[:,i,:][:]
        end
        exp_M = zeros(size(M_samples_all)[2],size(R_matrix_samples)[2])
        for i = 1:size(R_matrix_samples)[2]
                exp_M[:,i] = mean(M_samples_all .== i,dims=1)[:]
        end
        return R_samples_all, M_samples_all, exp_r, d_exp_r, xp, pxp, exp_M
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# MCMC convergence diagnostics (R-hat/ESS) for the r- and M-chains
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# M is a categorical (model-index) variable, so it is diagnosed via its
# one-hot indicator variables I_{M_i=1}, I_{M_i=2}, ... (columns named
# M_i_eq_1, M_i_eq_2, ...) rather than on the raw integer codes; see
# expand_discrete_indicators/MCMC_convergence_diagnostics in
# Functions_TuringModels.jl, reused here as-is.
function MCMC_RandEffects_Diagnostics(R_matrix, M_matrix;
                                       max_rhat = 1.01, min_ess_per_chain = 100)
        N_model = size(R_matrix,2)
        Sub_Num = size(M_matrix,2)

        r_chn  = Chains(R_matrix, [Symbol("r_$i") for i = 1:N_model])
        r_diagnostics = MCMC_convergence_diagnostics(r_chn;
                        max_rhat=max_rhat, min_ess_per_chain=min_ess_per_chain)

        M_chn  = Chains(M_matrix, [Symbol("M_$i") for i = 1:Sub_Num])
        discrete_parameters = Dict(Symbol("M_$i") => N_model for i = 1:Sub_Num)
        M_diagnostics = MCMC_convergence_diagnostics(M_chn;
                        max_rhat=max_rhat, min_ess_per_chain=min_ess_per_chain,
                        discrete_parameters=discrete_parameters)

        return (; r_diagnostics = r_diagnostics, M_diagnostics = M_diagnostics)
end
export MCMC_RandEffects_Diagnostics

function MCMC_BMS_Statistics(rng::AbstractRNG,L_matrix; N_Sampling = Int(1e5),
                N_Sampling_BOR = Int(1e5),
                N_Chains = 40, α = 1., N_scale=1., ϵ=1.,
                N_thin = 50, N_burn_in = 1000,
                uniform_initial=false, test_plotting = true,
                run_diagnostics = true, max_rhat = 1.01, min_ess_per_chain = 100,
                N_diagnostics_max = 1000,
                N_change = -1, ratio_change = 0.05,
                ifverbose = false)
        Sub_Num = size(L_matrix)[1]
        if N_change == -1
                N_change=max(1,Int64(floor(Sub_Num * ratio_change)))
        end

        # Absolute (1:N_Sampling) sample indices kept by the legacy two-step
        # process -- burn-in slice R_matrix[N_burn_in:N_Sampling,:,:], then
        # thin-slice [1:N_thin:(N_Sampling-N_burn_in-1),:,:] -- and,
        # independently, by the diagnostics diag_thin slice. Both are
        # reproduced exactly (same index arithmetic, just evaluated up front
        # in absolute coordinates), so MCMC_BMS only needs to capture M at the
        # iterations either grid actually uses instead of storing every
        # iteration of every chain.
        L_bi = N_Sampling - N_burn_in + 1
        final_abs = N_burn_in .+ (collect(1:N_thin:(N_Sampling-N_burn_in-1)) .- 1)
        if run_diagnostics
                diag_thin = max(1, Int(floor(L_bi / N_diagnostics_max)))
                diag_abs = N_burn_in .+ (collect(1:diag_thin:L_bi) .- 1)
        else
                diag_abs = Int[]
        end
        capture_abs = sort(union(final_abs, diag_abs))
        M_preview_subjects = test_plotting ? min(3, Sub_Num) : 0

        R_matrix, M_capture_matrix, π_matrix, M_preview, capture_iters_used =
                MCMC_BMS(rng,L_matrix;
                        N_Sampling=N_Sampling,N_Chains=N_Chains,α=α,
                        N_scale=N_scale,ϵ=ϵ,N_change=N_change,
                        uniform_initial=uniform_initial,
                        M_capture_iters=capture_abs,
                        M_preview_subjects=M_preview_subjects,
                        ifverbose=ifverbose);
        # capture_iters_used == capture_abs by construction (MCMC_BMS captures
        # exactly the requested indices, in the same order); look up each
        # grid's rows within the single captured M array.
        capture_row = Dict(idx => j for (j,idx) in enumerate(capture_iters_used))

        bor_res = Sampling_H0_H1_rOnly(rng,L_matrix,α; N_Sampling = N_Sampling_BOR);
        BOR, d_BOR = bor_res.BOR, bor_res.d_BOR;
        if ifverbose 
                @show (BOR, d_BOR)
        end

        # R-hat/ESS diagnostics on a thinned subset of the post-burn-in chains.
        # M's one-hot expansion (Sub_Num x N_model indicator columns; see
        # expand_discrete_indicators) makes diagnosing the full un-thinned
        # chain prohibitively slow/memory-heavy for realistic Sub_Num/
        # N_Sampling/N_Chains, so we cap the diagnostics input size instead.
        if run_diagnostics
                R_diag = R_matrix[diag_abs,:,:]
                M_diag = M_capture_matrix[[capture_row[i] for i in diag_abs],:,:]
                r_diagnostics, M_diagnostics =
                        MCMC_RandEffects_Diagnostics(R_diag, M_diag;
                                max_rhat=max_rhat, min_ess_per_chain=min_ess_per_chain)
        else
                r_diagnostics, M_diagnostics = nothing, nothing
        end

        # thinning
        R_matrix_samples = R_matrix[final_abs,:,:];
        M_matrix_samples = M_capture_matrix[[capture_row[i] for i in final_abs],:,:];
        π_matrix_samples = π_matrix[final_abs,:];

        # Testing
        if test_plotting
                N_model = size(L_matrix)[2]
                figure(figsize = (12,8))
                ax = subplot(2,2,1)
                ax.plot(π_matrix[N_burn_in:N_Sampling,:])
                ax = subplot(2,2,2)
                ax.plot(M_preview[N_burn_in:min(N_burn_in+999,N_Sampling),:,1])
                for i = 1:N_model
                        ax = subplot(2,2,3)
                        ax.hist(R_matrix_samples[:,i,:][:])
                        ax = subplot(2,2,4)
                        ax.plot(R_matrix_samples[:,i,1])
                end

                # π and R trace figure: 1 panel for π_matrix, 1 panel per model for R_matrix[:,i,:]
                figure(figsize = (4*(1+N_model), 4))
                ax = subplot(1, 1+N_model, 1)
                ax.plot(π_matrix[N_burn_in:N_Sampling,:])
                ax.set_title("π_matrix")
                for i = 1:N_model
                        ax = subplot(1, 1+N_model, i+1)
                        ax.plot(R_matrix[N_burn_in:N_Sampling,i,:])
                        ax.set_title("R_matrix[:,$i,:]")
                end
                tight_layout()

                # M trace figure: 1 panel per subject (first few subjects only)
                N_sub_plot = size(M_preview, 2)
                figure(figsize = (4*N_sub_plot, 4))
                for i = 1:N_sub_plot
                        ax = subplot(1, N_sub_plot, i)
                        ax.plot(M_preview[N_burn_in:N_Sampling,i,:])
                        ax.set_title("M_matrix[:,$i,:]")
                end
                tight_layout()
        end
        R_samples_all, M_samples_all, exp_r, d_exp_r, xp, pxp, exp_M =
                MCMC_BMS_Statistics(R_matrix_samples,M_matrix_samples,BOR)
        return (; 
                # R_matrix_samples = R_matrix_samples,
                # M_matrix_samples = M_matrix_samples,
                R_samples_all = R_samples_all,
                M_samples_all = M_samples_all,
                π_matrix_samples = π_matrix_samples,
               exp_r = exp_r, d_exp_r = d_exp_r,
               xp = xp, pxp = pxp, exp_M = exp_M, BOR = BOR, d_BOR = d_BOR,
               r_diagnostics = r_diagnostics, M_diagnostics = M_diagnostics,)

end
MCMC_BMS_Statistics(L_matrix;kwargs...) = MCMC_BMS_Statistics(Random.default_rng(),L_matrix;kwargs...)
export MCMC_BMS_Statistics


