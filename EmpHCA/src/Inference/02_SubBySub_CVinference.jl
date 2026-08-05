################################################################################
# Code for cross-validated subject-by-subject inference
# --->  it uses parallel processing to run inference for different subjects 
#       across different CPU nodes      
################################################################################
using Distributed

@everywhere begin
        using EmpHCA
        using Random
        using DataFrames
        using CSV
        import StatsPlots
        using LinearAlgebra
        using Turing, MCMCChains, Distributions
        using JLD2
        using AdvancedMH
end

# ----------------------------------------------------------------------
# inference information
# ----------------------------------------------------------------------
n_chains = 10
chain_length = 10000
burn_in_length = 1000
thinning = 1

# ----------------------------------------------------------------------
# train-test split by room-reversal pairs
# ----------------------------------------------------------------------
@everywhere function traintest_split_by_reversal(Xinds, as)
        Xinds_set1 = typeof(Xinds)([]); Xinds_set2 = typeof(Xinds)([]);
        as_set1 = typeof(as)([]);       as_set2 = typeof(as)([]);
        for i = eachindex(Xinds)
                x = Xinds[i]; x_tilde = [x[2],x[1]]
                if x_tilde ∈ Xinds
                        j = findmax([x_temp == x_tilde for
                                                x_temp = Xinds])[2]
                        if j>i
                                push!(Xinds_set1,Xinds[i]);
                                push!(as_set1,as[i]);

                                push!(Xinds_set2,Xinds[j]);
                                push!(as_set2,as[j]);
                        end
                end
        end
        return Xinds_set1, Xinds_set2, as_set1, as_set2
end

# ----------------------------------------------------------------------
# train-test split + unconstrained l-inference + packaging
# ----------------------------------------------------------------------
@everywhere function fit_cv_package(rng, Xinds, as,
                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                n_chains, chain_length, burn_in_length, thinning)
        Xinds_set1, Xinds_set2, as_set1, as_set2 =
                traintest_split_by_reversal(Xinds, as)

        FITresultsEmpl_set1 = EmpL_fit(rng, Xinds_set1, as_set1,
                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                n_chains, chain_length, burn_in_length, thinning)

        FITresultsEmpl_set2 = EmpL_fit(rng, Xinds_set2, as_set2,
                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                n_chains, chain_length, burn_in_length, thinning)

        return (; FITresultsEmpl_CV = [FITresultsEmpl_set1, FITresultsEmpl_set2],
                        as = [as_set1,as_set2],
                        Xinds = [Xinds_set1,Xinds_set2])
end

# ----------------------------------------------------------------------
# per-subject inference
# ----------------------------------------------------------------------
@everywhere function run_subject_inference(i_sub, expspec, selDF,
                Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                n_chains, chain_length, burn_in_length, thinning)
        rng = Xoshiro(2024 + i_sub)

        df = selDF[selDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        as = df.action .+ 1
        
        if expspec.has_gb_split
                Ginds = (df.Gtrials .== 1)
                Binds = (df.Gtrials .== 0)
        end
        
        # ----------------------------------------------------------------------
        # inference
        # ----------------------------------------------------------------------
        if !expspec.has_gb_split
                data_save = fit_cv_package(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)
        else
                data_save_G = fit_cv_package(rng, Xinds[Ginds], as[Ginds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                data_save_B = fit_cv_package(rng, Xinds[Binds], as[Binds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                data_save = [data_save_G, data_save_B]
        end

        # ----------------------------------------------------------------------
        # save results
        # ----------------------------------------------------------------------
        save(expspec.fig_path * "MCMC_sub" * string(i_sub) * "_CV.jld2",
                "data_save", data_save)

        return nothing
end

for i_exp = 1:3
        # ----------------------------------------------------------------------
        # load data
        # ----------------------------------------------------------------------
        expspec = ExperimentSpecification(i_exp)
        data = load_clean_data(expspec)
        ExcDF = data.ExcDF; selDF = data.selDF; subjectIDs = data.subjectIDs

        # ----------------------------------------------------------------------
        # load rooms
        # ----------------------------------------------------------------------
        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

        # ----------------------------------------------------------------------
        # inference loop over subjects
        # ----------------------------------------------------------------------
        pmap(i_sub -> run_subject_inference(i_sub, expspec, selDF,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning),
                subjectIDs)
end