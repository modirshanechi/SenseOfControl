################################################################################
# Code for non-hierarchical subject-by-subject inference
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
                # --------------------------------------------------------------
                # BradleyTerry inference
                # --------------------------------------------------------------
                FITresultsBD = BradTerry_fit(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                # --------------------------------------------------------------
                # unconstrained l-inference
                # --------------------------------------------------------------
                FITresultsEmpl = EmpL_fit(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                # --------------------------------------------------------------
                # Na inference
                # --------------------------------------------------------------
                FITresultsNa = Na_fit(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)


                # --------------------------------------------------------------
                # model selection: l0, l1, l2
                # --------------------------------------------------------------
                MSresultsEmp = Emp_model_selection(rng, Xinds, as,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                
                
        else
                # --------------------------------------------------------------
                # BradleyTerry inference
                # --------------------------------------------------------------
                FITresultsBDG = BradTerry_fit(rng, Xinds[Ginds], as[Ginds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsBDB = BradTerry_fit(rng, Xinds[Binds], as[Binds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsBD = [FITresultsBDG, FITresultsBDB]

                # --------------------------------------------------------------
                # unconstrained l-inference
                # --------------------------------------------------------------
                FITresultsEmplG = EmpL_fit(rng, Xinds[Ginds], as[Ginds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsEmplB = EmpL_fit(rng, Xinds[Binds], as[Binds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsEmpl = [FITresultsEmplG, FITresultsEmplB]

                # --------------------------------------------------------------
                # Na inference
                # --------------------------------------------------------------
                FITresultsNaG = Na_fit(rng, Xinds[Ginds], as[Ginds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsNaB = Na_fit(rng, Xinds[Binds], as[Binds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                FITresultsNa = [FITresultsNaG, FITresultsNaB]

                # --------------------------------------------------------------
                # model selection: l0, l1, l2
                # --------------------------------------------------------------
                MSresultsEmpG = Emp_model_selection(rng, Xinds[Ginds], as[Ginds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                MSresultsEmpB = Emp_model_selection(rng, Xinds[Binds], as[Binds],
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax,
                        n_chains, chain_length, burn_in_length, thinning)

                MSresultsEmp = [MSresultsEmpG, MSresultsEmpB]
        end
        
        # ----------------------------------------------------------------------
        # save results
        # ----------------------------------------------------------------------
        save(expspec.fig_path * "MCMC_sub" * string(i_sub) * ".jld2",
                "MSresultsEmp", MSresultsEmp, "FITresultsBD", FITresultsBD,
                "FITresultsEmpl", FITresultsEmpl, "FITresultsNa", FITresultsNa)

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