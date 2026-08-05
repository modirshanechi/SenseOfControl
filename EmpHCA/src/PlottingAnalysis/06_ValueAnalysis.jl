################################################################################
# Code for analyzing the room values under different models to discover
# potential heuristics linked to the general participants
# ---> Fig S13
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random, Statistics, HypothesisTests
using DataFrames
using CSV
using JLD2

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)


save_name_tag = "Value/"
infdata_path(i) = "MCMC_sub" * string(i) * ".jld2"

Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

# for accelerating the analysis and averaging
thinning = 100

# ----------------------------------------------------------------------
# Room value evaluation
# ----------------------------------------------------------------------
function value_evaluation(chnemp_dfs, chnBD_dfs, valid_rooms,
                        BMS, subjectIDs,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                        thinning = thinning)
    Nsub = length(subjectIDs); N_rooms2 = length(valid_rooms)
    mAll_hat = zeros(Nsub) .* NaN; 
    RoomValsEmpl   = [zeros(N_rooms2) .* NaN for i = 1:Nsub]
    RoomValsBD     = [zeros(N_rooms2) .* NaN for i = 1:Nsub]
    RoomRanksEmpl   = [zeros(N_rooms2) .* NaN for i = 1:Nsub]
    RoomRanksBD     = [zeros(N_rooms2) .* NaN for i = 1:Nsub]

    pmAll = BMS.exp_M
    for i = 1:Nsub
        @show i
        # ----------------------------------------------------------------------
        # inference data
        # ----------------------------------------------------------------------
        chnemp_df = chnemp_dfs[i][1:thinning:end, :]; 
        chnBD_df = chnBD_dfs[i][1:thinning:end, :]; 
        mAll_hat[i] = findmax(pmAll[i,:])[2]
        # ----------------------------------------------------------------------
        # room values for Emp-l
        # ----------------------------------------------------------------------
        βs = chnemp_df.β; ls = exp.(chnemp_df.logl)
        room_values  = zeros(length(ls),N_rooms2)
        room_ranks   = zeros(length(ls),N_rooms2)
        for i_samp = eachindex(ls)
            emp_model_hat = emplK(ls[i_samp],1.,1)
            for i_room = 1:N_rooms2
                proom = Prooms[valid_rooms[i_room]]
                p, StateS, StateSDict, N_s = 
                    gold_env_setup(proom, ΔState, ΔStateDict, Xmax, Ymax)
                room_values[i_samp,i_room] = βs[i_samp] * 
                            p2empCat(p, StateSDict[(0,0)], emp_model_hat)
            end
            room_values[i_samp,:] .-= room_values[i_samp,1]
            room_ranks[i_samp,:] = Rank(room_values[i_samp,:])
        end
        RoomValsEmpl[i] .= mean(room_values, dims = 1)[:]
        RoomRanksEmpl[i] .= mean(room_ranks, dims = 1)[:]

        # ----------------------------------------------------------------------
        # room values for BD
        # ----------------------------------------------------------------------
        βs = chnBD_df.βθ; 
        θs = Matrix(chnBD_df[:,[Symbol("θ[$i]") for i = valid_rooms]]) .* βs
        θs[:,1] .= 0
        RoomValsBD[i] .= mean(θs, dims = 1)[:]
        RoomRanksBD[i] .= mean(Rank(θs[j,:]) for j = 1:size(θs)[1])
    end
    return (;   RoomValsEmpl = RoomValsEmpl, RoomRanksEmpl = RoomRanksEmpl,
                RoomValsBD = RoomValsBD, RoomRanksBD = RoomRanksBD,
                mAll_hat = mAll_hat, pmAll = pmAll)
end

# ----------------------------------------------------------------------
# Room value plotting: Spearman correlations vs. model probability, by-model
# bar plot, and room-rank-difference bar plots
# ----------------------------------------------------------------------
function room_value_plotting(res, valid_rooms, N_rooms, SavePath)
    Nsub = length(res.mAll_hat)
    io_log = IOBuffer()
    pvals = Float64[]

    # ----------------------------------------------------------------------
    # bar plot of the per-subject Spearman correlation, grouped by the
    # best-fitting model (Emp-l, Gen), with a t-test between the two groups
    # ----------------------------------------------------------------------
    spearman_Empl_BD =
        [cor(res.RoomRanksEmpl[i], res.RoomRanksBD[i]) for i = 1:Nsub]

    group_labels = ["Emp-l", "Gen"]
    group_inds = [findall(res.mAll_hat .== m) for m = [3, 4]]
    group_colors = ["#a02384", "white"]
    edge_colors = ["#a02384", "k"]

    Test_result = UnequalVarianceTTest(spearman_Empl_BD[group_inds[1]],
                                        spearman_Empl_BD[group_inds[2]])
    corr_pval = pvalue(Test_result)
    log_test(io_log,
        "Testing: Spearman(Emp-l, Gen) for Emp-l vs Gen participants -- " *
        "Figure: RoomValCorr_byModel", Test_result; pval = corr_pval)
    push!(pvals, corr_pval)

    figure(figsize=(4,5))
    ax = subplot(1,1,1)
    means = [mean(spearman_Empl_BD[inds]) for inds = group_inds]
    sems  = [std(spearman_Empl_BD[inds]) / sqrt(length(inds)) for inds = group_inds]
    ax.bar(1:2, means, yerr = sems, color = group_colors,
                edgecolor = edge_colors, capsize = 3)
    for i_g = 1:2
        xj = i_g .+ 0.15 .* (rand(rng, length(group_inds[i_g])) .- 0.5)
        ax.plot(xj, spearman_Empl_BD[group_inds[i_g]], ".k", alpha = 0.5)
    end
    ax.text(1.5, 0.95, "p=" * Func_pval_string(corr_pval),
                ha = "center", va = "top")
    ax.set_xticks(1:2); ax.set_xticklabels(group_labels)
    ax.set_ylabel("Spearman(Emp-l, Gen)")
    ax.set_ylim([0,1])
    tight_layout()
    savefig(SavePath * "RoomValCorr_byModel.pdf")
    savefig(SavePath * "RoomValCorr_byModel.png")
    savefig(SavePath * "RoomValCorr_byModel.svg")

    # ----------------------------------------------------------------------
    # per-subject difference in room rank (Emp-l model vs BD/Gen model),
    # separately for Emp-l participants and general participants, in
    # paper room order, with a per-room t-test between the two groups
    # ----------------------------------------------------------------------
    rank_diff_persubj =
        [res.RoomRanksBD[i] .- res.RoomRanksEmpl[i] for i = 1:Nsub]

    bar_colors = [group_colors[1], group_colors[2]]
    bar_edgecolors = [edge_colors[1], edge_colors[2]]
    bar_group_labels = ["Emp-l participants", "Gen participants"]
    width = 0.35

    room_pvals = Vector{Union{Missing,Float64}}(missing, N_rooms)
    for i_paper = 1:N_rooms
        i_room = PaperRoomOrder[i_paper]
        j_room = findfirst(valid_rooms .== i_room)
        if j_room !== nothing
            ys1 = [rank_diff_persubj[i][j_room] for i = group_inds[1]]
            ys2 = [rank_diff_persubj[i][j_room] for i = group_inds[2]]
            Test_result = UnequalVarianceTTest(ys1, ys2)
            pval = pvalue(Test_result)
            log_test(io_log,
                "Testing: rank(Gen) - rank(Emp-l) for Emp-l vs Gen " *
                "participants, room $i_room (paper position $i_paper) -- " *
                "Figure: RoomRankDiff", Test_result; pval = pval)
            room_pvals[i_paper] = pval
            push!(pvals, pval)
        end
    end

    fig_specs = [
        ("RoomRankDiff", identity, "rank(Gen) - rank(Emp-l)"),
        # commented out per request: the absolute-value figure is disabled,
        # kept here in case it needs to be re-enabled later
        # ("RoomRankDiff_abs", abs, "|rank(Gen) - rank(Emp-l)|"),
    ]
    for (fig_name, transform_fun, ylab) = fig_specs
        figure(figsize=(10,4))
        ax = subplot(1,1,1)
        first_label = [true, true]
        for i_paper = 1:N_rooms
            i_room = PaperRoomOrder[i_paper]
            j_room = findfirst(valid_rooms .== i_room)
            if j_room !== nothing
                bar_tops = zeros(2)
                for (i_g, g) = enumerate([1, 2])   # Emp-l, Gen participants
                    inds = group_inds[g]
                    ys = [transform_fun(rank_diff_persubj[i][j_room])
                                            for i = inds]
                    x0 = i_paper + (i_g - 1.5) * width
                    lbl = first_label[i_g] ? bar_group_labels[i_g] : nothing
                    sem = std(ys) / sqrt(length(ys))
                    ax.bar(x0, mean(ys), yerr = sem, width = width,
                                color = bar_colors[i_g],
                                edgecolor = bar_edgecolors[i_g], capsize = 3,
                                label = lbl)
                    xj = x0 .+ 0.2 .* width .* (rand(rng, length(ys)) .- 0.5)
                    ax.plot(xj, ys, ".k", alpha = 0.5)
                    bar_tops[i_g] = mean(ys) + sem
                    first_label[i_g] = false
                end
                if fig_name == "RoomRankDiff" && room_pvals[i_paper] !== missing
                    ax.text(i_paper, maximum(bar_tops) + 0.3,
                                "p=" * Func_pval_string(room_pvals[i_paper]),
                                ha = "center", va = "bottom",
                                fontsize = 6, rotation = 90)
                end
            end
        end
        ax.axhline(0, color = "k", linewidth = 0.5)
        ax.set_xticks(1:N_rooms)
        ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
        ax.set_xlim([0,N_rooms+1]);
        ax.set_ylim([-length(valid_rooms),length(valid_rooms)])
        ax.set_xlabel("room (paper order)")
        ax.set_ylabel(ylab)
        ax.legend()
        tight_layout()
        savefig(SavePath * fig_name * ".pdf")
        savefig(SavePath * fig_name * ".png")
        savefig(SavePath * fig_name * ".svg")
    end

    open(SavePath * "stats.txt", "w") do f
        write(f, String(take!(io_log)))
    end
    return pvals
end


# ----------------------------------------------------------------------
# Loop over the three experiments
# ----------------------------------------------------------------------
all_pvals = Float64[]

for i_exp = 1:3
    # ----------------------------------------------------------------------
    # Loading data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec);
    subjectIDs = data.subjectIDs; selDF = data.selDF
    Nsub = length(subjectIDs)

    # ----------------------------------------------------------------------
    # Loading inference data
    # ----------------------------------------------------------------------
    infdata = [load(expspec.fig_path * infdata_path(i_sub)) for i_sub = subjectIDs]
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    
    # ----------------------------------------------------------------------
    # Choosing the valid rooms
    # ----------------------------------------------------------------------
    if i_exp == 1
        valid_rooms = 1:N_rooms
    else
        valid_rooms = valid_rooms_E23
    end
    
    # ----------------------------------------------------------------------
    # Comparing preferences
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        BMS = BMSdata["BMSAll"]
        chnemp_dfs = [d["FITresultsEmpl"].chnemp_df for d = infdata]
        chnBD_dfs = [d["FITresultsBD"].chnBD_df for d = infdata]
        
        res = value_evaluation(chnemp_dfs, chnBD_dfs, valid_rooms, 
                        BMS, subjectIDs,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                        thinning = thinning)
        append!(all_pvals, room_value_plotting(res, valid_rooms, N_rooms,
                        expspec.fig_path * save_name_tag))

    else
        BMS = BMSdata["BMSAll"][1];
        chnemp_dfs = [d["FITresultsEmpl"][1].chnemp_df for d = infdata]
        chnBD_dfs = [d["FITresultsBD"][1].chnBD_df for d = infdata]        

        res_gold = value_evaluation(chnemp_dfs, chnBD_dfs, valid_rooms, 
                        BMS, subjectIDs,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                        thinning = thinning)
        append!(all_pvals, room_value_plotting(res_gold, valid_rooms, N_rooms,
                        expspec.fig_path * save_name_tag * "Gold"))


        BMS = BMSdata["BMSAll"][2]; BMSEmp = BMSdata["BMSEmp"][2]
        chnemp_dfs = [d["FITresultsEmpl"][2].chnemp_df for d = infdata]
        chnBD_dfs = [d["FITresultsBD"][2].chnBD_df for d = infdata]
        
        res_bomb = value_evaluation(chnemp_dfs, chnBD_dfs, valid_rooms, 
                        BMS, subjectIDs,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                        thinning = thinning)
        append!(all_pvals, room_value_plotting(res_bomb, valid_rooms, N_rooms,
                        expspec.fig_path * save_name_tag * "Bomb"))
    end
    close("all")
end

# ----------------------------------------------------------------------
# FDR correction across all Emp-l vs Gen t-tests (correlations + room ranks)
# ----------------------------------------------------------------------
println("--------------------------------")
println("Value analysis (Emp-l vs Gen participants): all p-values")
@show length(all_pvals)
R0, argR0, pval_thresh = FDR_control_pval(all_pvals; FDR = 0.05)
@show pval_thresh
@show sum(R0)
