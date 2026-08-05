################################################################################
# Code for plotting normalized accuracy rates + parameter consistency
# ---> Fig 2F and G3
# ---> Fig 3E
# ---> Fig 4A2
# ---> Fig 5D and E3
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random, Statistics, HypothesisTests
using DataFrames
using CSV
using JLD2
using NNlib: softmax

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)


save_name_tag = "AccRate/"

Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

# for accelerating the analysis and averaging
thinning = 100

# ----------------------------------------------------------------------
# Accuracy calculation
# ----------------------------------------------------------------------
function p2accuracy(p)
    mean((p .> 0.5) .+ (p .== 0.5) ./ 2)
end

function accuracy_calculation(selDF, BMS, subjectIDs,
                        CV_XindsSets, CV_asSets, CV_chn_dfsSets,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)

    Nsub = length(subjectIDs)
    cons_inds = zeros(Nsub) .* NaN;
    Na_acc = zeros(Nsub) .* NaN;    Na_acc_rel = zeros(Nsub) .* NaN;
    EKly_acc = zeros(Nsub) .* NaN;  EKly_acc_rel = zeros(Nsub) .* NaN;
    Hmax_acc = zeros(Nsub) .* NaN;  Hmax_acc_rel = zeros(Nsub) .* NaN;
    Emp1_acc = zeros(Nsub) .* NaN;  Emp1_acc_rel = zeros(Nsub) .* NaN;
    Empl_acc = zeros(Nsub) .* NaN;  Empl_acc_rel = zeros(Nsub) .* NaN;
    
    mAll_hat = zeros(Nsub) .* NaN; 
    logl_hat = zeros(Nsub, 2) .* NaN; Empl_acc_perfold = zeros(Nsub, 2) .* NaN; 

    for i = 1:Nsub
        @show i
        i_sub = subjectIDs[i]
        
        # ----------------------------------------------------------------------
        # choices
        # ----------------------------------------------------------------------
        df = selDF[selDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]

        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1
        
        # ----------------------------------------------------------------------
        # consistency index
        # ----------------------------------------------------------------------
        cons_inds[i], cons_trials = choice_consistency(Xinds, as_inds;
                                                        ifpassinds = true)
        
        # ----------------------------------------------------------------------
        # Na accuracy
        # ----------------------------------------------------------------------
        p_hats = zeros(length(as_inds))
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([size(Prooms[x[1]])[1], 
                            size(Prooms[x[2]])[1]])
            p_hats[t] = pa[as[t]]
        end
        mp_hatsNa = p_hats[cons_trials]
        Na_acc[i] = p2accuracy(mp_hatsNa)
        Na_acc_rel[i] = Na_acc[i] / cons_inds[i]

        # ----------------------------------------------------------------------
        # Klyubin emp accuracy
        # ----------------------------------------------------------------------
        # evaluating room Klyubin-empowerment values
        eK_vals = zeros(N_rooms); 
        eK_model = empKly(max_iter=1000)
        for i_room = 1:N_rooms
            p, StateS, StateSDict, N_s = 
                gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                    Xmax, Ymax)
            eK_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eK_model)
        end
        # action prob
        pK_hats = zeros(length(as))
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([eK_vals[x[1]], eK_vals[x[2]]])
            pK_hats[t] = pa[as[t]]
        end
        pK_hats = pK_hats[cons_trials]
        EKly_acc[i] = p2accuracy(pK_hats)
        EKly_acc_rel[i] = EKly_acc[i] / cons_inds[i]


        # ----------------------------------------------------------------------
        # Hmax accuracy
        # ----------------------------------------------------------------------
        # evaluating room Hmax values
        Hm_vals = zeros(N_rooms); 
        Hm_model = Hmax(tol=0.0)
        for i_room = 1:N_rooms
            p, StateS, StateSDict, N_s = 
                gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                    Xmax, Ymax)
            Hm_vals[i_room] = p2empCat(p, StateSDict[(0,0)], Hm_model)
        end
        # action prob
        pHm_hats = zeros(length(as))
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([Hm_vals[x[1]], Hm_vals[x[2]]])
            pHm_hats[t] = pa[as[t]]
        end
        pHm_hats = pHm_hats[cons_trials]
        Hmax_acc[i] = p2accuracy(pHm_hats)
        Hmax_acc_rel[i] = Hmax_acc[i] / cons_inds[i]

        # ----------------------------------------------------------------------
        # Emp-1 accuracy
        # ----------------------------------------------------------------------
        # evaluating empowerment-1 values
        eE1_vals = zeros(N_rooms)
        eE1_model = emplK(1.,1.,1)
        for i_room = 1:N_rooms
            p, StateS, StateSDict, N_s = 
                gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                    Xmax, Ymax)
            eE1_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eE1_model)
        end
        # action prob
        pE1_hats = zeros(length(as))
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([eE1_vals[x[1]], eE1_vals[x[2]]])
            pE1_hats[t] = pa[as[t]]
        end
        pE1_hats = pE1_hats[cons_trials]
        Emp1_acc[i] = p2accuracy(pE1_hats)
        Emp1_acc_rel[i] = Emp1_acc[i] / cons_inds[i]

        # ----------------------------------------------------------------------
        # Cross-validated Emp-l accuracy
        # ----------------------------------------------------------------------
        # CV_XindsSets/CV_asSets (built via traintest_split_by_reversal in
        # Inference/02_SubBySub_CVinference.jl) are restricted to the same
        # repeat-reversal trial population as cons_trials/choice_consistency
        # above, so Empl_acc_rel is directly comparable to Na_acc_rel/
        # EKly_acc_rel/Emp1_acc_rel below despite not re-filtering by
        # cons_trials here.
        mAll_hat[i] = findmax(BMS.exp_M[i,:])[2]
        for i_fold = 1:2
            # picking inferred parameters of fold i_fold
            chn_df = CV_chn_dfsSets[i][i_fold][1:thinning:end, :]
            βs = chn_df.β; ls = exp.(chn_df.logl)
            logl_hat[i, i_fold] = mean(chn_df.logl)

            # picking the other, testing fold
            CV_Xinds = CV_XindsSets[i][1 + mod(i_fold,2)]
            CV_as = CV_asSets[i][1 + mod(i_fold,2)]

            # evaluating empowerment and probabilities
            emp_vals_hat = zeros(length(ls), N_rooms)
            p_hats = zeros(length(ls),length(CV_as))
            for i_samp = eachindex(ls)
                # evaluating room empowerments
                emp_model_hat = emplK(ls[i_samp],1.,1)
                for i_room = 1:N_rooms
                    p, StateS, StateSDict, N_s = 
                        gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                            Xmax, Ymax)
                    emp_vals_hat[i_samp,i_room] = 
                            p2empCat(p, StateSDict[(0,0)], emp_model_hat)
                end
                # evaluating choice probabilities
                for t = eachindex(CV_Xinds)
                    x = CV_Xinds[t]
                    pa = softmax([emp_vals_hat[i_samp,x[1]], 
                                  emp_vals_hat[i_samp,x[2]]] .* βs[i_samp])
                    p_hats[i_samp,t] = pa[CV_as[t]]
                end
            end
            mp_hats = mean(p_hats,dims=1)[:]
            Empl_acc_perfold[i, i_fold] = p2accuracy(mp_hats)
        end
        Empl_acc[i] = mean(Empl_acc_perfold[i,:])
        Empl_acc_rel[i] = Empl_acc[i] / cons_inds[i]
    end

    return (; cons_inds = cons_inds, 
            Na_acc = Na_acc,        Na_acc_rel = Na_acc_rel,
            EKly_acc = EKly_acc,    EKly_acc_rel = EKly_acc_rel,
            Hmax_acc = Hmax_acc,    Hmax_acc_rel = Hmax_acc_rel,
            Emp1_acc = Emp1_acc,    Emp1_acc_rel = Emp1_acc_rel,
            Empl_acc = Empl_acc,    Empl_acc_rel = Empl_acc_rel,
            mAll_hat = mAll_hat,    logl_hat = logl_hat,
            Empl_acc_perfold = Empl_acc_perfold)
end

# ----------------------------------------------------------------------
# Accuracy plotting -- three groups (Na / Emp / General)
# ----------------------------------------------------------------------
function acc_plotting_threegroups(CVResults, SavePath; color = MainColors.GB[1])

    PlotInds = [(CVResults.mAll_hat .== 2), (CVResults.mAll_hat .== 3),
                (CVResults.mAll_hat .== 4)]
    IndLegends = ["NaParts", "EmpParts", "General"]

    Legends = ["Na", "EKly", "Emp1", "Hmax", "Emp-ell"]
    Ys = [  CVResults.Na_acc_rel,   CVResults.EKly_acc_rel, CVResults.Emp1_acc_rel, 
            CVResults.Hmax_acc_rel, CVResults.Empl_acc_rel]

    io_log1 = IOBuffer()
    figure(figsize=(11,6))
    for i_group = 1:3
        ax = subplot(1,3,i_group)
        x = 1:length(Ys)
        YPlots = [y[PlotInds[i_group]] for y = Ys]
        mYs = [mean(y) for y = YPlots]
        dYs = [std(y)/sqrt(length(y)) for y = YPlots]

        ax.bar(x, mYs,  color = color, alpha = 1.0)
        ax.errorbar(x, mYs, yerr=dYs,color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)

        for j = eachindex(YPlots)
            Y_test = YPlots[end] - YPlots[j]
            Test_result = OneSampleTTest(Y_test)
            pval = pvalue(Test_result)
            logBF = BIC_OneSampleTTest(Y_test)
            log_test(io_log1,
                "Testing: normalized accuracy of Emp-ell model vs $(Legends[j]) model " *
                "(paired difference), subgroup '$(IndLegends[i_group])' -- " *
                "Figure: AccRate_ThreeGroups",
                Test_result; pval = pval, logBF = logBF)
            ax.text(x[j], 0.85 + 0.05 * mod(j,3),
                    string("p:", Func_pval_string(pval),
                        ", lBF:", Func_logBF_string(logBF)),
                    fontsize=8, horizontalalignment="center", rotation=0)
        end
        for j = eachindex(YPlots[1])
            x_temp = Array(x) .+ (0.2 * (rand(rng) - 0.5))
            y_temp = [y[j] for y = YPlots]
            ax.plot(x_temp, y_temp, ".k",alpha = 0.2)
        end
        ax.set_xlim([x[1] - 1,x[end]+1]);
        ax.set_xticks(x); ax.set_xticklabels(Legends)
        ax.set_ylim([0.5,1.0]);
        ax.set_ylabel("normalized accuracy")
        ax.set_title(IndLegends[i_group] * "; n = " * string(length(YPlots[1])))
    end
    tight_layout()

    savefig(SavePath * "AccRate_ThreeGroups.pdf")
    savefig(SavePath * "AccRate_ThreeGroups.png")
    savefig(SavePath * "AccRate_ThreeGroups.svg")
    open(SavePath * "AccRate_ThreeGroups_stats.txt", "w") do f
        write(f, String(take!(io_log1)))
    end

    io_log2 = IOBuffer()
    figure(figsize=(10,4))
    for i_group = 1:3
        ax = subplot(1,3,i_group)
        Ys = [  CVResults.logl_hat[PlotInds[i_group],1],
                CVResults.logl_hat[PlotInds[i_group],2]]
        ρ = round(cor(Ys...),digits = 2)
        ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
        x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
        x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
        ax.plot([x_min,x_max],[x_min,x_max],"--k")
        ax.plot([x_min,x_max],[1,1] .* 0.,"--k")
        ax.plot([1,1] .* 0.,[x_min,x_max],"--k")
        ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
        Test_result = CorrelationTest(Ys...)
        pval = pvalue(Test_result)
        logBF = BIC_CorrelationTest(Ys...)
        log_test(io_log2,
            "Testing: correlation between log-l-hat from CV fold 1 vs fold 2, " *
            "subgroup '$(IndLegends[i_group])' -- Figure: LConsistency_ThreeGroups",
            Test_result; pval = pval, logBF = logBF)
        ax.set_ylabel("log-l-hat (set 2)")
        ax.set_xlabel("log-l-hat (set 1)")
        ax.legend(["ρ = " * string(ρ)])
        ax.set_title("pval = " * Func_pval_string(pval) *
                    ", lBF = " * Func_logBF_string(logBF))
    end
    tight_layout()

    savefig(SavePath * "LConsistency_ThreeGroups.pdf")
    savefig(SavePath * "LConsistency_ThreeGroups.png")
    savefig(SavePath * "LConsistency_ThreeGroups.svg")
    open(SavePath * "LConsistency_ThreeGroups_stats.txt", "w") do f
        write(f, String(take!(io_log2)))
    end
end


# ----------------------------------------------------------------------
# Accuracy plotting -- combined
# ----------------------------------------------------------------------
function acc_plotting_combined(CVResults, SavePath; color = MainColors.GB[1])

    PlotInds = [(CVResults.mAll_hat .== 3) .|| (CVResults.mAll_hat .== 4)]
    IndLegends = ["EmpParts + General"]

    Legends = ["Na", "EKly", "Emp1", "Hmax", "Emp-ell"]
    Ys = [  CVResults.Na_acc_rel,   CVResults.EKly_acc_rel, CVResults.Emp1_acc_rel, 
            CVResults.Hmax_acc_rel, CVResults.Empl_acc_rel]

    io_log1 = IOBuffer()
    figure(figsize=(4,6))
    for i_group = [1]
        ax = subplot(1,1,i_group)
        x = 1:length(Ys)
        YPlots = [y[PlotInds[i_group]] for y = Ys]
        mYs = [mean(y) for y = YPlots]
        dYs = [std(y)/sqrt(length(y)) for y = YPlots]

        ax.bar(x, mYs,  color = color, alpha = 1.0)
        ax.errorbar(x, mYs, yerr=dYs,color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)

        for j = eachindex(YPlots)
            Y_test = YPlots[end] - YPlots[j]
            Test_result = OneSampleTTest(Y_test)
            pval = pvalue(Test_result)
            logBF = BIC_OneSampleTTest(Y_test)
            log_test(io_log1,
                "Testing: normalized accuracy of Emp-ell model vs $(Legends[j]) model " *
                "(paired difference), subgroup '$(IndLegends[i_group])' -- " *
                "Figure: AccRate_Combined",
                Test_result; pval = pval, logBF = logBF)
            ax.text(x[j], 0.85 + 0.05 * mod(j,3),
                    string("p:", Func_pval_string(pval),
                        ", lBF:", Func_logBF_string(logBF)),
                    fontsize=8, horizontalalignment="center", rotation=0)
        end
        for j = eachindex(YPlots[1])
            x_temp = Array(x) .+ (0.2 * (rand(rng) - 0.5))
            y_temp = [y[j] for y = YPlots]
            ax.plot(x_temp, y_temp, ".k",alpha = 0.2)
        end
        ax.set_xlim([x[1] - 1,x[end]+1]);
        ax.set_xticks(x); ax.set_xticklabels(Legends)
        ax.set_ylim([0.5,1.0]);
        ax.set_ylabel("normalized accuracy")
        ax.set_title(IndLegends[i_group] * "; n = " * string(length(YPlots[1])))
    end
    tight_layout()

    savefig(SavePath * "AccRate_Combined.pdf")
    savefig(SavePath * "AccRate_Combined.png")
    savefig(SavePath * "AccRate_Combined.svg")
    open(SavePath * "AccRate_Combined_stats.txt", "w") do f
        write(f, String(take!(io_log1)))
    end

    io_log2 = IOBuffer()
    figure(figsize=(4,4))
    for i_group = [1]
        ax = subplot(1,1,i_group)
        Ys = [  CVResults.logl_hat[PlotInds[i_group],1],
                CVResults.logl_hat[PlotInds[i_group],2]]
        ρ = round(cor(Ys...),digits = 2)
        ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
        x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
        x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
        ax.plot([x_min,x_max],[x_min,x_max],"--k")
        ax.plot([x_min,x_max],[1,1] .* 0.,"--k")
        ax.plot([1,1] .* 0.,[x_min,x_max],"--k")
        ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
        Test_result = CorrelationTest(Ys...)
        pval = pvalue(Test_result)
        logBF = BIC_CorrelationTest(Ys...)
        log_test(io_log2,
            "Testing: correlation between log-l-hat from CV fold 1 vs fold 2, " *
            "subgroup '$(IndLegends[i_group])' -- Figure: LConsistency_Combined",
            Test_result; pval = pval, logBF = logBF)
        ax.set_ylabel("log-l-hat (set 2)")
        ax.set_xlabel("log-l-hat (set 1)")
        ax.legend(["ρ = " * string(ρ)])
        ax.set_title("pval = " * Func_pval_string(pval) *
                    ", lBF = " * Func_logBF_string(logBF))
    end
    tight_layout()

    savefig(SavePath * "LConsistency_Combined.pdf")
    savefig(SavePath * "LConsistency_Combined.png")
    savefig(SavePath * "LConsistency_Combined.svg")
    open(SavePath * "LConsistency_Combined_stats.txt", "w") do f
        write(f, String(take!(io_log2)))
    end
end

# ----------------------------------------------------------------------
# Loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    # ----------------------------------------------------------------------
    # Loading data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec);
    subjectIDs = data.subjectIDs; selDF = data.selDF
    
    # ----------------------------------------------------------------------
    # Loading inference data
    # ----------------------------------------------------------------------
    infdata = [load(expspec.fig_path * "MCMC_sub" *
                    string(i_sub) * "_CV.jld2") for i_sub = subjectIDs]
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    
    # ----------------------------------------------------------------------
    # Evaluating and plotting accuracy rates
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        BMS = BMSdata["BMSAll"];

        CV_XindsSets = [d["data_save"].Xinds for d = infdata]
        CV_asSets = [d["data_save"].as for d = infdata]
        CV_chn_dfsSets = [[d["data_save"].FITresultsEmpl_CV[1].chnemp_df,
                        d["data_save"].FITresultsEmpl_CV[2].chnemp_df] for 
                                                                d = infdata]

        CVResults = accuracy_calculation(selDF, BMS, subjectIDs,
                        CV_XindsSets, CV_asSets, CV_chn_dfsSets,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)
        if i_exp == 1
            color = MainColors.GB[1]
        else
            color = "#435663"
        end
        acc_plotting_threegroups(CVResults, expspec.fig_path * save_name_tag;
                                    color = color)
        acc_plotting_combined(CVResults, expspec.fig_path * save_name_tag;
                                    color = color)
    else
        BMS = BMSdata["BMSAll"][1]; 
        
        CV_XindsSets = [d["data_save"][1].Xinds for d = infdata]
        CV_asSets = [d["data_save"][1].as for d = infdata]
        CV_chn_dfsSets = [[d["data_save"][1].FITresultsEmpl_CV[1].chnemp_df,
                        d["data_save"][1].FITresultsEmpl_CV[2].chnemp_df] for 
                                                                d = infdata]
        CVResults = accuracy_calculation(selDF[selDF.Gtrials .== 1, :], 
                        BMS, subjectIDs,
                        CV_XindsSets, CV_asSets, CV_chn_dfsSets,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)
        acc_plotting_threegroups(CVResults, 
                expspec.fig_path * save_name_tag * "Gold";
                color = MainColors.GB[1])
        acc_plotting_combined(CVResults, 
                expspec.fig_path * save_name_tag * "Gold";
                color = MainColors.GB[1])

        BMS = BMSdata["BMSAll"][2];
        
        CV_XindsSets = [d["data_save"][2].Xinds for d = infdata]
        CV_asSets = [d["data_save"][2].as for d = infdata]
        CV_chn_dfsSets = [[d["data_save"][2].FITresultsEmpl_CV[1].chnemp_df,
                        d["data_save"][2].FITresultsEmpl_CV[2].chnemp_df] for 
                                                                d = infdata]

        CVResults = accuracy_calculation(selDF[selDF.Gtrials .== 0, :], 
                        BMS, subjectIDs,
                        CV_XindsSets, CV_asSets, CV_chn_dfsSets,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)
    
        acc_plotting_threegroups(CVResults, 
                expspec.fig_path * save_name_tag * "Bomb";
                color = MainColors.GB[2])
        acc_plotting_combined(CVResults, 
                expspec.fig_path * save_name_tag * "Bomb";
                color = MainColors.GB[2])
    end
end
