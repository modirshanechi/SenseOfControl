################################################################################
# Code for plotting room preferences for different participant groups
# ---> Fig 2H
# ---> Fig 5F
# ---> Fig S6--S9
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


save_name_tag = "RoomPref/"
infdata_path(i) = "MCMC_sub" * string(i) * ".jld2"

Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

# for accelerating the analysis and averaging
thinning = 10

# ----------------------------------------------------------------------
# Room preference evaluation
# ----------------------------------------------------------------------
function preference_evaluation(selDF, chnemp_dfs, chnNa_dfs, 
                        BMS, BMSEmp, subjectIDs, EmpSubjects,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax;
                        thinning = thinning)
    Nsub = length(subjectIDs)
    mAll_hat = zeros(Nsub) .* NaN; mEmp_hat = zeros(Nsub) .* NaN;
    prefMat_raws   = [zeros(N_rooms, N_rooms) .* NaN for i = 1:Nsub]
    prefMat_models = [zeros(N_rooms, N_rooms) .* NaN for i = 1:Nsub]

    for i = 1:Nsub
        @show i
        i_sub = subjectIDs[i]
        
        # ----------------------------------------------------------------------
        # choices
        # ----------------------------------------------------------------------
        df = selDF[selDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        Xchosen = df.chosenroom
        # ----------------------------------------------------------------------
        # inference data
        # ----------------------------------------------------------------------
        chnemp_df = chnemp_dfs[i][1:thinning:end, :]; 
        chnNa_df = chnNa_dfs[i][1:thinning:end, :]
        mAll_hat[i] = findmax(BMS.exp_M[i,:])[2]

        # ----------------------------------------------------------------------
        # room values
        # ----------------------------------------------------------------------
        if i ∈ EmpSubjects
            mEmp_hat[i] = findmax(BMSEmp.exp_M[EmpSubjects .== i,:][:])[2]
            βs = chnemp_df.β; ls = exp.(chnemp_df.logl)
            room_values  = zeros(length(ls),N_rooms)
            for i_samp = eachindex(ls)
                emp_model_hat = emplK(ls[i_samp],1.,1)
                for i_room = 1:N_rooms
                    p, StateS, StateSDict, N_s = 
                        gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                            Xmax, Ymax)
                    room_values[i_samp,i_room] = βs[i_samp] * 
                                p2empCat(p, StateSDict[(0,0)], emp_model_hat)
                end
            end
        else
            βs = chnNa_df.βa
            room_values  = zeros(length(βs),N_rooms)
            for i_samp = eachindex(βs)
                for i_room = 1:N_rooms
                    room_values[i_samp,i_room] = 
                                    βs[i_samp] * size(Prooms[i_room])[1]
                end
            end
        end

        # ----------------------------------------------------------------------
        # pairwise preferences
        # ----------------------------------------------------------------------
        for i1_paper = 1:N_rooms
        for i2_paper = (i1_paper+1):N_rooms
            i1 = PaperRoomOrder[i1_paper]
            i2 = PaperRoomOrder[i2_paper]
            t_temp = [findfirst(==([i1,i2]), Xinds),
                      findfirst(==([i2,i1]), Xinds)]
            t_temp = t_temp[isnothing.(t_temp) .== 0]
            if length(t_temp) > 0
                mp1 = mean((Xchosen[t_temp].+1) .== i1)
                mp2 = 1 - mp1
                prefMat_raws[i][i1_paper, i2_paper] = mp1 - mp2
                prefMat_raws[i][i2_paper, i1_paper] = mp2 - mp1

                v1s = room_values[:,i1]; 
                v2s = room_values[:,i2]; Δvs = v2s .- v1s

                p1s = 1 ./ (1 .+ exp.(Δvs)); p2s = 1 .- p1s
                mp1 = mean(p1s); mp2 = mean(p2s);
                prefMat_models[i][i1_paper, i2_paper] = mp1 - mp2
                prefMat_models[i][i2_paper, i1_paper] = mp2 - mp1
            end
        end
        end
    end
    return prefMat_raws, prefMat_models, mAll_hat, mEmp_hat
end

# ----------------------------------------------------------------------
# Preference plotting: conf matrix
# ----------------------------------------------------------------------
nanmean(x) = isempty(x) ? NaN : mean(x)
elementwise_nanmean(mats, N_rooms) =
    [nanmean(filter(!isnan, [mat[r, c] for mat in mats]))
     for r = 1:N_rooms, c = 1:N_rooms]

function conf_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, SavePath)
    Legends = ["L < 1", "L = 1", "L > 1", "Na"]
    PlotCol = vcat(MainColors.lcol, "#808080")
    PlotInds = [((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 1),
                ((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 2),
                ((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 3),
                (mAll_hat .== 2)]
    
    # confusion matrices
    for m = eachindex(Legends)
        n = sum(PlotInds[m])
        fig = figure(figsize=(12,6))
        if n > 1
            my_raw   = elementwise_nanmean(prefMat_raws[PlotInds[m]], N_rooms)
            my_model = elementwise_nanmean(prefMat_models[PlotInds[m]], N_rooms)
            
            ρ = cor(my_raw[:][isnan.(my_raw[:]) .== 0],
                    my_model[:][isnan.(my_raw[:]) .== 0])

            Y = my_raw
            ax = subplot(1,2,1)
            cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
            fig.colorbar(cp, ax=ax)
            ax.set_xticks(0:(N_rooms - 1)); 
            ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_yticks(0:(N_rooms - 1)); 
            ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_title("Data; P(RY - RX); " * Legends[m] * "; n = " * string(n))

            Y = my_model
            ax = subplot(1,2,2)
            cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
            fig.colorbar(cp, ax=ax)
            ax.set_xticks(0:(N_rooms - 1)); 
            ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_yticks(0:(N_rooms - 1)); 
            ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_title("Model; P(RY - RX); r = " * string(round(ρ,digits=3)))
        
            tight_layout()
        end
        savefig(SavePath * "PrefMat_M" * string(m) * ".pdf")
        savefig(SavePath * "PrefMat_M" * string(m) * ".png")
        savefig(SavePath * "PrefMat_M" * string(m) * ".svg")
    end
end

# ----------------------------------------------------------------------
# Preference plotting: room pairs
# ----------------------------------------------------------------------
function pair_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, SavePath; 
                            roompairs = [[1,2], [3,4]])
    Legends = ["L < 1", "L = 1", "L > 1", "Na"]
    PlotCol = vcat(MainColors.lcol, "#808080")
    PlotInds = [((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 1),
                ((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 2),
                ((mAll_hat .== 3) .|| (mAll_hat .== 4)) .&& (mEmp_hat .== 3),
                (mAll_hat .== 2)]
    
    # room 1vs2 and 3vs4
    for i_rooms = roompairs
        i1, i2 = i_rooms; x = 1:2;
        RoomLegends = ["Room " * string(i1), "Room " * string(i2)]
        io_log = IOBuffer()
        figure(figsize=(12,3))
        for m = eachindex(Legends)
            n = sum(PlotInds[m])
            ax = subplot(1,4,m)
            if n > 1
                Y_test = [d[i1,i2] for d = prefMat_raws][PlotInds[m]]
                Y_test = Y_test[isnan.(Y_test) .== 0]
                Y = [[(1+y)/2 , (1-y)/2] for y = Y_test]
                my = mean(Y); dy = std(Y) / sqrt(length(Y))
                ax.bar(x, my, color = PlotCol[m])
                ax.errorbar(x,my,yerr=dy,color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                
                Test_result = OneSampleTTest(Y_test)
                pval = pvalue(Test_result)
                logBF = [BIC_OneSampleTTest(Y_test), -BIC_OneSampleTTest(Y_test)]
                log_test(io_log,
                    "Testing: raw preference score for Room $i1 vs Room $i2 " *
                    "against 0, subgroup '$(Legends[m])' (n = $n) -- " *
                    "Figure: RoomPair_$(i1)_$(i2)",
                    Test_result; pval = pval, logBF = logBF[1])
                ax.set_title(Legends[m] *
                        "; p:" * Func_pval_string(pval) *
                        ", lBF:" * Func_logBF_string(logBF[1]))
            end
            ax.set_ylabel("ratio of choice out of 2")
            ax.set_xticks(x); ax.set_xticklabels(RoomLegends)
            ax.set_ylim([0,1.]); ax.set_xlim([0,x[end]+1])
        end
        tight_layout()
        savefig(SavePath * "RoomPair_" * string(i1) * "_" * string(i2) * ".pdf")
        savefig(SavePath * "RoomPair_" * string(i1) * "_" * string(i2) * ".png")
        savefig(SavePath * "RoomPair_" * string(i1) * "_" * string(i2) * ".svg")
        open(SavePath * "RoomPair_" * string(i1) * "_" * string(i2) * "_stats.txt", "w") do f
            write(f, String(take!(io_log)))
        end
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
    infdata = [load(expspec.fig_path * infdata_path(i_sub)) for i_sub = subjectIDs]
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    
    # ----------------------------------------------------------------------
    # Evaluating and plotting room preferences
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        BMS = BMSdata["BMSAll"]; BMSEmp = BMSdata["BMSEmp"]
        chnemp_dfs = [d["FITresultsEmpl"].chnemp_df for d = infdata]
        chnNa_dfs = [d["FITresultsNa"].chnA_df for d = infdata]
        EmpSubjects = BMSdata["EmpSubjects"]

        prefMat_raws, prefMat_models, mAll_hat, mEmp_hat = 
            preference_evaluation(selDF, chnemp_dfs, chnNa_dfs, 
                        BMS, BMSEmp, subjectIDs, EmpSubjects,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)

        conf_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag)
        pair_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag)
    else
        BMS = BMSdata["BMSAll"][1]; BMSEmp = BMSdata["BMSEmp"][1]
        chnemp_dfs = [d["FITresultsEmpl"][1].chnemp_df for d = infdata]
        chnNa_dfs = [d["FITresultsNa"][1].chnA_df for d = infdata]
        EmpSubjects = BMSdata["EmpSubjects"][1]

        prefMat_raws, prefMat_models, mAll_hat, mEmp_hat = 
            preference_evaluation(selDF[selDF.Gtrials .== 1, :], 
                        chnemp_dfs, chnNa_dfs, 
                        BMS, BMSEmp, subjectIDs, EmpSubjects,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)

        conf_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag * "Gold")
        pair_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag * "Gold")

        
        BMS = BMSdata["BMSAll"][2]; BMSEmp = BMSdata["BMSEmp"][2]
        chnemp_dfs = [d["FITresultsEmpl"][2].chnemp_df for d = infdata]
        chnNa_dfs = [d["FITresultsNa"][2].chnA_df for d = infdata]
        EmpSubjects = BMSdata["EmpSubjects"][2]

        prefMat_raws, prefMat_models, mAll_hat, mEmp_hat = 
            preference_evaluation(selDF[selDF.Gtrials .== 0, :], 
                        chnemp_dfs, chnNa_dfs, 
                        BMS, BMSEmp, subjectIDs, EmpSubjects,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax)

        conf_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag * "Bomb")
        pair_preference_plotting(prefMat_raws, prefMat_models, 
                            mAll_hat, mEmp_hat, 
                            expspec.fig_path * save_name_tag * "Bomb")
    end
    close("all")
end
