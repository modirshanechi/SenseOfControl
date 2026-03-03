using PyPlot
using EmpHCA
using LinearAlgebra
using Statistics
using Random
using DataFrames
using CSV
using JLD2
using HypothesisTests

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExp2Data/Figures/"
Path_Load1 = "data/Experiment1/clean/"
Path_Load2 = "data/Experiment2/clean/"

for plot_cond = ["", "outliers", "all"]
# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ExcDF1 = DataFrame(CSV.File(Path_Load1 * "ExclusionInfo.csv"))
dataDF1 = DataFrame(CSV.File(Path_Load1 * "SelectionData.csv"))

ExcDF2 = DataFrame(CSV.File(Path_Load2 * "ExclusionInfo.csv"))
dataDF2 = DataFrame(CSV.File(Path_Load2 * "SelectionData.csv"))

if plot_cond == "all"
        subjectIDs1 = ExcDF1.subject
        subjectIDs2 = ExcDF2.subject
        name_tage = "_all"
elseif plot_cond == "outliers"
        subjectIDs1 = ExcDF1.subject[ExcDF1.task_outliers .== 1]
        subjectIDs2 = ExcDF2.subject[ExcDF2.task_outliers .== 1]
        name_tage = "_outliers"
else
        subjectIDs1 = ExcDF1.subject[ExcDF1.task_outliers .== 0]
        subjectIDs2 = ExcDF2.subject[ExcDF2.task_outliers .== 0]
        name_tage = ""
end

# ----------------------------------------------------------------------
# Simple statistics Exp 2
# ----------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

Data1 = []
for i_sub = subjectIDs1
        @show i_sub
        # ----------------------------------------------------------------------
        # selecting data
        # ----------------------------------------------------------------------
        df = dataDF1[dataDF1.subject .== i_sub, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1

        r1p = mean(as_inds[[1 ∈ x for x = Xinds]] .== 1)
        C = choice_consistency(Xinds, as_inds, ifraw = true)
        
        push!(Data1,(; r1p=r1p,C=C))
end

Data2 = []
for i_sub = subjectIDs2
        @show i_sub
        # ----------------------------------------------------------------------
        # selecting data
        # ----------------------------------------------------------------------
        df = dataDF2[dataDF2.subject .== i_sub, :]
        df = df[df.timeout .== false, :]

        
        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1


        Ginds = df.Gtrials .== 1
        Binds = df.Gtrials .== 0

        r1p = mean(as_inds[[1 ∈ x for x = Xinds]] .== 1)
        r1pG = mean(as_inds[Ginds][[1 ∈ x for x = Xinds[Ginds]]] .== 1)
        r1pB = mean(as_inds[Binds][[1 ∈ x for x = Xinds[Binds]]] .== 1)

        CG = choice_consistency(Xinds[Ginds], as_inds[Ginds], ifraw = true)
        CB = choice_consistency(Xinds[Binds], as_inds[Binds], ifraw = true)

        CGB = choice_consistency(Xinds[Ginds], as_inds[Ginds], 
                                 Xinds[Binds], as_inds[Binds])

        push!(Data2,
                (; r1p=r1p,r1pG=r1pG,r1pB=r1pB,
                CG=CG,CB=CB,CGB=CGB, Cond = df.GB_condition[1]))
end


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
Y1s = [ [d.r1p for d = Data1],
        [d.C for d = Data1]]
Y2s = [ hcat([[d.r1pG,d.r1pB,d.r1p] for d = Data2]...),
        hcat([[d.CG,d.CB,d.CGB] for d = Data2]...)]
YLabel = ["Room 1 preference","Choice Consistency"]
XLabel = ["E1"  ,"E2(GB)-G","E2(GB)-B","E2(GB)-G+B"
                ,"E2(BG)-G","E2(BG)-B","E2(BG)-G+B"]
figure(figsize = (8,8))
for i = 1:2
        XTicks = []
        ax = subplot(2,1,i)
        Y = Y1s[i]
        my = mean(Y); dy = std(Y) / sqrt(length(Y))
        ax.bar(1,my,color=MainColors.GB[1])
        ax.errorbar(1,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        ax.plot(1 .+ 0.2 .* (rand(length(Y)) .- 0.5), Y, ".k",alpha=0.3)
        push!(XTicks, 1)
        for j_cond = [1,0]
                Y = Y2s[i][:,[d.Cond for d = Data2] .== j_cond]
                δx = 1.5 + mod(j_cond + 1, 2) * 0.5 + mod(j_cond + 1, 2) * size(Y)[1]
                for j = 1:size(Y)[1]
                        my = mean(Y[j,:]); dy = std(Y[j,:]) / sqrt(length(Y[j,:]))
                        push!(XTicks, δx + j)
                        ax.bar(δx + j,my,color=MainColors.GB[j])
                        ax.errorbar(δx + j,my,yerr=dy,color="k",
                                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                end
                for j = 1:size(Y)[2]
                        y = Y[:,j]; δ = 0.2 * (rand() - 0.5)
                        ax.plot(δx .+ (1:length(y)) .+ δ, y, ".k",alpha=0.3)
                        ax.plot(δx .+ (1:length(y)) .+ δ, y, "k",alpha=0.05)
                end
        end
        ax.set_xticks(XTicks); ax.set_xticklabels(XLabel)
        ax.set_ylabel(YLabel[i]); 
        ax.set_ylim([0,1]); ax.set_xlim([0,XTicks[end]+1])
end
tight_layout()
savefig(Path_Save * "R1P_Const" * name_tage * ".pdf")
savefig(Path_Save * "R1P_Const" * name_tage * ".png")
savefig(Path_Save * "R1P_Const" * name_tage * ".svg")

Y1s = [ [d.r1p for d = Data1],
        [d.C for d = Data1]]
Y2s = [ hcat([[d.r1pG,d.r1pB,d.r1p] for d = Data2]...),
        hcat([[d.CG,d.CB,d.CGB] for d = Data2]...)]
YLabel = ["Room 1 preference","Choice Consistency"]
XLabel = ["E1" ,"E2-G","E2-B","E2-G+B"]
figure(figsize = (6,8))
for i = 1:2
        XTicks = []
        ax = subplot(2,1,i)
        Y = Y1s[i]
        my = mean(Y); dy = std(Y) / sqrt(length(Y))
        ax.bar(1,my,color=MainColors.GB[1])
        ax.errorbar(1,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        ax.plot(1 .+ 0.2 .* (rand(length(Y)) .- 0.5), Y, ".k",alpha=0.3)
        push!(XTicks, 1)
        Y = Y2s[i]
        δx = 1.5
        for j = 1:size(Y)[1]
                my = mean(Y[j,:]); dy = std(Y[j,:]) / sqrt(length(Y[j,:]))
                push!(XTicks, δx + j)
                ax.bar(δx + j,my,color=MainColors.GB[j])
                ax.errorbar(δx + j,my,yerr=dy,color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        end
        for j = 1:size(Y)[2]
                y = Y[:,j]; δ = 0.2 * (rand() - 0.5)
                ax.plot(δx .+ (1:length(y)) .+ δ, y, ".k",alpha=0.3)
                ax.plot(δx .+ (1:length(y)) .+ δ, y, "k",alpha=0.05)
        end
        ax.set_xticks(XTicks); ax.set_xticklabels(XLabel)
        ax.set_ylabel(YLabel[i]); 
        ax.set_ylim([0,1]); ax.set_xlim([0,XTicks[end]+1])
end
tight_layout()
savefig(Path_Save * "R1P_Const2" * name_tage * ".pdf")
savefig(Path_Save * "R1P_Const2" * name_tage * ".png")
savefig(Path_Save * "R1P_Const2" * name_tage * ".svg")


# ----------------------------------------------------------------------
# Testing E2(GB) versus E2(BG)
# ----------------------------------------------------------------------
Ys = [  [d.CG for d = Data2],
        [d.CB for d = Data2],
        [d.CGB for d = Data2]]
Xs = [d.Cond for d = Data2]
Labels = ["CG", "CB", "CGB"]

for i = eachindex(Ys)
        println("---------------------------------------------------------------")
        println("Exp 2 GB versus BG")
        @show Labels[i]
        println("---------------------------------------------------------------")
        y0 = Ys[i][Xs .== 0]; y1 = Ys[i][Xs .== 1]
        Test_result = UnequalVarianceTTest(y0,y1)
        @show Test_result
        logBF = BIC_UnequalVarianceTTest(y0,y1)
        @show logBF
end

# ----------------------------------------------------------------------
# Testing E1 versus E2
# ----------------------------------------------------------------------
Ys = [ [d.C for d = Data1], [d.CG for d = Data2]]
Xs = [d.Cond for d = Data2]
Labels = ["Exp 1 versus Exp 2 (BG)", "Exp 1 versus Exp 2 (GB)","Exp 1 versus Exp 2 (All)"]

for i = 1:3
        println("---------------------------------------------------------------")
        @show Labels[i]
        println("---------------------------------------------------------------")
        y0 = Ys[1]; 
        if i < 3
                y1 = Ys[2][Xs .== (i-1)]
        else
                y1 = Ys[2]
        end
        Test_result = UnequalVarianceTTest(y0,y1)
        @show Test_result
        logBF = BIC_UnequalVarianceTTest(y0,y1)
        @show logBF
end

# ----------------------------------------------------------------------
# within E2
# ----------------------------------------------------------------------
Ys = [  [[d.CG for d = Data2],[d.CB for d = Data2]],
        [[d.CG for d = Data2],[d.CGB for d = Data2]],
        [[d.CB for d = Data2],[d.CGB for d = Data2]]]
Xs = [d.Cond for d = Data2]
Labels = ["Exp 2: G versus B", "Exp 2: G versus GB", "Exp 2: B versus GB"]
CondLabels = ["BG","GB","All"]

for i = 1:3
        for i_cond = 1:3
                println("---------------------------------------------------------------")
                println("---------------------------------------------------------------")
                @show CondLabels[i_cond]
                println("--------------------------")
                @show Labels[i]
                println("---------------------------------------------------------------")
                println("---------------------------------------------------------------")
                y0 = Ys[i][1]; y1 = Ys[i][2];
                if i_cond < 3
                        y0 = y0[Xs .== (i_cond-1)]
                        y1 = y1[Xs .== (i_cond-1)]
                end
                Test_result = OneSampleTTest(y0,y1)
                @show Test_result
                logBF = BIC_OneSampleTTest(y0,y1)
                @show logBF
        end
end
end