################################################################################
# Code for consistency calculation with respect to the time passed since the 
# last feedback trial (reported in the Supplementary Materials)
################################################################################
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

feedback_period = 7
# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ExcDF1 = DataFrame(CSV.File(Path_Load1 * "ExclusionInfo.csv"))
dataDF1 = DataFrame(CSV.File(Path_Load1 * "SelectionData.csv"))

ExcDF2 = DataFrame(CSV.File(Path_Load2 * "ExclusionInfo.csv"))
dataDF2 = DataFrame(CSV.File(Path_Load2 * "SelectionData.csv"))

subjectIDs1 = ExcDF1.subject[ExcDF1.task_outliers .== 0]
subjectIDs2 = ExcDF2.subject[ExcDF2.task_outliers .== 0]

# ----------------------------------------------------------------------
# Feedback vs. the rest
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
        df[!, "afeedback"] = mod.(df.trial .- 1, feedback_period) .+ 1
        df = df[df.trial .> feedback_period, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1
        afeed = df.afeedback

        r1p = [mean(as_inds[[1 ∈ x for x = Xinds] .& (afeed .== t)] .== 1) 
                                                for t = 1:feedback_period]

        C = [subchoice_consistency(Xinds[afeed .== t], as_inds[afeed .== t],
                                   Xinds[afeed .!= t], as_inds[afeed .!= t]) for t = 1:feedback_period]
        push!(Data1,(; r1p=r1p,C=C))
end

Data2 = []
for i_sub = subjectIDs2
        @show i_sub
        # ----------------------------------------------------------------------
        # selecting data
        # ----------------------------------------------------------------------
        df = dataDF2[dataDF2.subject .== i_sub, :]
        for g = 0:1
                df.trial[df.Gtrials .== g] .= 1:sum(df.Gtrials .== g)
        end
        df[!, "afeedback"] = mod.(df.trial .- 1, feedback_period) .+ 1
        df = df[df.trial .> feedback_period, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1
        afeed = df.afeedback


        Ginds = df.Gtrials .== 1
        Binds = df.Gtrials .== 0

        r1pG = [mean(
                as_inds[[1 ∈ x for x = Xinds] .& (afeed .== t) .& Ginds] .== 1) 
                                                for t = 1:feedback_period]
        r1pB = [mean(
                as_inds[[1 ∈ x for x = Xinds] .& (afeed .== t) .& Binds] .== 1) 
                                                for t = 1:feedback_period]
        CG = [subchoice_consistency(
                Xinds[(afeed .== t) .& Ginds], as_inds[(afeed .== t) .& Ginds],
                Xinds[(afeed .!= t) .& Ginds], as_inds[(afeed .!= t) .& Ginds]) for t = 1:feedback_period]
        CB = [subchoice_consistency(
                Xinds[(afeed .== t) .& Binds], as_inds[(afeed .== t) .& Binds],
                Xinds[(afeed .!= t) .& Binds], as_inds[(afeed .!= t) .& Binds]) for t = 1:feedback_period]

        push!(Data2,
                (; r1pG=r1pG,r1pB=r1pB,
                CG=CG,CB=CB, Cond = df.GB_condition[1]))
end


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
R1Ps = [[d.r1p  for d = Data1],
        [d.r1pG for d = Data2],
        [d.r1pB for d = Data2]]
Cs = [  [d.C  for d = Data1],
        [d.CG for d = Data2],
        [d.CB for d = Data2]]
Ys = [R1Ps,Cs]
YLabel = ["Room 1 preference","Choice Consistency"]
nametag = ["R1P","Const"]
GroupLabel = ["E1"  ,"E2-G","E2-B"]
GroupColor = [MainColors.GB[1], MainColors.GB[1], MainColors.GB[2]]
for i = eachindex(Ys)
        figure(figsize = (6,8))
        for j = eachindex(Ys[i])
                ax = subplot(length(Ys[i]),1,j)
                Y = Ys[i][j]
                my = zeros(feedback_period); dy = zeros(feedback_period)
                yset = []  
                for t = 1:feedback_period
                        temp = [y[t] for y = Y]; temp = temp[isnan.(temp) .== 0]
                        push!(yset, temp)
                        my[t] = mean(temp);  
                        dy[t] = std(temp) / sqrt(length(temp))
                end
                ax.bar(1:feedback_period,my,color=GroupColor[j])
                ax.errorbar(1:feedback_period,my,yerr=dy,color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
                for t = 1:feedback_period
                        ax.plot(t .+ 0.2 .* (rand(length(yset[t])) .- 0.5), 
                                yset[t], ".k",alpha=0.3)
                end
                ax.set_title(GroupLabel[j])
                ax.set_xticks(1:feedback_period)
                ax.set_ylabel(YLabel[i]); ax.set_xlabel("time since feedback"); 
                ax.set_ylim([0,1]); ax.set_xlim([0,feedback_period+1])
        end
        tight_layout()
        savefig(Path_Save * "AFeed_" * nametag[i] * ".pdf")
        savefig(Path_Save * "AFeed_" * nametag[i] * ".png")
        savefig(Path_Save * "AFeed_" * nametag[i] * ".svg")
end
