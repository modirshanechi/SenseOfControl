################################################################################
# Code for performing two control analyses on Experiments 1-2 reported in
# ---> Fig S11--S12
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


save_name_tag = "Control/"

Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

# ----------------------------------------------------------------------
# Comparing room 1 preferences of outliers in GB and BG conditions (Fig S11)
# ----------------------------------------------------------------------
expspec1 = ExperimentSpecification(1);
expspec2 = ExperimentSpecification(2);

data1 = load_clean_data(expspec1; ifoutliers = 1)
data2 = load_clean_data(expspec2; ifoutliers = 1)

GBcond = [data2.selDF.GB_condition[data2.selDF.subject .== i_sub][1] 
                for i_sub = data2.subjectIDs]

p1e1 = zeros(length(data1.subjectIDs));
p1e2 = zeros(length(data2.subjectIDs));
p1e2G = zeros(length(data2.subjectIDs));
p1e2B = zeros(length(data2.subjectIDs));

function room1preference(df)
    df = df[df.timeout .== false, :]
    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    as_inds = df.chosenroom .+ 1
    return mean(as_inds[[1 ∈ x for x = Xinds]] .== 1)
end

for (i, i_sub) = enumerate(data1.subjectIDs)
        @show i_sub
        df = data1.selDF[data1.selDF.subject .== i_sub, :]
        p1e1[i] = room1preference(df)
end

for (i, i_sub) = enumerate(data2.subjectIDs)
        @show i_sub
        df = data2.selDF[data2.selDF.subject .== i_sub, :]
        p1e2[i] = room1preference(df)
        p1e2G[i] = room1preference(df[df.Gtrials .== 1,:])
        p1e2B[i] = room1preference(df[df.Gtrials .== 0,:])
end

# Plotting
YLabel = "Room 1 preference for outliers"
XLabel = ["E1"  ,"E2(GB)-G","E2(GB)-B","E2(GB)-G+B"
                ,"E2(BG)-G","E2(BG)-B","E2(BG)-G+B"]

figure(figsize = (8,4))
ax = subplot(1,1,1)
XTicks = Vector{Float64}([])

Y = p1e1
my = mean(Y); dy = std(Y) / sqrt(length(Y))
ax.bar(1,my,color=MainColors.GB[1])
ax.errorbar(1,my,yerr=dy,color="k",
        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
ax.plot(1 .+ 0.2 .* (rand(rng, length(Y)) .- 0.5), Y, ".k",alpha=0.3)
push!(XTicks, 1)

Ys = [p1e2G, p1e2B, p1e2]
for j_cond = [1,0]
        δx = 1.5 + mod(j_cond + 1, 2) * 0.5 + mod(j_cond + 1, 2) * length(Ys)
        for j = eachindex(Ys)
                Y = Ys[j][GBcond .== j_cond]
                my = mean(Y); dy = std(Y) / sqrt(length(Y))
                push!(XTicks, δx + j)
                ax.bar(δx + j,my,color=MainColors.GB[j])
                ax.errorbar(δx + j,my,yerr=dy,color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        end
        for j = 1:sum(GBcond .== j_cond)
                y = [Y[GBcond .== j_cond][j] for Y = Ys]; 
                δ = 0.2 * (rand(rng) - 0.5)
                ax.plot(δx .+ (1:length(y)) .+ δ, y, ".k",alpha=0.3)
                ax.plot(δx .+ (1:length(y)) .+ δ, y, "k",alpha=0.05)
        end
end
ax.set_xticks(XTicks); ax.set_xticklabels(XLabel)
ax.set_ylabel(YLabel); 
ax.set_ylim([0,1]); ax.set_xlim([0,XTicks[end]+1])

tight_layout()
savefig(expspec2.fig_path * save_name_tag * "R1P.pdf")
savefig(expspec2.fig_path * save_name_tag * "R1P.png")
savefig(expspec2.fig_path * save_name_tag * "R1P.svg")


# ----------------------------------------------------------------------
# Consistency calculation with respect to the time passed since the
# last feedback trial (Fig S12)
# ----------------------------------------------------------------------
data1 = load_clean_data(expspec1)
data2 = load_clean_data(expspec2)

feedback_period = 7
conste1  = zeros(length(data1.subjectIDs), feedback_period);
conste2G = zeros(length(data2.subjectIDs), feedback_period);
conste2B = zeros(length(data2.subjectIDs), feedback_period);

function feedbackconsistency(df, feedback_period)
    df[!, "afeedback"] = mod.(df.trial .- 1, feedback_period) .+ 1
    df = df[df.trial .> feedback_period, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    as_inds = df.chosenroom .+ 1
    afeed = df.afeedback
    
    return [subchoice_consistency(
                Xinds[afeed .== t], as_inds[afeed .== t],
                Xinds[afeed .!= t], as_inds[afeed .!= t]) 
                                    for t = 1:feedback_period]
end

for (i, i_sub) = enumerate(data1.subjectIDs)
        @show i_sub
        df = data1.selDF[data1.selDF.subject .== i_sub, :]
        conste1[i, :] = 
            feedbackconsistency(df, feedback_period)
end

for (i, i_sub) = enumerate(data2.subjectIDs)
        @show i_sub
        df = data2.selDF[data2.selDF.subject .== i_sub, :]
        conste2G[i, :] = 
            feedbackconsistency(df[df.Gtrials .== 1,:], feedback_period)
        conste2B[i, :] = 
            feedbackconsistency(df[df.Gtrials .== 0,:], feedback_period)
end

# Plotting
Ys = [conste1, conste2G, conste2B]
YLabel = "Choice Consistency"
GroupLabel = ["E1"  ,"E2-G","E2-B"]
GroupColor = [MainColors.GB[1], MainColors.GB[1], MainColors.GB[2]]
figure(figsize = (6,8))
for i = eachindex(Ys)
        ax = subplot(length(Ys),1,i)
        yset = [Ys[i][:,t] for t = 1:feedback_period]
        yset = [y[isnan.(y) .== 0] for y = yset]
        my = mean.(yset)
        dy = std.(yset) ./ sqrt.(length.(yset)); 

        ax.bar(1:feedback_period,my,color=GroupColor[i])
        ax.errorbar(1:feedback_period,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        for t = 1:feedback_period
                ax.plot(t .+ 0.2 .* (rand(rng, length(yset[t])) .- 0.5), 
                        yset[t], ".k",alpha=0.3)
        end
        ax.set_title(GroupLabel[i])
        ax.set_xticks(1:feedback_period)
        ax.set_ylabel(YLabel); ax.set_xlabel("time since feedback"); 
        ax.set_ylim([0,1]); ax.set_xlim([0,feedback_period+1])
end
tight_layout()
savefig(expspec2.fig_path * save_name_tag * "AFeed.pdf")
savefig(expspec2.fig_path * save_name_tag * "AFeed.png")
savefig(expspec2.fig_path * save_name_tag * "AFeed.svg")

