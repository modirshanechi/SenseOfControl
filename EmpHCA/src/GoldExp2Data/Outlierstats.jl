################################################################################
# Code for outlier statistics
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random, Statistics
using DataFrames
using CSV
using JLD2
using JSON3
import StatsBase: countmap

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Load = "data/Experiment2/clean/"

# ------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
selDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))
surveyDF = DataFrame(CSV.File(Path_Load * "SurveyData.csv"))
metaDF_survey = DataFrame(CSV.File(Path_Load * "SurveyMetaData.csv"))

# ------------------------------------------------------------------------------
# outliers stats
# ------------------------------------------------------------------------------
println("------------------------------------------------")
println("------------------------------------------------")
println("------------------------------------------------")
println("Total number of participants:")
println(sum(size(ExcDF)[1]))
println("Task-based rejection rate:")
println(mean(ExcDF.task_outliers))
println("Good participants:")
println(sum(1 .- ExcDF.task_outliers))
println("Total rejection rate:")
println(mean(ExcDF.outliers))
println("Good participants:")
println(sum(1 .- ExcDF.outliers))


println("------------------------------------------------")
println("------------------------------------------------")
println("------------------------------------------------")
println(countmap(surveyDF.sex[ExcDF.task_outliers .== 0]))
println("Age:")
x = surveyDF.age[ExcDF.task_outliers .== 0]; x = x[ismissing.(x) .== 0]
print(mean(x))
print("+-")
println(std(x))
println("Time-outs bomb")
println(countmap(ExcDF.selection_timeout_bomb))
println("Time-outs gold")
println(countmap(ExcDF.selection_timeout_gold))
println("Time-outs sum")
println(sum((ExcDF.selection_timeout_gold .>= 5) .|
            (ExcDF.selection_timeout_bomb .>= 5) ))

println("------------------------------------------------")
println("------------------------------------------------")
println("------------------------------------------------")
GInds = ExcDF.GB_condition .== 1
println("Total number of participants - GB condition:")
println(sum(size(ExcDF[GInds,:])[1]))
println("Task-based rejection rate - GB condition:")
println(mean(ExcDF.task_outliers[GInds]))
println("Good participants - GB condition:")
println(sum(1 .- ExcDF.task_outliers[GInds]))
println("Total rejection rate - GB condition:")
println(mean(ExcDF.outliers[GInds]))
println("Good participants - GB condition:")
println(sum(1 .- ExcDF.outliers[GInds]))

println("------------------------------------------------")
println("------------------------------------------------")
println("------------------------------------------------")
BInds = ExcDF.GB_condition .== 0
println("Total number of participants - BG condition:")
println(sum(size(ExcDF[BInds,:])[1]))
println("Task-based rejection rate - BG condition:")
println(mean(ExcDF.task_outliers[BInds]))
println("Good participants - BG condition:")
println(sum(1 .- ExcDF.task_outliers[BInds]))
println("Total rejection rate - BG condition:")
println(mean(ExcDF.outliers[BInds]))
println("Good participants - BG condition:")
println(sum(1 .- ExcDF.outliers[BInds]))

# ------------------------------------------------------------------------------
# reward stats
# ------------------------------------------------------------------------------
valid_rooms = [0,1,2,4,5,6,7,8,9] .+ 1
Prooms, ΔState, ΔStateDict = gold_proom_sets(); Prooms = Prooms[valid_rooms]
N_rooms = length(Prooms); Ymax = 1; Xmax = 1
Xinds = vcat([[[i,j] for j = (1:N_rooms)[(1:N_rooms) .!= i]] 
                                                for i = 1:N_rooms]...)
emp1 = zeros(N_rooms); emp1_model = emplK(1.,1.,1)
for i_room = 1:N_rooms
    p, StateS, StateSDict, N_s = 
        gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                            Xmax, Ymax)
    emp1[i_room] = p2empCat(p, StateSDict[(0,0)], emp1_model)
end
Pgold = emp1 ./ (length(ΔState) - 1);

FBPeriod = 7; 
# chance level
Nsamp = 100000
ChanceGold  = zeros(Nsamp); 
ChanceBomb  = zeros(Nsamp); 
ChanceNet  = zeros(Nsamp); 
ChanceRoom1 = zeros(Nsamp)
seed = 2024; rng = MersenneTwister(seed)
for i = 1:Nsamp
    if mod(i,10000) == 0
        @show i
    end
    Xinds_temp = Xinds[randperm(rng, length(Xinds))]
    gold_temp = 0
    bomb_temp = 0
    choice_temp = Int.(zeros(length(Xinds))); 
    Xch_temp = Int.(zeros(length(Xinds)))
    for j = eachindex(Xinds_temp)
        choice_temp[j] = rand(rng,[1,2])
        Xch_temp[j] = Xinds_temp[j][choice_temp[j]]
        gold_temp += Pgold[Xch_temp[j]]
        bomb_temp += (1 - Pgold[Xch_temp[j]])
        if mod(j, FBPeriod) == 0 
            gold_temp = ceil(gold_temp)
            bomb_temp = floor(bomb_temp)
        end
    end
    ChanceRoom1[i] = mean(Xch_temp[[1 ∈ x for x = Xinds_temp]] .== 1)
    ChanceGold[i] = gold_temp
    ChanceBomb[i] = bomb_temp
    ChanceNet[i] = gold_temp - bomb_temp
end
mGChance = mean(ChanceGold)
mGChance_plus =  quantile(ChanceGold,0.975)

mBChance = mean(ChanceBomb)
mBChance_plus = quantile(ChanceBomb,0.025)

mBGChance = mean(ChanceNet)
mBGChance_plus =  quantile(ChanceNet,0.975)

mR1Chance = mean(ChanceRoom1)
mR1Chance_plus =  quantile(ChanceRoom1,0.975)

# ------------------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------------------
figure(figsize = (14,8))

ax = subplot(2,4,1)
ax.hist(ExcDF.room1preference[ExcDF.task_outliers .== 0], 
                                0.0:0.025:1, alpha =0.5)
ax.hist(ExcDF.room1preference[ExcDF.task_outliers .== 1], 
                                0.0:0.025:1, alpha =0.5)
x_plot = [1,1] .* mR1Chance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mR1Chance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("Room 1 preferences (all)"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])

ax = subplot(2,4,2)
ax.hist(ExcDF.room1preference_gold[ExcDF.task_outliers .== 0], 
                                0.0:0.025:1, alpha =0.5)
ax.hist(ExcDF.room1preference_gold[ExcDF.task_outliers .== 1], 
                                0.0:0.025:1, alpha =0.5)
x_plot = [1,1] .* mR1Chance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mR1Chance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("Room 1 preferences (Gold)"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])

ax = subplot(2,4,3)
ax.hist(ExcDF.room1preference_bomb[ExcDF.task_outliers .== 0], 
                                0.0:0.025:1, alpha =0.5)
ax.hist(ExcDF.room1preference_bomb[ExcDF.task_outliers .== 1], 
                                0.0:0.025:1, alpha =0.5)
x_plot = [1,1] .* mR1Chance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mR1Chance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("Room 1 preferences (Bomb)"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])

grange = -15:1:25
ax = subplot(2,4,1 + 4)
ax.hist(ExcDF.collected_gold[ExcDF.task_outliers .== 0] .-
        ExcDF.collected_bomb[ExcDF.task_outliers .== 0], 
        grange, alpha =0.5)
ax.hist(ExcDF.collected_gold[ExcDF.task_outliers .== 1] .- 
        ExcDF.collected_bomb[ExcDF.task_outliers .== 1], 
        grange, alpha =0.5)
x_plot = [1,1] .* mBGChance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mBGChance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("net gold - bomb"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])
        

grange = 23:1:50
ax = subplot(2,4,2 + 4)
ax.hist(ExcDF.collected_gold[ExcDF.task_outliers .== 0], 
                                grange, alpha =0.5)
ax.hist(ExcDF.collected_gold[ExcDF.task_outliers .== 1], 
                                grange, alpha =0.5)
x_plot = [1,1] .* mGChance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mGChance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("collected gold"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])

ax = subplot(2,4,3+4)
ax.hist(ExcDF.collected_bomb[ExcDF.task_outliers .== 0], 
                                grange, alpha =0.5)
ax.hist(ExcDF.collected_bomb[ExcDF.task_outliers .== 1], 
                                grange, alpha =0.5)
x_plot = [1,1] .* mBChance; y_plot = ax.get_ylim()
ax.plot(x_plot, y_plot, "r")
x_plot = [1,1] .* mBChance_plus; 
ax.plot(x_plot, y_plot, "--r")
ax.set_ylim(y_plot)
ax.set_xlabel("collected bomb"); ax.set_ylabel("count")
ax.legend(["chance","95% chance","task included","task excluded"])


y = ExcDF.comprehension1_error .+ 
    ExcDF.comprehension2_error .+
    ExcDF.comprehension3_error
ax = subplot(2,4,4)
ax.hist(y[ExcDF.task_outliers .== 0],  -0.25:0.5:20.25, alpha =0.5)
ax.hist(y[ExcDF.task_outliers .== 1],  -0.25:0.5:20.25, alpha =0.5)
ax.set_xlabel("comprehension errors"); ax.set_ylabel("count")
ax.legend(["task included","task excluded"])


ax = subplot(2,4,4 + 4)
ax.hist(ExcDF.attention_check_fail[ExcDF.outliers .== 0], -0.25:0.5:3.25, alpha =0.5)
ax.hist(ExcDF.attention_check_fail[ExcDF.outliers .== 1], -0.25:0.5:3.25, alpha =0.5)
ax.set_xlabel("attention check failed"); ax.set_ylabel("count")
ax.legend(["included","excluded"])

tight_layout()
savefig(Path_Load * "Exclusion.pdf")
savefig(Path_Load * "Exclusion.svg")
savefig(Path_Load * "Exclusion.png")
