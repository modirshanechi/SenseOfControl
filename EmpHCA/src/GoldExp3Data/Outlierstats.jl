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

Path_Load = "data/Experiment3/clean/"

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
print(mean(surveyDF.age[ExcDF.task_outliers .== 0]))
print("+-")
println(std(surveyDF.age[ExcDF.task_outliers .== 0]))
println("Time-outs")
println(countmap(ExcDF.selection_timeout))

# ------------------------------------------------------------------------------
# reward stats
# ------------------------------------------------------------------------------
valid_rooms = [0,1,2,4,5,6,7,8,9] .+ 1
Prooms, ΔState, ΔStateDict = gold_proom_sets(); Prooms = Prooms[valid_rooms]
N_rooms = length(Prooms); Ymax = 1; Xmax = 1
Xinds = vcat([[[i,j] for j = (1:N_rooms)[(1:N_rooms) .!= i]] 
                                                for i = 1:N_rooms]...)

# chance level
Nsamp = 100000
ChanceRoom1 = zeros(Nsamp)
seed = 2024; rng = MersenneTwister(seed)
for i = 1:Nsamp
    if mod(i,10000) == 0
        @show i
    end
    Xinds_temp = Xinds[randperm(rng, length(Xinds))]
    choice_temp = Int.(zeros(length(Xinds))); 
    Xch_temp = Int.(zeros(length(Xinds)))
    for j = eachindex(Xinds_temp)
        choice_temp[j] = rand(rng,[1,2])
        Xch_temp[j] = Xinds_temp[j][choice_temp[j]]
    end
    ChanceRoom1[i] = mean(Xch_temp[[1 ∈ x for x = Xinds_temp]] .== 1)
end
mR1Chance = mean(ChanceRoom1)
mR1Chance_plus =  quantile(ChanceRoom1,0.975)

# ------------------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------------------
figure(figsize = (14,4))

ax = subplot(1,3,1)
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


y = ExcDF.comprehension1_error .+ 
    ExcDF.comprehension2_error
ax = subplot(1,3,2)
ax.hist(y[ExcDF.task_outliers .== 0],  -0.25:0.5:20.25, alpha =0.5)
ax.hist(y[ExcDF.task_outliers .== 1],  -0.25:0.5:20.25, alpha =0.5)
ax.set_xlabel("comprehension errors"); ax.set_ylabel("count")
ax.legend(["task included","task excluded"])


ax = subplot(1,3,3)
ax.hist(ExcDF.attention_check_fail[ExcDF.outliers .== 0], -0.25:0.5:3.25, alpha =0.5)
ax.hist(ExcDF.attention_check_fail[ExcDF.outliers .== 1], -0.25:0.5:3.25, alpha =0.5)
ax.set_xlabel("attention check failed"); ax.set_ylabel("count")
ax.legend(["included","excluded"])

tight_layout()
savefig(Path_Load * "Exclusion.pdf")
savefig(Path_Load * "Exclusion.svg")
savefig(Path_Load * "Exclusion.png")
