################################################################################
# Code for outlier statistics
################################################################################
using PyPlot
using EmpHCA
using Random
using DataFrames

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)

# ------------------------------------------------------------------------------
# load data 
# ------------------------------------------------------------------------------
expspec = ExperimentSpecification(2)
data = load_clean_data(expspec); ExcDF = data.ExcDF
print_outlierstats(expspec, data)

# ------------------------------------------------------------------------------
# reward stats under chance level
# ------------------------------------------------------------------------------
mGChance, mGChance_plus, mBChance, mBChance_plus,
    mBGChance, mBGChance_plus, mR1Chance, mR1Chance_plus = 
                                            room_null_stats(expspec; rng = rng)

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
savefig(expspec.fig_path * "Exclusion.pdf")
savefig(expspec.fig_path * "Exclusion.svg")
savefig(expspec.fig_path * "Exclusion.png")

