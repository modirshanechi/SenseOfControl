################################################################################
# Code for plotting parameter inference results
# ---> Fig 2G2
# ---> Fig 4A1 and B
# ---> Fig 5E2
# ---> Fig S5B-C
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


save_name_tag = "Parameters/"
infdata_path(i) = "MCMC_sub" * string(i) * ".jld2"

# ----------------------------------------------------------------------
# Plotting parameters
# ----------------------------------------------------------------------
function plot_parameters(chn_dfs, BMS, SavePath)
    Nsub = length(chn_dfs)
    l_hat = zeros(Nsub); logl_hat = zeros(Nsub); β_hat = zeros(Nsub); 
    m_hat = zeros(Nsub); pm = zeros(Nsub);
    for i = 1:Nsub
        chn_df = chn_dfs[i]
        pm[i], m_hat[i] = findmax(BMS.exp_M[i,:])
        β_hat[i] = mean(chn_df.β)
        logl_hat[i] = mean(chn_df.logl)
        l_hat[i] = exp(mean(chn_df.logl))
    end

    # --------------------------------------------------------------------------
    # l-hat
    # --------------------------------------------------------------------------
    ps = 0.0:0.001:1.
    y = l_hat; x = Int.(m_hat)
    figure(figsize = (5,5))
    ax = subplot(1,1,1)
    for i = unique(x)
        ax.plot(ps, ps .^ mean(y[x .== i]), color = MainColors.lcol[i], lw=2.5)
    end
    for i = eachindex(x)
        ax.plot(ps, ps .^ y[i], color = MainColors.lcol[x[i]],alpha=0.1)
    end
    ax.plot(ps, ps, "--k", alpha = 0.5)
    ax.set_xlim([0,1]);ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_xlabel("p_max"); ax.set_ylabel("p_max^l-hat")
    tight_layout()
    savefig(SavePath * "LHat_ps.pdf")
    savefig(SavePath * "LHat_ps.png")
    savefig(SavePath * "LHat_ps.svg")

    c = BMS.exp_M[:,end] .- BMS.exp_M[:,1]
    y = logl_hat; 
    figure()
    ax = subplot(1,1,1)
    ax.plot([0,0],[-1.05,1.05],"--k")
    temp_min = findmin(y)[1] - 0.05
    temp_max = findmax(y)[1] + 0.05
    ax.plot([temp_min,temp_max],[0,0],"--k")
    ax.plot([temp_min,temp_max],[-1,-1],"--k", alpha = 0.1)
    ax.plot([temp_min,temp_max],[+1,+1],"--k", alpha = 0.1)
    for i = unique(x)
        ax.plot(y[x .== i],c[x .== i],".",color = MainColors.lcol[i], alpha=0.5)
    end
    ax.set_ylim([-1.05,1.05]); ax.set_xlim([temp_min,temp_max])
    ax.set_xlabel("log l-hat"); ax.set_ylabel("p(l>1) - p(l<1)")
    tight_layout()

    savefig(SavePath * "LHat_pm.pdf")
    savefig(SavePath * "LHat_pm.png")
    savefig(SavePath * "LHat_pm.svg")

    # --------------------------------------------------------------------------
    # β-hat
    # --------------------------------------------------------------------------
    y = β_hat; y[y .> 15] .= 15; y_range = 0:0.1:15
    figure(figsize = (5,5))
    ax = subplot(1,1,1)
    for i = unique(x)
        ax.hist(y[x .== i], 
                y_range, 
                color = MainColors.lcol[i], alpha = 0.5)
    end
    ax.set_xlim([0,y_range[end]])
    ax.set_xlabel("β-hat"); ax.set_ylabel("count")
    tight_layout()
    savefig(SavePath * "BHat.pdf")
    savefig(SavePath * "BHat.png")
    savefig(SavePath * "BHat.svg")
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
    subjectIDs = data.subjectIDs

    # ----------------------------------------------------------------------
    # Loading inference data
    # ----------------------------------------------------------------------
    infdata = [load(expspec.fig_path * infdata_path(i_sub)) for i_sub = subjectIDs]
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    
    # ----------------------------------------------------------------------
    # Plotting parameters
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        BMS = BMSdata["BMSEmp"]
        chn_dfs = [d["FITresultsEmpl"].chnemp_df for d = infdata]
        chn_dfs = chn_dfs[BMSdata["EmpSubjects"]]
        
        plot_parameters(chn_dfs, BMS,
                        expspec.fig_path * save_name_tag)
    else
        BMS = BMSdata["BMSEmp"][1]
        chn_dfs = [d["FITresultsEmpl"][1].chnemp_df for d = infdata]
        chn_dfs = chn_dfs[BMSdata["EmpSubjects"][1]]
        
        plot_parameters(chn_dfs, BMS,
                        expspec.fig_path * save_name_tag * "Gold")
        
        BMS = BMSdata["BMSEmp"][2]
        chn_dfs = [d["FITresultsEmpl"][2].chnemp_df for d = infdata]
        chn_dfs = chn_dfs[BMSdata["EmpSubjects"][2]]
        
        plot_parameters(chn_dfs, BMS,
                        expspec.fig_path * save_name_tag * "Bomb")
        
    end
end


# ----------------------------------------------------------------------
# Only for experiment 2
# ----------------------------------------------------------------------
i_exp = 2

# ----------------------------------------------------------------------
# Loading data
# ----------------------------------------------------------------------
expspec = ExperimentSpecification(i_exp)
data = load_clean_data(expspec);
subjectIDs = data.subjectIDs

# ----------------------------------------------------------------------
# Loading inference data
# ----------------------------------------------------------------------
infdata = [load(expspec.fig_path * infdata_path(i_sub)) for i_sub = subjectIDs]
BMSdata = load(expspec.fig_path * "BMS.jld2")
BMSAll = BMSdata["BMSAll"]; 

mAll_hat = [[findmax(B.exp_M[i,:])[2] for i = 1:size(B.exp_M)[1]] for B=BMSAll]

EmpSubjectsalone = [(1:length(m))[(m .== 3)] for m = mAll_hat] 
EmpSubjectsGen   = [(1:length(m))[(m .== 4)] for m = mAll_hat] 
EmpSubjects = [(1:length(m))[(m .== 3) .|| (m .== 4)] for m = mAll_hat] 

doubleEmpSubjectsSet = [intersect(EmpSubjects...),
                        intersect(EmpSubjectsalone...),
                        intersect(EmpSubjectsGen...)]
BMSEmpNames = ["Gen+Emp", "EmpAlone", "General"]

# ---------------------------------------------------------------------
# Choosing double-Emp subjects
# ---------------------------------------------------------------------
for i_plot = eachindex(BMSEmpNames)
    doubleEmpSubjects = doubleEmpSubjectsSet[i_plot]
    chn_dfs = [
            [d["FITresultsEmpl"][i].chnemp_df for d = infdata][doubleEmpSubjects]
            for i = 1:2]

    # ---------------------------------------------------------------------
    # Log-hat in Gold versus Bomb
    # ---------------------------------------------------------------------
    logl_hat = [[mean(d.logl) for d = chn_df] for chn_df = chn_dfs]
    dlogl_hat = [[std(d.logl) for d = chn_df] for chn_df = chn_dfs]


    io_log = IOBuffer()

    figure(figsize=(4,4))

    ax = subplot(1,1,1)
    Ys = logl_hat; dYs = dlogl_hat
    ρ = round(cor(Ys...),digits = 2)
    ax.plot(Ys[2],Ys[1],".k",alpha = 0.5)
    ax.errorbar(Ys[2],Ys[1],xerr = dYs[2], yerr=dYs[1],color="k",alpha=0.2,
                    linewidth=1,drawstyle="steps",linestyle="",capsize=0)
    x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
    x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
    ax.plot([x_min,x_max],[x_min,x_max],"--k")
    ax.plot([x_min,x_max],[0,0],"--k")
    ax.plot([0,0],[x_min,x_max],"--k")
    ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
    Test_result = CorrelationTest(Ys...)
    pval = pvalue(Test_result)
    logBF = BIC_CorrelationTest(Ys...)
    log_test(io_log,
        "Testing: correlation between per-subject mean log-l-hat (logl_hat) " *
        "estimated from Gold-condition trials vs Bomb-condition trials " *
        "(Experiment 2, double-Emp subjects) -- Figure: lhat_paired",
        Test_result; pval = pval, logBF = logBF)
    ax.set_ylabel("log l-hat-G")
    ax.set_xlabel("log l-hat-B")
    ax.legend(["ρ = " * string(ρ)])
    ax.set_title("pval = " * Func_pval_string(pval) *
                ", lBF = " * Func_logBF_string(logBF))

    tight_layout()
    savefig(expspec.fig_path * save_name_tag * "lhat_paired" * BMSEmpNames[i_plot] * ".pdf")
    savefig(expspec.fig_path * save_name_tag * "lhat_paired" * BMSEmpNames[i_plot] * ".png")
    savefig(expspec.fig_path * save_name_tag * "lhat_paired" * BMSEmpNames[i_plot] * ".svg")
    open(expspec.fig_path * save_name_tag * "lhat_paired_stats" * BMSEmpNames[i_plot] * ".txt", "w") do f
        write(f, String(take!(io_log)))
    end
end
