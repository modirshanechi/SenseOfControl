################################################################################
# Code for comparing BIC-based log evidence vs. bridge-based log evidence
# ---> Fig S14
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random, Statistics
using DataFrames
using CSV
using JLD2

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)
save_name_tag = "ModelSelection/ControlBridgeSampling/"

# ----------------------------------------------------------------------
# BIC-vs-bridge plotting
# ----------------------------------------------------------------------
function plot_L1L2(L1, L2, SavePath; method_label = "Bridge Sampling")
    ΔL1 = hcat([L1[:,i] .- L1[:,3] for i = [1,2,4]]...)
    ΔL2 = hcat([L2[:,i] .- L2[:,3] for i = [1,2,4]]...)

    xlabel_str = "log P(D|M) - log P(D|Emp-l); $(method_label)"
    ylabel_str = "log P(D|M) - log P(D|Emp-l); BIC"
    
    figure(figsize=(16,5))
    for i = 1:3
        x = ΔL2[:,i]
        y = ΔL1[:,i]

        off_diag = (sign.(x) .!= sign.(y)) .&& (x .!= 0) .&& (y .!= 0)
        num_off  = sum(off_diag); num_on  = sum(1 .- off_diag)
        ρ        = round(cor(x, y), digits=5)

        ax = subplot(1,3,i)
        ax.plot(x[.!off_diag], y[.!off_diag], ".k", alpha=0.5,
                    label="same sign (n=$(num_on))")
        ax.plot(x[off_diag], y[off_diag], ".r", alpha=0.5,
                    label="opposite sign (n=$(num_off))")
        minx, maxx = ax.get_xlim()
        ax.plot([minx, maxx], [0,0], "--k", linewidth=1)
        ax.plot([0,0], [minx, maxx], "--k", linewidth=1)
        ax.plot([minx, maxx], [minx, maxx], "--k", linewidth=1)
        ax.set_xlim([minx, maxx]); ax.set_ylim([minx, maxx])
        ax.set_aspect(1)
        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(ylabel_str)
        ax.set_title("M = " * ModelNames.MAll[[1,2,4][i]] * "; ρ = $(ρ)")
        ax.legend()
    end
    tight_layout()
    savefig(SavePath * "logLcomparison.pdf")
    savefig(SavePath * "logLcomparison.png")
    savefig(SavePath * "logLcomparison.svg")
end

# ----------------------------------------------------------------------
# loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    # ----------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec);
    subjectIDs = data.subjectIDs

    # ----------------------------------------------------------------------
    # load inference data
    # ----------------------------------------------------------------------
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    LMatBIC = BMSdata["L_matrixAll"]
    Bridgedata = load(expspec.fig_path * "BridgeSampLmatrix.jld2")
    LMatBridg = Bridgedata["L_matrixAll"]

    # ----------------------------------------------------------------------
    # extracting log-evidence matrices and plotting the comparison
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        L1 = BMSdata["L_matrixAll"]
        L2 = Bridgedata["L_matrixAll"]
        plot_L1L2(L1, L2, expspec.fig_path * save_name_tag)
    else
        L1 = BMSdata["L_matrixAll"][1]
        L2 = Bridgedata["L_matrixAll"][1]
        plot_L1L2(L1, L2, expspec.fig_path * save_name_tag * "Gold")

        L1 = BMSdata["L_matrixAll"][2]
        L2 = Bridgedata["L_matrixAll"][2]
        plot_L1L2(L1, L2, expspec.fig_path * save_name_tag * "Bomb")
    end
end
