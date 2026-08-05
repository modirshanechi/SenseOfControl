################################################################################
# Code for survey analysis for Experiment 3
# ---> Fig S18-S19
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random
using Turing, MCMCChains, Distributions
using DataFrames
using CSV
using JLD2
using AdvancedMH
using HypothesisTests
using MultivariateStats

import StatsPlots
import StatsBase: countmap


cosine_sim(x,y) = dot(x,y)/(norm(x)*norm(y))

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42


rng = Xoshiro(2024)

load_name_tag = ""
save_name_tag = "Survey/"
infdata_path(i) = "MCMC_sub" * string(i) * ".jld2"

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
expspec = ExperimentSpecification(3)
data = load_clean_data(expspec);
ExcDF = data.ExcDF; dataDF = data.selDF;
subjectIDs = ExcDF.subject[ExcDF.outliers .== 0]

# ------------------------------------------------------------------------------
# Cleaning survey data
# ------------------------------------------------------------------------------
metaDF_survey = data.metaDF_survey; 
surveyDF = data.surveyDF; 
metaDF_survey_clean = metaDF_survey[metaDF_survey.Attention .== 0,:]

QNames = metaDF_survey_clean.QNames

# ------------------------------------------------------------------------------
# Load group level inference data
# ------------------------------------------------------------------------------
BMSdata = load(expspec.fig_path * load_name_tag *  "BMS.jld2")
BMSAll = BMSdata["BMSAll"]

infdata = [load(expspec.fig_path * infdata_path(i_sub)) for i_sub = subjectIDs]
mAllhat = [findmax(BMSAll.exp_M[i,:])[2] for i = 1:size(BMSAll.exp_M)[1]]

# ------------------------------------------------------------------------------
# Adding model info
# ------------------------------------------------------------------------------
sDFClean = filter(row -> row.Subject ∈ subjectIDs, surveyDF)
select!(sDFClean, "Subject", 
    "age", "gender", "sex", "education", "strategy")

sDFClean.loglhat = NaN .* zeros(length(subjectIDs))
sDFClean.valid = zeros(length(subjectIDs)) .== 1

for i = eachindex(subjectIDs)
    i_sub = subjectIDs[i]
    chnemp_df = infdata[i]["FITresultsEmpl"].chnemp_df
    sDFClean.loglhat[sDFClean.Subject .== i_sub] = [mean(chnemp_df.logl)]
    sDFClean.valid[sDFClean.Subject .== i_sub] = [mAllhat[i] ∈ [3,4]]
end

# ------------------------------------------------------------------------------
# Adding survey responses
# ------------------------------------------------------------------------------
QNamesClean = Vector{String}([])
QTypes = Vector{String}([])
for q = QNames
    if q[1:3] == "LOC"
        q2 = metaDF_survey_clean.IPC[metaDF_survey_clean.QNames .== q][1]
        push!(QTypes, "LOC_" * q2)
        q2 = q2 * "_" * q
    else
        q2 = q
        push!(QTypes, "IUS")
    end
    sDFClean[:,q2] .= 0.
    for i_sub = subjectIDs
        sDFClean[sDFClean.Subject .== i_sub, q2] .= 
                    surveyDF[surveyDF.Subject .== i_sub, q]
    end
    push!(QNamesClean,q2)
end
QTypes = QTypes[sortperm(QNamesClean)]
sort!(QNamesClean)

# Keeping only Emp-l and general participants
sDFClean = sDFClean[sDFClean.valid,:]

# ------------------------------------------------------------------------------
# General stats
# ------------------------------------------------------------------------------
fig = figure(figsize=(10,8))
Y = cor(Matrix(sDFClean[:,QNamesClean]))
ax = subplot(1,1,1)
cp = ax.imshow(Y,vmin=-1.0,vmax=1.0,cmap="RdBu")
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:(length(QNamesClean)-1)); ax.set_xticklabels(QNamesClean,rotation=90)
ax.set_yticks(0:(length(QNamesClean)-1)); ax.set_yticklabels(QNamesClean)
ax.set_title("cross correlation")

tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyStats.png")
savefig(expspec.fig_path * save_name_tag * "SurveyStats.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyStats.svg")

# ------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------
X_PCA = Matrix(sDFClean[:, QNamesClean])
X_PCA = Matrix(X_PCA')
μ = mean(X_PCA, dims=2)
σ = std(X_PCA, dims=2); σ[σ .== 0] .= 1
X_scaled = (X_PCA .- μ) ./ σ
M_PCA = fit(PCA, X_scaled, mean = 0)
figure(figsize=(5,10))
ax = subplot(2,1,1)
ax.plot(M_PCA.prinvars ./ M_PCA.tvar,"-o",color = "k")
ax.set_xlabel("PC"); ax.set_ylabel("PC var")
ax = subplot(2,1,2)
ax.plot(cumsum(M_PCA.prinvars) ./ M_PCA.tvar,"-o",color = "k")
ax.plot([0,length(M_PCA.prinvars)-1],[0.8,0.8],"--r")
ax.plot([0,length(M_PCA.prinvars)-1],[0.9,0.9],"-r")
ax.set_xlabel("PC"); ax.set_ylabel("cumulative PC var")
tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCA.png")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCA.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCA.svg")


# ------------------------------------------------------------------------------
# Log-l hat versus PCs
# ------------------------------------------------------------------------------
Y = predict(M_PCA, X_scaled)'
figure(figsize=(11.69,3))
for i_q = 1:4
    ax = subplot(1,4,i_q)
    y = Y[:,i_q]; l = sDFClean.loglhat
    
    inds = ismissing.(l) .== 0
    y = y[inds]; l = l[inds];
    inds = isnan.(l) .== 0
    y = y[inds]; l = l[inds];
    l = Float64.(l)
    
    Test_result = CorrelationTest(y,l)
    @show Test_result
    pval = pvalue(Test_result); ρ = cor(l,y)
    logBF = BIC_CorrelationTest(y,l)
    ax.plot(l,y,".",color = "k",alpha=0.5)
    ymin,ymax = ax.get_ylim()
    ax.plot([0,0],[ymin,ymax],"--k"); ax.set_ylim([ymin,ymax])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("PC " * string(i_q) * 
                 "; p=" * Func_pval_string(pval) *
                ", lBF=" * Func_logBF_string(logBF))
    ax.set_xlabel("log l-hat"); ax.set_ylabel("PC");
    ax.legend(["ρ=" * string(round(ρ,digits=3))])
end
tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyResultsPCA_LHat.png")
savefig(expspec.fig_path * save_name_tag * "SurveyResultsPCA_LHat.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyResultsPCA_LHat.svg")


# ------------------------------------------------------------------------------
# Log-l hat versus raw scores
# ------------------------------------------------------------------------------
QTypesU = unique(QTypes)
Y = [mean(Matrix(sDFClean[:, QNamesClean[QTypes .== q]]),dims=2) for q = QTypesU]
Y = hcat(Y...)
figure(figsize=(11.69,3))
for i_q = 1:4
    ax = subplot(1,4,i_q)
    y = Y[:,i_q]; l = sDFClean.loglhat
    
    inds = ismissing.(l) .== 0
    y = y[inds]; l = l[inds];
    inds = isnan.(l) .== 0
    y = y[inds]; l = l[inds];
    l = Float64.(l)
    
    Test_result = CorrelationTest(y,l)
    @show Test_result
    pval = pvalue(Test_result); ρ = cor(l,y)
    logBF = BIC_CorrelationTest(y,l)
    ax.plot(l,y,".",color = "k",alpha=0.5)
    ymin,ymax = ax.get_ylim()
    ax.plot([0,0],[ymin,ymax],"--k"); ax.set_ylim([ymin,ymax])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(QTypesU[i_q] * 
                 "; p=" * Func_pval_string(pval) *
                ", lBF=" * Func_logBF_string(logBF))
    ax.set_xlabel("log l-hat"); ax.set_ylabel("survey score");
    ax.legend(["ρ=" * string(round(ρ,digits=3))])
end
tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyResultsRaw_LHat.png")
savefig(expspec.fig_path * save_name_tag * "SurveyResultsRaw_LHat.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyResultsRaw_LHat.svg")


# ------------------------------------------------------------------------------
# Bootstrapping
# ------------------------------------------------------------------------------
N_boot = 10000
X_PCA = Matrix(sDFClean[:, QNamesClean])
X_PCA = Matrix(X_PCA')

M_PCA_samples = []
for i_boot = 1:N_boot
    X_PCA_samp = X_PCA[:,rand(rng, 1:size(X_PCA)[2],size(X_PCA)[2])]
    μ = mean(X_PCA_samp, dims=2)
    σ = std(X_PCA_samp, dims=2); σ[σ .== 0] .= 1
    X_PCA_samp = (X_PCA_samp .- μ) ./ σ
    push!(M_PCA_samples,fit(PCA, X_PCA_samp, maxoutdim=36, pratio = 1., mean = 0))
end

main_qs = 1:30
coresponding_inds = zeros(main_qs,N_boot)
max_coxine = zeros(main_qs,N_boot)
for i_q = main_qs
    @show i_q
    for i_m = 1:N_boot
        m = M_PCA_samples[i_m]
        temp = abs.([cosine_sim(M_PCA.proj[:,i_q],m.proj[:,j]) for j = main_qs])
        max_coxine[i_q,i_m] , coresponding_inds[i_q,i_m] = findmax(temp)
    end
end

# ------------------------------------------------------------------------------
# Variances
# ------------------------------------------------------------------------------
figure(figsize=(10,10))
ax = subplot(2,2,1)
ax.plot(M_PCA.prinvars ./ M_PCA.tvar,"-o",color = "k")
ax.set_xlabel("PC"); ax.set_ylabel("PC var")
ax.set_title("raw observation")
ax = subplot(2,2,2)
Ys = [m.prinvars ./ m.tvar for m = M_PCA_samples]
my = mean(Ys); dy = std(Ys)
ax.plot(1:length(my),my,"-o",color = "k")
ax.errorbar(1:length(my),my,yerr=dy,color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
ax.set_xlabel("PC"); ax.set_ylabel("PC var")
ax.set_title("bootstrapped")
ax = subplot(2,2,3)
ax.plot(cumsum(M_PCA.prinvars) ./ M_PCA.tvar,"-o",color = "k")
ax.plot([0,length(M_PCA.prinvars)-1],[0.8,0.8],"--r")
ax.plot([0,length(M_PCA.prinvars)-1],[0.9,0.9],"-r")
ax.set_xlabel("PC"); ax.set_ylabel("cumulative PC var")
ax = subplot(2,2,4)
Ys = [cumsum(m.prinvars) ./ m.tvar for m = M_PCA_samples]
my = mean(Ys); dy = std(Ys)
ax.plot(1:length(my),my,"-o",color = "k")
ax.errorbar(1:length(my),my,yerr=dy,color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
ax.plot([0,length(M_PCA.prinvars)-1],[0.8,0.8],"--r")
ax.plot([0,length(M_PCA.prinvars)-1],[0.9,0.9],"-r")
ax.set_xlabel("PC"); ax.set_ylabel("cumulative PC var")            
tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot.png")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot.svg")


# ------------------------------------------------------------------------------
# Within similarity
# ------------------------------------------------------------------------------
figure(figsize=(7,5))
ax = subplot(1,1,1)
my = mean(max_coxine,dims=2)[:]; dy = std(max_coxine,dims=2)[:]
ax.plot(1:length(my),my,"-o",color = "k")
ax.errorbar(1:length(my),my,yerr=dy,color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
ax.plot([1,length(my)],[1,1] .* mean(my[(length(my)-5):end]),"-o",color="r")
ax.set_xlabel("PC"); ax.set_ylabel("max corresponding cosine loading")
ax.set_title("bootstrapped")
ax.set_ylim([0.4,1]); ax.set_xlim([0,length(my)+1])  
tight_layout()
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot2.png")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot2.pdf")
savefig(expspec.fig_path * save_name_tag * "SurveyStatsPCABoot2.svg")

