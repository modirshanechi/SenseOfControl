################################################################################
# Code for survey analysis
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

Path_Save = "src/GoldExp3Data/Figures/GroupLevelSurvey/"
Path_Load_Inf = "src/GoldExp3Data/Figures/"
Path_Load = "data/Experiment3/clean/"

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.outliers .== 0]

# ------------------------------------------------------------------------------
# Cleaning survey data
# ------------------------------------------------------------------------------
metaDF_survey = DataFrame(CSV.File(Path_Load * "SurveyMetaData.csv"))
surveyDF = DataFrame(CSV.File(Path_Load * "SurveyData.csv"))

metaDF_survey_clean = metaDF_survey[metaDF_survey.Attention .== 0,:]

# Removing participants of who didn't respond to even one question
QNames = metaDF_survey_clean.QNames
subjectIDs = subjectIDs[isnan.(sum(Matrix(
                        surveyDF[subjectIDs,QNames]),dims=2)).==0]

# ------------------------------------------------------------------------------
# Load group level inference data
# ------------------------------------------------------------------------------
BMSdata = load(Path_Load_Inf * "GroupLevel/BMS.jld2")
BMSdataLHat = load(Path_Load_Inf * "GroupLevel/BMSbasedLHat.jld2")

# ------------------------------------------------------------------------------
# Adding model info
# ------------------------------------------------------------------------------
sDFClean = filter(row -> row.Subject ∈ subjectIDs, surveyDF)
select!(sDFClean, "Subject", 
    "age", "gender", "sex", "education", "strategy")
    
sDFClean.Model = ["" for i = 1:length(subjectIDs)]
sDFClean.lhat = NaN .* zeros(length(subjectIDs))
sDFClean.valid = zeros(length(subjectIDs)) .== 1

ModelLegend = ["l<1","l=1","l>1","Nact"]

for i_sub = subjectIDs
    temp = load(Path_Load_Inf * "inference_data_sub" * string(i_sub) * ".jld2")
    chnemp_df = temp["chnemp_df"]

    BMSAll_temp = BMSdata["BMSAll"]
    BMS_temp = BMSdata["BMS"]
    BMS_sIDs = BMSdata["subjectIDs"]
    chnemp_dfs_temp = BMSdataLHat["chnemp_dfs"]
    subjectIDs_L_temp = BMSdataLHat["subjectIDsL"]

    pmAll = BMSAll_temp.exp_M[BMS_sIDs .== i_sub,:][:]
    mAll_hat = findmax(pmAll)[2]
    if mAll_hat == 3
        pm = BMS_temp.exp_M[subjectIDs_L_temp .== i_sub,:][:]
        m_hat = findmax(pm)[2]
        sDFClean.Model[sDFClean.Subject .== i_sub] = [ModelLegend[m_hat]]

        df_temp = chnemp_dfs_temp[subjectIDs_L_temp .== i_sub][1]
        if m_hat == 1
            ls = df_temp.l0
        elseif m_hat == 2
            ls = ones(length(df_temp.l0))
        elseif m_hat == 3
            ls = df_temp.l2
        end
        l_hat = mean(ls)
        sDFClean.lhat[sDFClean.Subject .== i_sub] = [l_hat]
        sDFClean.valid[sDFClean.Subject .== i_sub] = [true]
    elseif mAll_hat == 2
        sDFClean.Model[sDFClean.Subject .== i_sub] = [ModelLegend[4]]
        sDFClean.valid[sDFClean.Subject .== i_sub] = [true]
    end
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
        push!(QTypes, "IUC")
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
savefig(Path_Save * "SurveyStats_details.png")
savefig(Path_Save * "SurveyStats_details.pdf")
savefig(Path_Save * "SurveyStats_details.svg")


# ------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------
X_PCA = Matrix(sDFClean[:, QNamesClean])
X_PCA = Matrix(X_PCA')
M_PCA = fit(PCA, X_PCA)
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
savefig(Path_Save * "SurveyStatsPCA_details.png")
savefig(Path_Save * "SurveyStatsPCA_details.pdf")
savefig(Path_Save * "SurveyStatsPCA_details.svg")


# ------------------------------------------------------------------------------
# survey responses: 2 groups
# ------------------------------------------------------------------------------
ModelSet_plot = ["l<1","l=1","l>1"]; 
Color_plot = MainColors.lcol#[MainColors.lcol[1],MainColors.lcol[3]]

Y = predict(M_PCA, Matrix(sDFClean[:, QNamesClean])')'
Ls = [sDFClean.lhat]
Ms = [sDFClean.Model]
figure(figsize=(13.33,3))
for i_q = 1:4
    ax = subplot(1,4,i_q)
    y = Y[:,i_q]
    m = Ms[1]; l = log.(Ls[1])
    
    inds = ismissing.(l) .== 0
    y = y[inds]; m = m[inds]; l = l[inds];
    inds = isnan.(l) .== 0
    y = y[inds]; m = m[inds]; l = l[inds];
    l = Float64.(l)
    
    Test_result = CorrelationTest(y,l)
    @show Test_result
    pval = pvalue(Test_result); ρ = cor(l,y)
    logBF = BIC_CorrelationTest(y,l)
    for j = 1:3
        if sum(m .== ModelSet_plot[j]) > 0
            y_temp = y[m .== ModelSet_plot[j]]
            l_temp = l[m .== ModelSet_plot[j]]
            ax.plot(l_temp,y_temp,".",color = Color_plot[j],alpha=0.5)
        end
    end
    ax.set_title("PC " * string(i_q) * 
                 "; p=" * Func_pval_string(pval) *
                ", lBF=" * Func_logBF_string(logBF))
    ax.set_xlabel("log l-hat"); ax.set_ylabel("PC");
    ax.legend(["ρ=" * string(round(ρ,digits=3))])
end
tight_layout()
savefig(Path_Save * "SurveyResultsPCA_details_LHat.png")
savefig(Path_Save * "SurveyResultsPCA_details_LHat.pdf")
savefig(Path_Save * "SurveyResultsPCA_details_LHat.svg")


# ------------------------------------------------------------------------------
# survey responses: 2 groups
# ------------------------------------------------------------------------------
ModelSet_plot = ["l<1","l=1","l>1"]; 
Color_plot = MainColors.lcol#[MainColors.lcol[1],MainColors.lcol[3]]

QTypesU = unique(QTypes)
Y = [mean(Matrix(sDFClean[:, QNamesClean[QTypes .== q]]),dims=2) for q = QTypesU]
Y = hcat(Y...)
Ls = [sDFClean.lhat]
Ms = [sDFClean.Model]
figure(figsize=(13.33,3))
for i_q = 1:4
    ax = subplot(1,4,i_q)
    y = Y[:,i_q]
    m = Ms[1]; l = log.(Ls[1])
    
    inds = ismissing.(l) .== 0
    y = y[inds]; m = m[inds]; l = l[inds];
    inds = isnan.(l) .== 0
    y = y[inds]; m = m[inds]; l = l[inds];
    l = Float64.(l)
    
    Test_result = CorrelationTest(y,l)
    @show Test_result
    pval = pvalue(Test_result); ρ = cor(l,y)
    logBF = BIC_CorrelationTest(y,l)
    for j = 1:3
        if sum(m .== ModelSet_plot[j]) > 0
            y_temp = y[m .== ModelSet_plot[j]]
            l_temp = l[m .== ModelSet_plot[j]]
            ax.plot(l_temp,y_temp,".",color = Color_plot[j],alpha=0.5)
        end
    end
    ax.set_title(QTypesU[i_q] * 
                 "; p=" * Func_pval_string(pval) *
                ", lBF=" * Func_logBF_string(logBF))
    ax.set_xlabel("log l-hat"); ax.set_ylabel("PC");
    ax.legend(["ρ=" * string(round(ρ,digits=3))])
end
tight_layout()
savefig(Path_Save * "SurveyResultsRaw_details_LHat.png")
savefig(Path_Save * "SurveyResultsRaw_details_LHat.pdf")
savefig(Path_Save * "SurveyResultsRaw_details_LHat.svg")




# ------------------------------------------------------------------------------
# Bootstrapping
# ------------------------------------------------------------------------------
N_boot = 10000
X_PCA = Matrix(sDFClean[:, QNamesClean])
X_PCA = Matrix(X_PCA')


M_PCA_samples = []
for i_boot = 1:N_boot
    X_PCA_samp = X_PCA[:,rand(1:size(X_PCA)[2],size(X_PCA)[2])]
    push!(M_PCA_samples,fit(PCA, X_PCA_samp, maxoutdim=36, pratio = 1.))
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
# variances
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
savefig(Path_Save * "SurveyStatsPCABoot_details.png")
savefig(Path_Save * "SurveyStatsPCABoot_details.pdf")
savefig(Path_Save * "SurveyStatsPCABoot_details.svg")


# ------------------------------------------------------------------------------
# within similarity
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
savefig(Path_Save * "SurveyStatsPCABoot_details2.png")
savefig(Path_Save * "SurveyStatsPCABoot_details2.pdf")
savefig(Path_Save * "SurveyStatsPCABoot_details2.svg")

# ------------------------------------------------------------------------------
# Y and lodaings of first 4 PCs
# ------------------------------------------------------------------------------
for i_q = 1:4
    Loadings = zeros(length(QNamesClean),N_boot)
    Loadings_sum = zeros(length(unique(QTypes)),N_boot)
    
    iq_sing = findmax(abs.(M_PCA.proj[:,i_q]))[2]
    q_sign = sign(M_PCA.proj[iq_sing,i_q])
    for i_m = 1:N_boot
        m = M_PCA_samples[i_m]
        i_qcor = Int(coresponding_inds[i_q,i_m])
        Loadings[:,i_m] = m.proj[:,i_qcor] .* 
                        sign(m.proj[iq_sing,i_qcor]) .* q_sign
        Loadings_sum[:,i_m] = 
                [mean(Loadings[QTypes .== q,i_m]) for q = unique(QTypes)]
    end
    
    figure(figsize=(9,6))
    ax = subplot(1,1,1)
    my = mean(Loadings,dims=2)[:]; dy = std(Loadings,dims=2)[:]
    ax.bar(1:length(QNamesClean),my,color="k",alpha = 0.5)
    ax.errorbar(1:length(my),my,yerr=dy,color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([0,length(QNamesClean)+1],[0,0],"--k")
    ax.set_title("PC " * string(i_q))
    ax.set_xticks(1:length(QNamesClean)); ax.set_xticklabels(QNamesClean,rotation=90)
    ax.set_xlim([0,length(QNamesClean)+1]); 
    tight_layout()
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_PC" * string(i_q) * ".png")
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_PC" * string(i_q) * ".pdf")
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_PC" * string(i_q) * ".svg")


    figure(figsize=(6,6))
    ax = subplot(1,1,1)
    my = mean(Loadings_sum,dims=2)[:]; dy = std(Loadings_sum,dims=2)[:]
    ax.bar(1:length(my),my,color="k",alpha = 0.5)
    ax.errorbar(1:length(my),my,yerr=dy,color="k",
            linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([0,length(unique(QTypes))+1],[0,0],"--k")
    ax.set_title("PC " * string(i_q))
    ax.set_xticks(1:length(unique(QTypes))); ax.set_xticklabels(unique(QTypes),rotation=90)
    ax.set_xlim([0,length(unique(QTypes))+1]); 
    tight_layout()
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_sum_PC" * string(i_q) * ".png")
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_sum_PC" * string(i_q) * ".pdf")
    savefig(Path_Save * "SurveyResultsPCABoot_loadings_sum_PC" * string(i_q) * ".svg")
end


