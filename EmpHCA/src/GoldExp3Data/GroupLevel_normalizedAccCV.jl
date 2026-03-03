################################################################################
# Code for evaluating and plotting normalized accuracy rate
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random, Statistics
using DataFrames
using CSV
using JLD2
using AdvancedMH
using HypothesisTests


PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExp3Data/Figures/CVAccuracy/"

Path_Load_GInf1 = "src/GoldExp1Data/Figures/GroupLevel/"
Path_Load_Inf1 = "src/GoldExp1Data/Figures/"
Path_Load_CVInf1 = "src/GoldExp1Data/Figures/CVAccuracy/"
Path_Load1 = "data/Experiment1/clean/"

Path_Load_GInf2 = "src/GoldExp2Data/Figures/GroupLevel/"
Path_Load_CVInf2 = "src/GoldExp2Data/Figures/CVAccuracy/"
Path_Load_Inf2 = "src/GoldExp2Data/Figures/"
Path_Load2 = "data/Experiment2/clean/"

Path_Load_GInf3 = "src/GoldExp3Data/Figures/GroupLevel/"
Path_Load_CVInf3 = "src/GoldExp3Data/Figures/CVAccuracy/"
Path_Load_Inf3 = "src/GoldExp3Data/Figures/"
Path_Load3 = "data/Experiment3/clean/"


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ExcDF1 = DataFrame(CSV.File(Path_Load1 * "ExclusionInfo.csv"))
dataDF1 = DataFrame(CSV.File(Path_Load1 * "SelectionData.csv"))
subjectIDs1 = ExcDF1.subject[ExcDF1.task_outliers .== 0]

ExcDF2 = DataFrame(CSV.File(Path_Load2 * "ExclusionInfo.csv"))
dataDF2 = DataFrame(CSV.File(Path_Load2 * "SelectionData.csv"))
subjectIDs2 = ExcDF2.subject[ExcDF2.task_outliers .== 0]

ExcDF3 = DataFrame(CSV.File(Path_Load3 * "ExclusionInfo.csv"))
dataDF3 = DataFrame(CSV.File(Path_Load3 * "SelectionData.csv"))
subjectIDs3 = ExcDF3.subject[ExcDF3.task_outliers .== 0]

# ----------------------------------------------------------------------
# Load group level inference data
# ----------------------------------------------------------------------
BMSdata1 = load(Path_Load_GInf1 * "BMS.jld2")
BMSdataLHat1 = load(Path_Load_CVInf1 * "BMSbasedLHat_CV.jld2")

BMSdata2 = load(Path_Load_GInf2 * "BMS.jld2")
BMSdataLHat2 = load(Path_Load_CVInf2 * "BMSbasedLHat_CV.jld2")

BMSdata3 = load(Path_Load_GInf3 * "BMS.jld2")
BMSdataLHat3 = load(Path_Load_CVInf3 * "BMSbasedLHat_CV.jld2")

# ----------------------------------------------------------------------
# Rooms
# ----------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
Data1 = []
for i_sub = subjectIDs1
    @show i_sub
    # ----------------------------------------------------------------------
    # Inference data
    # ----------------------------------------------------------------------
    temp = load(Path_Load_Inf1 * "inference_data_sub" * string(i_sub) * ".jld2")
    chnemp_df = temp["chnemp_df"]
    
    # ----------------------------------------------------------------------
    # data management
    # ----------------------------------------------------------------------
    df = dataDF1[dataDF1.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    
    as_inds = df.chosenroom .+ 1
    as = df.action .+ 1

    # ----------------------------------------------------------------------
    # consistency index
    # ----------------------------------------------------------------------
    consistency_index, cons_trials = choice_consistency(Xinds, as_inds;
                                                    ifpassinds = true)
    # ----------------------------------------------------------------------
    # baseline consistency
    # ----------------------------------------------------------------------
    βas = [1.]
    p_hats = zeros(length(βas),length(as_inds))
    for i = eachindex(βas)
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([size(Prooms[x[1]])[1], 
                          size(Prooms[x[2]])[1]] .* βas[i] )
            p_hats[i,t] = pa[as[t]]
        end
    end
    mp_hatsNa = mean(p_hats,dims=1)[:][cons_trials]
    prediction_indexNa = mean((mp_hatsNa .> 0.5) .+ (mp_hatsNa .== 0.5) ./ 2)
    rel_prediction_indexNa = prediction_indexNa / consistency_index
    rel_chance_index = 0.5/consistency_index

    # ----------------------------------------------------------------------
    # Klyubin emp
    # ----------------------------------------------------------------------
    # evaluating empowerment and probabilities
    eK_vals = zeros(N_rooms)
    pK_hats = zeros(length(as))
    eK_model = empKly(max_iter=1000)
    for i_room = 1:N_rooms
        p, StateS, StateSDict, N_s = 
            gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                Xmax, Ymax)
        eK_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eK_model)
    end
    for t = eachindex(Xinds)
        x = Xinds[t]
        pa = softmax([eK_vals[x[1]], eK_vals[x[2]]])
        pK_hats[t] = pa[as[t]]
    end
    pK_hats = pK_hats[cons_trials]
    prediction_indexKly = mean((pK_hats .> 0.5) .+ 
                                (pK_hats .== 0.5) ./ 2)
    rel_prediction_indexKly = prediction_indexKly / consistency_index

    # ----------------------------------------------------------------------
    # Emp-1
    # ----------------------------------------------------------------------
    # evaluating empowerment and probabilities
    eE1_vals = zeros(N_rooms)
    pE1_hats = zeros(length(as))
    eE1_model = emplK(1.,1.,1)
    for i_room = 1:N_rooms
        p, StateS, StateSDict, N_s = 
            gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                Xmax, Ymax)
        eE1_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eE1_model)
    end
    for t = eachindex(Xinds)
        x = Xinds[t]
        pa = softmax([eE1_vals[x[1]], eE1_vals[x[2]]])
        pE1_hats[t] = pa[as[t]]
    end
    pE1_hats = pE1_hats[cons_trials]
    prediction_indexE1 = mean((pE1_hats .> 0.5) .+ 
                                (pE1_hats .== 0.5) ./ 2)
    rel_prediction_indexE1 = prediction_indexE1 / consistency_index

    # ----------------------------------------------------------------------
    # model selection
    # ----------------------------------------------------------------------
    pmAll = BMSdata1["BMSAll"].exp_M[subjectIDs1 .== i_sub,:][:]
    mAll_hat = findmax(pmAll)[2]

    df_temp_all = BMSdataLHat1["chnemp_dfsPlusOthers"][subjectIDs1 .== i_sub][1]
    l_hat_set = [];  logl_hat_set = []; 
    dl_hat_set = []; dlogl_hat_set = []; 
    prediction_index_set = []; mp_hats = [];
    for i_set = 1:2
        # choosing trained parameters
        df_temp = df_temp_all.chnemp_df[i_set]
        βs = df_temp.β; ls = df_temp.l
        
        # choosing test data
        Xinds_temp = df_temp_all.Xinds[1 + mod(i_set,2)]
        as_temp = df_temp_all.as[1 + mod(i_set,2)]

        # evaluating empowerment and probabilities
        emp_vals_hat = zeros(length(ls),N_rooms)
        p_hats = zeros(length(ls),length(as_temp))
        for i = eachindex(ls)
            emp_model_hat = emplK(ls[i],1.,1)
            for i_room = 1:N_rooms
                p, StateS, StateSDict, N_s = 
                    gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                        Xmax, Ymax)
                emp_vals_hat[i,i_room] = p2empCat(p, StateSDict[(0,0)],
                                                        emp_model_hat)
            end
            for t = eachindex(Xinds_temp)
                x = Xinds_temp[t]
                pa = softmax([emp_vals_hat[i,x[1]], 
                                emp_vals_hat[i,x[2]]] .* βs[i])
                p_hats[i,t] = pa[as_temp[t]]
            end
        end
        mp_hats_temp = mean(p_hats,dims=1)[:]
        prediction_index_temp = mean((mp_hats_temp .> 0.5) .+ 
                                        (mp_hats_temp .== 0.5) ./ 2)
        # pushing data for each fold
        push!(l_hat_set,mean(ls)); push!(logl_hat_set,mean(log.(ls)))
        push!(dl_hat_set,std(ls)); push!(dlogl_hat_set,std(log.(ls)))
        push!(prediction_index_set, prediction_index_temp)
        push!(mp_hats, mp_hats_temp)
    end
    prediction_index = mean(prediction_index_set)
    rel_prediction_index = prediction_index / consistency_index
    
    
    # ----------------------------------------------------------------------
    # pushing
    # ----------------------------------------------------------------------
    push!(Data1,
        (; pmAll = pmAll, mAll_hat = mAll_hat,
             l_hat_set =  l_hat_set,  logl_hat_set= logl_hat_set,
            dl_hat_set = dl_hat_set, dlogl_hat_set=dlogl_hat_set,
            consistency_index = consistency_index,
            rel_prediction_index    = rel_prediction_index,
            rel_prediction_indexKly = rel_prediction_indexKly,
            rel_prediction_indexE1  = rel_prediction_indexE1,
            rel_prediction_indexNa  = rel_prediction_indexNa,
            rel_chance_index        = rel_chance_index,
            prediction_index    = prediction_index,
            prediction_indexKly = prediction_indexKly,
            prediction_indexE1  = prediction_indexE1,
            prediction_indexNa  = prediction_indexNa))
end

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Experiment2
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
Data2 = []; GorB = [1,0]
for i_sub = subjectIDs2
    @show i_sub
    # ----------------------------------------------------------------------
    # Inference data
    # ----------------------------------------------------------------------
    temp = load(Path_Load_Inf2 * "inference_data_sub" * string(i_sub) * ".jld2")
    data_temp = []
    for i_GorB = 1:2
        chnemp_df = [temp["chnemp_dfG"],temp["chnemp_dfB"]][i_GorB]
        # ----------------------------------------------------------------------
        # selecting data
        # ----------------------------------------------------------------------
        df = dataDF2[dataDF2.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        df = df[df.Gtrials .== GorB[i_GorB],:]
        
        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1

        # ----------------------------------------------------------------------
        # consistency index
        # ----------------------------------------------------------------------
        consistency_index, cons_trials = choice_consistency(Xinds, as_inds;
                                                    ifpassinds = true)
        # ----------------------------------------------------------------------
        # Baseline consistency
        # ----------------------------------------------------------------------
        βas = [1.]
        p_hats = zeros(length(βas),length(as_inds))
        for i = eachindex(βas)
            for t = eachindex(Xinds)
                x = Xinds[t]
                pa = softmax([size(Prooms[x[1]])[1], 
                              size(Prooms[x[2]])[1]] .* βas[i] )
                p_hats[i,t] = pa[as[t]]
            end
        end
        mp_hatsNa = mean(p_hats,dims=1)[:][cons_trials]
        prediction_indexNa = mean(mp_hatsNa)
        rel_prediction_indexNa = prediction_indexNa / consistency_index
        rel_chance_index = 0.5/consistency_index

        # ----------------------------------------------------------------------
        # Klyubin emp
        # ----------------------------------------------------------------------
        # evaluating empowerment and probabilities
        eK_vals = zeros(N_rooms)
        pK_hats = zeros(length(as))
        eK_model = empKly(max_iter=1000)
        for i_room = 1:N_rooms
            p, StateS, StateSDict, N_s = 
                gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                    Xmax, Ymax)
            eK_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eK_model)
        end
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([eK_vals[x[1]], eK_vals[x[2]]])
            pK_hats[t] = pa[as[t]]
        end
        pK_hats = pK_hats[cons_trials]
        prediction_indexKly = mean((pK_hats .> 0.5) .+ 
                                    (pK_hats .== 0.5) ./ 2)
        rel_prediction_indexKly = prediction_indexKly / consistency_index

        # ----------------------------------------------------------------------
        # Emp-1
        # ----------------------------------------------------------------------
        # evaluating empowerment and probabilities
        eE1_vals = zeros(N_rooms)
        pE1_hats = zeros(length(as))
        eE1_model = emplK(1.,1.,1)
        for i_room = 1:N_rooms
            p, StateS, StateSDict, N_s = 
                gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                    Xmax, Ymax)
            eE1_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eE1_model)
        end
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([eE1_vals[x[1]], eE1_vals[x[2]]])
            pE1_hats[t] = pa[as[t]]
        end
        pE1_hats = pE1_hats[cons_trials]
        prediction_indexE1 = mean((pE1_hats .> 0.5) .+ 
                                    (pE1_hats .== 0.5) ./ 2)
        rel_prediction_indexE1 = prediction_indexE1 / consistency_index

        # ----------------------------------------------------------------------
        # model selection
        # ----------------------------------------------------------------------
        BMSAll_temp = [BMSdata2["BMSAllG"],BMSdata2["BMSAllB"]][i_GorB]
        BMS_temp    = [BMSdata2["BMSG"],   BMSdata2["BMSB"]   ][i_GorB]
        subjectIDs_L_temp = [BMSdata2["subjectIDsLG"],
                             BMSdata2["subjectIDsLB"]][i_GorB]

        pmAll = BMSAll_temp.exp_M[subjectIDs2 .== i_sub,:][:]
        mAll_hat = findmax(pmAll)[2]
        
        df_temp_all = BMSdataLHat2["chnemp_dfsAllPlusOthers"][i_GorB][
                                            subjectIDs2 .== i_sub][1]
        l_hat_set = [];   logl_hat_set = []; 
        dl_hat_set = []; dlogl_hat_set = []; 
        prediction_index_set = []; mp_hats = [];
        for i_set = 1:2
            # choosing trained parameters
            df_temp = df_temp_all.chnemp_df[i_set]
            βs = df_temp.β; ls = df_temp.l
            
            # choosing test data
            Xinds_temp = df_temp_all.Xinds[1 + mod(i_set,2)]
            Xs_temp = df_temp_all.Xs[1 + mod(i_set,2)]
            as_temp = df_temp_all.as[1 + mod(i_set,2)]

            # evaluating empowerment and probabilities
            emp_vals_hat = zeros(length(ls),N_rooms)
            p_hats = zeros(length(ls),length(as_temp))
            for i = eachindex(ls)
                emp_model_hat = emplK(ls[i],1.,1)
                for i_room = 1:N_rooms
                    p, StateS, StateSDict, N_s = 
                        gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                            Xmax, Ymax)
                    emp_vals_hat[i,i_room] = p2empCat(p, StateSDict[(0,0)],
                                                            emp_model_hat)
                end
                for t = eachindex(Xinds_temp)
                    x = Xinds_temp[t]
                    pa = softmax([emp_vals_hat[i,x[1]], 
                                    emp_vals_hat[i,x[2]]] .* βs[i])
                    p_hats[i,t] = pa[as_temp[t]]
                end
            end
            mp_hats_temp = mean(p_hats,dims=1)[:]
            prediction_index_temp = mean((mp_hats_temp .> 0.5) .+ 
                                            (mp_hats_temp .== 0.5) ./ 2)
            # pushing data
            push!(l_hat_set,mean(ls)); push!(logl_hat_set,mean(log.(ls)))
            push!(dl_hat_set,std(ls)); push!(dlogl_hat_set,std(log.(ls)))
            push!(prediction_index_set, prediction_index_temp)
            push!(mp_hats, mp_hats_temp)
        end
        prediction_index = mean(prediction_index_set)
        rel_prediction_index = prediction_index / consistency_index
    
        push!(data_temp,
                (; pmAll = pmAll, mAll_hat = mAll_hat, 
                l_hat_set  =  l_hat_set,  logl_hat_set= logl_hat_set,
                dl_hat_set = dl_hat_set, dlogl_hat_set=dlogl_hat_set,
                consistency_index = consistency_index,
                rel_prediction_index    = rel_prediction_index,
                rel_prediction_indexNa  = rel_prediction_indexNa,
                rel_prediction_indexE1  = rel_prediction_indexE1,
                rel_prediction_indexKly = rel_prediction_indexKly,
                rel_chance_index        = rel_chance_index,
                prediction_index    = prediction_index,
                prediction_indexKly = prediction_indexKly,
                prediction_indexE1  = prediction_indexE1,
                prediction_indexNa  = prediction_indexNa,
                Cond = df.GB_condition[1]))
    end
    push!(Data2, data_temp)
end

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Experiment 3
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
Data3 = []
for i_sub = subjectIDs3
    @show i_sub
    # ----------------------------------------------------------------------
    # Inference data
    # ----------------------------------------------------------------------
    temp = load(Path_Load_Inf3 * "inference_data_sub" * string(i_sub) * ".jld2")
    chnemp_df = temp["chnemp_df"]
    
    # ----------------------------------------------------------------------
    # data management
    # ----------------------------------------------------------------------
    df = dataDF3[dataDF3.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    
    as_inds = df.chosenroom .+ 1
    as = df.action .+ 1

    # ----------------------------------------------------------------------
    # consistency index
    # ----------------------------------------------------------------------
    consistency_index, cons_trials = choice_consistency(Xinds, as_inds;
                                                    ifpassinds = true)
    # ----------------------------------------------------------------------
    # baseline consistency
    # ----------------------------------------------------------------------
    βas = [1.]
    p_hats = zeros(length(βas),length(as_inds))
    for i = eachindex(βas)
        for t = eachindex(Xinds)
            x = Xinds[t]
            pa = softmax([size(Prooms[x[1]])[1], 
                          size(Prooms[x[2]])[1]] .* βas[i] )
            p_hats[i,t] = pa[as[t]]
        end
    end
    mp_hatsNa = mean(p_hats,dims=1)[:][cons_trials]
    prediction_indexNa = mean((mp_hatsNa .> 0.5) .+ (mp_hatsNa .== 0.5) ./ 2)
    rel_prediction_indexNa = prediction_indexNa / consistency_index
    rel_chance_index = 0.5/consistency_index

    # ----------------------------------------------------------------------
    # Klyubin emp
    # ----------------------------------------------------------------------
    # evaluating empowerment and probabilities
    eK_vals = zeros(N_rooms)
    pK_hats = zeros(length(as))
    eK_model = empKly(max_iter=1000)
    for i_room = 1:N_rooms
        p, StateS, StateSDict, N_s = 
            gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                Xmax, Ymax)
        eK_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eK_model)
    end
    for t = eachindex(Xinds)
        x = Xinds[t]
        pa = softmax([eK_vals[x[1]], eK_vals[x[2]]])
        pK_hats[t] = pa[as[t]]
    end
    pK_hats = pK_hats[cons_trials]
    prediction_indexKly = mean((pK_hats .> 0.5) .+ 
                                (pK_hats .== 0.5) ./ 2)
    rel_prediction_indexKly = prediction_indexKly / consistency_index

    # ----------------------------------------------------------------------
    # Emp-1
    # ----------------------------------------------------------------------
    # evaluating empowerment and probabilities
    eE1_vals = zeros(N_rooms)
    pE1_hats = zeros(length(as))
    eE1_model = emplK(1.,1.,1)
    for i_room = 1:N_rooms
        p, StateS, StateSDict, N_s = 
            gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                Xmax, Ymax)
        eE1_vals[i_room] = p2empCat(p, StateSDict[(0,0)], eE1_model)
    end
    for t = eachindex(Xinds)
        x = Xinds[t]
        pa = softmax([eE1_vals[x[1]], eE1_vals[x[2]]])
        pE1_hats[t] = pa[as[t]]
    end
    pE1_hats = pE1_hats[cons_trials]
    prediction_indexE1 = mean((pE1_hats .> 0.5) .+ 
                                (pE1_hats .== 0.5) ./ 2)
    rel_prediction_indexE1 = prediction_indexE1 / consistency_index

    # ----------------------------------------------------------------------
    # model selection
    # ----------------------------------------------------------------------
    pmAll = BMSdata3["BMSAll"].exp_M[subjectIDs3 .== i_sub,:][:]
    mAll_hat = findmax(pmAll)[2]

    df_temp_all = BMSdataLHat3["chnemp_dfsPlusOthers"][subjectIDs3 .== i_sub][1]
    l_hat_set = [];  logl_hat_set = []; 
    dl_hat_set = []; dlogl_hat_set = []; 
    prediction_index_set = []; mp_hats = [];
    for i_set = 1:2
        # choosing trained parameters
        df_temp = df_temp_all.chnemp_df[i_set]
        βs = df_temp.β; ls = df_temp.l
        
        # choosing test data
        Xinds_temp = df_temp_all.Xinds[1 + mod(i_set,2)]
        as_temp = df_temp_all.as[1 + mod(i_set,2)]

        # evaluating empowerment and probabilities
        emp_vals_hat = zeros(length(ls),N_rooms)
        p_hats = zeros(length(ls),length(as_temp))
        for i = eachindex(ls)
            emp_model_hat = emplK(ls[i],1.,1)
            for i_room = 1:N_rooms
                p, StateS, StateSDict, N_s = 
                    gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                        Xmax, Ymax)
                emp_vals_hat[i,i_room] = p2empCat(p, StateSDict[(0,0)],
                                                        emp_model_hat)
            end
            for t = eachindex(Xinds_temp)
                x = Xinds_temp[t]
                pa = softmax([emp_vals_hat[i,x[1]], 
                                emp_vals_hat[i,x[2]]] .* βs[i])
                p_hats[i,t] = pa[as_temp[t]]
            end
        end
        mp_hats_temp = mean(p_hats,dims=1)[:]
        prediction_index_temp = mean((mp_hats_temp .> 0.5) .+ 
                                        (mp_hats_temp .== 0.5) ./ 2)
        # pushing data for each fold
        push!(l_hat_set,mean(ls)); push!(logl_hat_set,mean(log.(ls)))
        push!(dl_hat_set,std(ls)); push!(dlogl_hat_set,std(log.(ls)))
        push!(prediction_index_set, prediction_index_temp)
        push!(mp_hats, mp_hats_temp)
    end
    prediction_index = mean(prediction_index_set)
    rel_prediction_index = prediction_index / consistency_index
    
    
    # ----------------------------------------------------------------------
    # pushing
    # ----------------------------------------------------------------------
    push!(Data3,
        (; pmAll = pmAll, mAll_hat = mAll_hat,
             l_hat_set =  l_hat_set,  logl_hat_set= logl_hat_set,
            dl_hat_set = dl_hat_set, dlogl_hat_set=dlogl_hat_set,
            consistency_index = consistency_index,
            rel_prediction_index    = rel_prediction_index,
            rel_prediction_indexKly = rel_prediction_indexKly,
            rel_prediction_indexE1  = rel_prediction_indexE1,
            rel_prediction_indexNa  = rel_prediction_indexNa,
            rel_chance_index        = rel_chance_index,
            prediction_index    = prediction_index,
            prediction_indexKly = prediction_indexKly,
            prediction_indexE1  = prediction_indexE1,
            prediction_indexNa  = prediction_indexNa))
end


# ----------------------------------------------------------------------
# plotting acc rate
# ----------------------------------------------------------------------
M1 = [d.mAll_hat for d = Data1]
M2G = [d[1].mAll_hat for d = Data2]
M2B = [d[2].mAll_hat for d = Data2]
M3 = [d.mAll_hat for d = Data3]

# model names
Legends = [ "Na-E1",  "EKly-E1",  "Emp1-E1",   "Emp-E1", 
            "Na-2G",  "EKly-2G",  "Emp1-2G",   "Emp-2G", 
            "Na-2B",  "EKly-2B",  "Emp1-2B",   "Emp-2B",
            "Na-E3",  "EKly-E3",  "Emp1-E3",   "Emp-E3"]

# participant types
f_emponly(x) = (x == 3)
f_others(x)  = (x != 3)
f_all(x)     = true
fset = [f_emponly, f_others, f_all]
fnames = ["EmpParts","OtherParts","AllParts"]

for i_f = eachindex(fset)
    nametag = fnames[i_f]
    f_part = fset[i_f]
    Ys1 = [ [   [d.rel_prediction_indexNa for d = Data1][f_part.(M1)],
                [d[1].rel_prediction_indexNa for d = Data2][f_part.(M2G)],
                [d[2].rel_prediction_indexNa for d = Data2][f_part.(M2B)],
                [d.rel_prediction_indexNa for d = Data3][f_part.(M3)]],

            [   [d.rel_prediction_indexKly for d = Data1][f_part.(M1)],
                [d[1].rel_prediction_indexKly for d = Data2][f_part.(M2G)],
                [d[2].rel_prediction_indexKly for d = Data2][f_part.(M2B)],
                [d.rel_prediction_indexKly for d = Data3][f_part.(M3)]],

            [   [d.rel_prediction_indexE1 for d = Data1][f_part.(M1)],
                [d[1].rel_prediction_indexE1 for d = Data2][f_part.(M2G)],
                [d[2].rel_prediction_indexE1 for d = Data2][f_part.(M2B)],
                [d.rel_prediction_indexE1 for d = Data3][f_part.(M3)]],

            [   [d.rel_prediction_index for d = Data1][f_part.(M1)],
                [d[1].rel_prediction_index for d = Data2][f_part.(M2G)],
                [d[2].rel_prediction_index for d = Data2][f_part.(M2B)],
                [d.rel_prediction_index for d = Data3][f_part.(M3)]],
        ]
    Ys2 = [ [   [d.prediction_indexNa for d = Data1][f_part.(M1)],
                [d[1].prediction_indexNa for d = Data2][f_part.(M2G)],
                [d[2].prediction_indexNa for d = Data2][f_part.(M2B)],
                [d.prediction_indexNa for d = Data3][f_part.(M3)]],

            [   [d.prediction_indexKly for d = Data1][f_part.(M1)],
                [d[1].prediction_indexKly for d = Data2][f_part.(M2G)],
                [d[2].prediction_indexKly for d = Data2][f_part.(M2B)],
                [d.prediction_indexKly for d = Data3][f_part.(M3)]],

            [   [d.prediction_indexE1 for d = Data1][f_part.(M1)],
                [d[1].prediction_indexE1 for d = Data2][f_part.(M2G)],
                [d[2].prediction_indexE1 for d = Data2][f_part.(M2B)],
                [d.prediction_indexE1 for d = Data3][f_part.(M3)]],

            [   [d.prediction_index for d = Data1][f_part.(M1)],
                [d[1].prediction_index for d = Data2][f_part.(M2G)],
                [d[2].prediction_index for d = Data2][f_part.(M2B)],
                [d.prediction_index for d = Data3][f_part.(M3)]],
        ]
    for i_Y = 1:2
        Ys = [Ys1, Ys2][i_Y]
        Ylabel = ["normalized accuracy", "raw accuracy"][i_Y]
        nametag = ["Norm", "Raw"][i_Y] * "Acc_" * nametag
        println("------------------------------------------------")
        println("------------------------------------------------")
        println(nametag)
        println("------------------------------------------------")
        println("------------------------------------------------")
        XLegends = []
        figure(figsize=(14,6))
        Colors = [MainColors.GB[1], MainColors.GB[1], MainColors.GB[2], "#435663"]
        ax = subplot(1,1,1)
        for i = eachindex(Ys[1])
            δ = (i-1) * (1 + length(Ys))
            x = δ .+ (1:length(Ys))
            push!(XLegends, Array(x))
            Ys_plot = [y[i] for y = Ys]
            mYs = [mean(y[i]) for y = Ys]
            dYs = [std(y[i])/sqrt(length(y[i])) for y = Ys]

            ax.bar(x, mYs,  color = Colors[i], alpha = 1.0)
            ax.errorbar(x, mYs, yerr=dYs,color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)

            for j = eachindex(Ys_plot)
                Y_test = Ys_plot[end] - Ys_plot[j]
                Test_result = OneSampleTTest(Y_test)
                @show Test_result
                pval = pvalue(Test_result)
                logBF = BIC_OneSampleTTest(Y_test)
                ax.text(x[j], 0.85 + 0.05 * mod(j,3),
                        string("p:", Func_pval_string(pval),
                            ", lBF:", Func_logBF_string(logBF)),
                        fontsize=8, horizontalalignment="center", rotation=0)
            end
            for j = eachindex(Ys_plot[1])
                x_temp = Array(x) .+ (0.2 * (rand() - 0.5))
                y_temp = [y[j] for y = Ys_plot]
                ax.plot(x_temp, y_temp, ".k",alpha = 0.2)
                ax.plot(x_temp, y_temp, "k",alpha = 0.05)
            end
        end
        XLegends = vcat(XLegends...)
        ax.set_xlim([XLegends[1] - 1,XLegends[end]+1]); 
        ax.set_xticks(XLegends); ax.set_xticklabels(Legends)
        ax.set_ylim([0.5,1.0]); 
        ax.set_ylabel(Ylabel)

        tight_layout()
        savefig(Path_Save * nametag * ".pdf")
        savefig(Path_Save * nametag * ".png")
        savefig(Path_Save * nametag * ".svg")
    end
end
            


# ----------------------------------------------------------------------
# L-consistency
# ----------------------------------------------------------------------
for i_f = eachindex(fset)
    nametag = fnames[i_f]
    f_part = fset[i_f]

    Y_set = [[  [d.l_hat_set for d = Data1][f_part.(M1)],
                [d[1].l_hat_set for d = Data2][f_part.(M2G)],
                [d[2].l_hat_set for d = Data2][f_part.(M2B)],
                [d.l_hat_set for d = Data3][f_part.(M3)],
                [d.logl_hat_set for d = Data1][f_part.(M1)],
                [d[1].logl_hat_set for d = Data2][f_part.(M2G)],
                [d[2].logl_hat_set for d = Data2][f_part.(M2B)],
                [d.logl_hat_set for d = Data3][f_part.(M3)]],
            [   [d.dl_hat_set for d = Data1][f_part.(M1)],
                [d[1].dl_hat_set for d = Data2][f_part.(M2G)],
                [d[2].dl_hat_set for d = Data2][f_part.(M2B)],
                [d.dl_hat_set for d = Data3][f_part.(M3)],
                [d.dlogl_hat_set for d = Data1][f_part.(M1)],
                [d[1].dlogl_hat_set for d = Data2][f_part.(M2G)],
                [d[2].dlogl_hat_set for d = Data2][f_part.(M2B)],
                [d.dlogl_hat_set for d = Data3][f_part.(M3)]]]

    Yref = [1,1,1,1,0,0,0,0]
    labels = ["l-hatE1","l-hatE2G","l-hatE2B","l-hatE3",
              "log-l-hatE1","log-l-hatE2G","log-l-hatE2B","log-l-hatE3"]

    figure(figsize=(14,9))
    for j = 1:8
        ax = subplot(2,4,j)
        Ys  = [hcat(Y_set[1][j]...)[1,:], hcat(Y_set[1][j]...)[2,:]]
        dYs = [hcat(Y_set[2][j]...)[1,:], hcat(Y_set[2][j]...)[2,:]]
        ρ = round(cor(Ys...),digits = 2)
        ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
        x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
        x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
        ax.plot([x_min,x_max],[x_min,x_max],"--k")
        ax.plot([x_min,x_max],[1,1] .* Yref[j],"--k")
        ax.plot([1,1] .* Yref[j],[x_min,x_max],"--k")
        ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
        Test_result = CorrelationTest(Ys...)
        @show Test_result
        pval = pvalue(Test_result)
        logBF = BIC_CorrelationTest(Ys...)
        ax.set_ylabel(labels[j] * " (set 2)")
        ax.set_xlabel(labels[j] * " (set 1)")
        ax.legend(["ρ = " * string(ρ)])
        ax.set_title("pval = " * Func_pval_string(pval) *
                    ", lBF = " * Func_logBF_string(logBF))
    end
    tight_layout()
    savefig(Path_Save * "Lconsistency_" * nametag * ".pdf")
    savefig(Path_Save * "Lconsistency_" * nametag * ".png")
    savefig(Path_Save * "Lconsistency_" * nametag * ".svg")
end
