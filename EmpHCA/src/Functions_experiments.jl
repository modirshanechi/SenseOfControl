################################################################################
# Code for general functions of data analysis
################################################################################

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for specifying the experiment
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
@concrete struct ExperimentSpecification
    exp
    data_path
    fig_path
    has_gb_split
    exclusion_fn
end
function ExperimentSpecification(n::Int)
    exp = n
    data_path = "data/Experiment" * string(n) * "/clean/"
    fig_path = "results/Experiment" * string(n) * "/"
    if n == 1
        ExperimentSpecification(
            exp, data_path, fig_path, false, ExcIndicatorExp1!)
    elseif n == 2
        ExperimentSpecification(
            exp, data_path, fig_path, true, ExcIndicatorExp2!)
    elseif n == 3
        ExperimentSpecification(
            exp, data_path, fig_path, false, ExcIndicatorExp3!)
    end
end
export ExperimentSpecification

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Loading clean data
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function load_clean_data(spec::ExperimentSpecification; ifoutliers = 0)
    ExcDF = DataFrame(CSV.File(spec.data_path * "ExclusionInfo.csv"))
    selDF = DataFrame(CSV.File(spec.data_path * "SelectionData.csv"))
    surveyDF = DataFrame(CSV.File(spec.data_path * "SurveyData.csv"))
    metaDF_survey = DataFrame(CSV.File(spec.data_path * "SurveyMetaData.csv"))
    subjectIDs = ExcDF.subject[ExcDF.task_outliers .== ifoutliers]
    return (; ExcDF=ExcDF, selDF=selDF, 
                surveyDF=surveyDF, metaDF_survey=metaDF_survey,
                subjectIDs=subjectIDs)
end 
export load_clean_data

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Outlier statistics
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function print_outlierstats(spec::ExperimentSpecification, data)
    ExcDF = data.ExcDF; surveyDF = data.surveyDF

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

    if !spec.has_gb_split
        println("Time-outs")
        println(countmap(ExcDF.selection_timeout))
    else
        println("Time-outs")
        println(sum((ExcDF.selection_timeout_gold .>= 5) .|
                (ExcDF.selection_timeout_bomb .>= 5) ))
        println("Time-outs bomb")
        println(countmap(ExcDF.selection_timeout_bomb))
        println("Time-outs gold")
        println(countmap(ExcDF.selection_timeout_gold))

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
    end
end
export print_outlierstats

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Chance level statistics for outliers
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function room_null_stats(spec::ExperimentSpecification; 
                    Ymax = 1, Xmax = 1, FBPeriod = 7, 
                    Nsamp = 100000, p_up = 0.975,
                    rng = MersenneTwister(2024))

    Prooms, ΔState, ΔStateDict = gold_proom_sets();
    N_rooms = length(Prooms); 
    if spec.exp != 2
        valid_rooms = 1:N_rooms
    else
        valid_rooms = valid_rooms_E23
        N_rooms = length(valid_rooms)
        Prooms = Prooms[valid_rooms]
    end
    # all combination of rooms
    Xinds = vcat([[[i,j] for j = (1:N_rooms)[(1:N_rooms) .!= i]] 
                                                for i = 1:N_rooms]...)
    # evaluation the gold probabilities
    emp1 = zeros(N_rooms); emp1_model = emplK(1.,1.,1)
    for i_room = 1:N_rooms
        p, StateS, StateSDict, N_s = 
            gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                Xmax, Ymax)
        emp1[i_room] = p2empCat(p, StateSDict[(0,0)], emp1_model)
    end
    Pgold = emp1 ./ (length(ΔState) - 1);

    
    if !spec.has_gb_split
        ChanceGold = zeros(Nsamp); 
        ChanceRoom1 = zeros(Nsamp)
        
        for i = 1:Nsamp
            if mod(i,10000) == 0
                @show i
            end
            Xinds_temp = Xinds[randperm(rng, length(Xinds))]
            gold_temp = 0
            choice_temp = Int.(zeros(length(Xinds))); 
            Xch_temp = Int.(zeros(length(Xinds)))
            for j = eachindex(Xinds_temp)
                choice_temp[j] = rand(rng,[1,2])
                Xch_temp[j] = Xinds_temp[j][choice_temp[j]]
                gold_temp += Pgold[Xch_temp[j]]
                if mod(j, FBPeriod) == 0 
                    gold_temp = ceil(gold_temp)
                end
            end
            ChanceRoom1[i] = mean(Xch_temp[[1 ∈ x for x = Xinds_temp]] .== 1)
            ChanceGold[i] = gold_temp
        end
        mChance = mean(ChanceGold)
        mChance_plus = quantile(ChanceGold,p_up)

        mR1Chance = mean(ChanceRoom1)
        mR1Chance_plus = quantile(ChanceRoom1,p_up)
        return mChance, mChance_plus, mR1Chance, mR1Chance_plus
    else
        # chance level
        ChanceGold  = zeros(Nsamp); 
        ChanceBomb  = zeros(Nsamp); 
        ChanceNet  = zeros(Nsamp); 
        ChanceRoom1 = zeros(Nsamp)

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
        mGChance_plus =  quantile(ChanceGold,p_up)

        mBChance = mean(ChanceBomb)
        mBChance_plus = quantile(ChanceBomb,1-p_up)

        mBGChance = mean(ChanceNet)
        mBGChance_plus =  quantile(ChanceNet,p_up)

        mR1Chance = mean(ChanceRoom1)
        mR1Chance_plus =  quantile(ChanceRoom1,p_up)
        return mGChance, mGChance_plus, mBChance, mBChance_plus,
                mBGChance, mBGChance_plus, mR1Chance, mR1Chance_plus
        end
end
export room_null_stats


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria Exp 1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function ExcIndicatorExp1!(df;
                        room1preference = 0.7,
                        selection_timeout = 10,
                        attention_check_fail = 2,
                        skip_survey = 1,
                        straightlining = 0.8,
                        zigzagging = 0.8)
    goodsubs_task = (df.room1preference .>= room1preference)
    goodsubs_task .&= (df.selection_timeout .< selection_timeout)
    df.task_outliers = (1 .- goodsubs_task) .== 1
    
    goodsubs_survey = (df.attention_check_fail .< attention_check_fail)
    goodsubs_survey .&= (df.skip_survey .< skip_survey)
    goodsubs_survey .&= (df.straightlining .< straightlining)
    goodsubs_survey .&= (df.zigzagging .< zigzagging)
    df.survey_outliers = (1 .- goodsubs_survey) .== 1

    df.outliers = (df.task_outliers .+ df.survey_outliers) .> 0
end
export ExcIndicatorExp1!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria Exp 2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function ExcIndicatorExp2!(df;
                        room1preference = 0.75,
                        selection_timeout_gold = 5,
                        selection_timeout_bomb = 5,
                        attention_check_fail = 2,
                        skip_survey = 1,
                        straightlining = 0.8,
                        zigzagging = 0.8)
    goodsubs_task   = (df.room1preference_gold .>= room1preference) .&
                      (df.room1preference_bomb .>= room1preference)
    goodsubs_task .&= (df.selection_timeout_gold .< selection_timeout_gold)
    goodsubs_task .&= (df.selection_timeout_bomb .< selection_timeout_bomb)
    df.task_outliers = (1 .- goodsubs_task) .== 1
    
    goodsubs_survey = (df.attention_check_fail .< attention_check_fail)
    goodsubs_survey .&= (df.skip_survey .< skip_survey)
    goodsubs_survey .&= (df.straightlining .< straightlining)
    goodsubs_survey .&= (df.zigzagging .< zigzagging)
    df.survey_outliers = (1 .- goodsubs_survey) .== 1

    df.outliers = (df.task_outliers .+ df.survey_outliers) .> 0
end
export ExcIndicatorExp2!

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria Exp 3
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function ExcIndicatorExp3!(df;
                        room1preference = 0.75,
                        selection_timeout = 5,
                        attention_check_fail = 2,
                        skip_survey = 1,
                        straightlining = 0.8,
                        zigzagging = 0.8)
    goodsubs_task   = (df.room1preference .>= room1preference)
    goodsubs_task .&= (df.selection_timeout .< selection_timeout)
    df.task_outliers = (1 .- goodsubs_task) .== 1
    
    goodsubs_survey = (df.attention_check_fail .< attention_check_fail)
    goodsubs_survey .&= (df.skip_survey .< skip_survey)
    goodsubs_survey .&= (df.straightlining .< straightlining)
    goodsubs_survey .&= (df.zigzagging .< zigzagging)
    df.survey_outliers = (1 .- goodsubs_survey) .== 1

    df.outliers = (df.task_outliers .+ df.survey_outliers) .> 0
end
export ExcIndicatorExp3!



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Consistency calculation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Xinds = pairs of room indices
# as_inds = chosen room
function choice_consistency(Xinds, as_inds; ifpassinds = false, ifraw = false)
    inds = [[x[2],x[1]] ∈ Xinds for x = Xinds]
    as_inds = as_inds[inds]; Xinds = Xinds[inds]; 
    # consistency
    consistency_indeces = zero(as_inds)
    for t = eachindex(as_inds)
            x = Xinds[t]
            a1 = as_inds[t]
            a2 = as_inds[[[x[2],x[1]] == x2 for x2 = Xinds]][1]
            consistency_indeces[t] = a1 == a2
    end
    consistency_index = mean(consistency_indeces)
    
    # correction
    if !ifraw
        consistency_index = (consistency_index + 1) / 2
    end
    if ifpassinds
        return consistency_index, inds
    else
        return consistency_index
    end
end
function choice_consistency(Xinds1, as_inds1, 
                            Xinds2, as_inds2; ifpassinds = false)
    # choosing repeated pairs
    inds11 = [[x[2],x[1]] ∈ Xinds1 for x = Xinds1]
    as_inds1 = as_inds1[inds11]; Xinds1 = Xinds1[inds11]; 
    inds22 = [[x[2],x[1]] ∈ Xinds2 for x = Xinds2]
    as_inds2 = as_inds2[inds22]; Xinds2 = Xinds2[inds22]; 

    # choosing common pairs
    inds1 = [[x[1],x[2]] ∈ Xinds2 for x = Xinds1]
    as_inds1 = as_inds1[inds1]; Xinds1 = Xinds1[inds1]; 
    inds2 = [[x[1],x[2]] ∈ Xinds1 for x = Xinds2]
    as_inds2 = as_inds2[inds2]; Xinds2 = Xinds2[inds2]; 
    
    
    if length(as_inds1) != length(as_inds2)
        error("Something wrong")
    end

    # consistency
    consistency_indeces = zero(as_inds1) .* 1.0
    for t = eachindex(as_inds1)
        x = Xinds1[t]
        a1s = [as_inds1[t],
                as_inds1[[[x[2],x[1]] == x2 for x2 = Xinds1]][1]]
        a2s = [as_inds2[[[x[1],x[2]] == x2 for x2 = Xinds2]][1],
                as_inds2[[[x[2],x[1]] == x2 for x2 = Xinds2]][1]]
        consistency_indeces[t] = mean([mean(a1s .== a) for a = a2s])
    end
    consistency_index = mean(consistency_indeces)
    if ifpassinds
        return consistency_index, inds1, inds2
    else
        return consistency_index
    end
end
export choice_consistency
function subchoice_consistency(Xinds1, as_inds1, 
                               Xinds2, as_inds2; ifpassinds = false)
    # choosing pairs of X1 whose inverse are in X2
    inds1 = [[x[2],x[1]] ∈ Xinds2 for x = Xinds1]
    as_inds1 = as_inds1[inds1]; Xinds1 = Xinds1[inds1]; 
    
    # consistency
    consistency_indeces = zero(as_inds1) .* 1.0
    for t = eachindex(as_inds1)
        x = Xinds1[t]
        a1 = as_inds1[t]
        a2 = as_inds2[[[x[2],x[1]] == x2 for x2 = Xinds2]][1]
        consistency_indeces[t] = a1 == a2
    end
    consistency_index = mean(consistency_indeces)
    if ifpassinds
        return consistency_index, inds1
    else
        return consistency_index
    end
end
export subchoice_consistency
