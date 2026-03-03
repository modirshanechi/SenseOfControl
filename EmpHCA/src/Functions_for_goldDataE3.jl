# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exclusion criteria
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

