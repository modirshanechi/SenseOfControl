# Data from Experiment 2

Description of different files:
1.  `SelectionData.csv` Contains the room selection data with the following entries:
    - `trial`: trial number
    - `GB_condition`: whether the participants belonged to the gold-first (`TRUE`) or bomb-first (`FALSE`) condition.
    - `subject`: subject number (1, 2, 3, ...)
    - `ID`: randomized subject ID
    - `timeout`: whether the trial was aborted because of a reaction time > 10 seconds
    - `rt`: reaction time; equals to -1 if the trial was aborted
    - `room1`: the index of the room shown on the left side of the screen*
    - `room2`: the index of the room shown on the right side of the screen*
    - `action`: whether the left (= 0) or right (= 1) room was selected; equals to -1 if the trial was aborted
    - `chosenroom`: the index of the chosen room*
    - `gold`: the 'number' of collected gold coins until that trial; this was rounded every 7 trials and shown to the participants as feedback.
    - `bomb`: the 'number' of exploded bombs until that trial; this was rounded every 7 trials and shown to the participants as feedback.
    - `Gtrials`: whether the trial was a gold trial (= 1)

2. `SelectionDataG.csv` is the subset of `SelectionData.csv` with `Gtrials=1`

3. `SelectionDataB.csv` is the subset of `SelectionData.csv` with `Gtrials=0`

4. `ExclusionInfo.csv` 
Contains several statistics that could show un-attentiveness.
    - Only `room1preference_gold`, `room1preference_bomb`, `selection_timeout_gold`, and `selection_timeout_bomb` were used to exclude participants from the main analysis. 
    - Additionally, `skip_survey`, `attention_check_fail`, `straightlining`, and `zigzagging` were used to exclude participants from the analysis of the survey data; 
    - See the manuscript and `EmpHCA/src/Functions_for_goldDataE2.jl`

5. `SurveyMetaData.csv`
Contains the survey metadata with the following entries:
    - `QNames`: short abbrevations to refer to different questions
    - `Qs`: the exact survey question
    - `Rs`: possible responses (coded as 0, 1, 2, ...)
    - `Attention`: whether the question is an attention check
    - `AttentionResp`: the correct response for attention checks; equals to -1 otherwise
    - `IPC`: the survey category for the LOC survey

6. `SurveyData.csv`
contains the survey data. There is a column associated with each `QNames` entry in `SurveyMetaData.csv`. The values correspond to participants' responses.


*: see the constant `PaperRoomOrder` in `EmpHCA/src/Functions_for_gold.jl` for the room numbering.