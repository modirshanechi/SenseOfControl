//----------------------------------------------------------------------------//
// General fixation trial
//----------------------------------------------------------------------------//
var fixation = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: '<div style="font-size:60px;">+</div>',
    choices: "NO_KEYS",
    trial_duration: function(){
            return jsPsych.randomization.sampleWithoutReplacement([250, 300, 400], 1)[0];
    },
    data: {
        task: 'fixation',
    },
    save_trial_parameters: {
        trial_duration: true,
    }
};

//----------------------------------------------------------------------------//
// General collected gold feedback trial
//----------------------------------------------------------------------------//
var gold_feedback_trial = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<div class="gold-coins-container">
            <img class="gold-coins-image" src="img/Goal.png" alt="Coin Image">
            <div class="gold-coins-text">
                <h2>You have earned</h2>
                <h1 id="gold_text2" > </h1>
            </div>
            <img class="gold-coins-image" src="img/Goal.png" alt="Coin Image">
        </div>
        <div class="gold-coins-container">
            <img class="gold-coins-image" src="img/Bomb.png" alt="Coin Image">
            <div class="gold-coins-text">
                <h2>You owe</h2>
                <h1 id="gold_text3" > </h1>
            </div>
            <img class="gold-coins-image" src="img/Bomb.png" alt="Coin Image">
        </div>
        <h1 id="gold_text4"></h1>`,
    choices: ['Continue'],
    data: {
        task: 'feedback_trial',
        total_bonus: function(){ return collected_gold}
    },
    on_start: function(){
        collected_gold = Math.ceil(collected_gold);
        delta_collected_gold = collected_gold - last_collected_gold;
        if(delta_collected_gold > 0){
            collected_gold += 1
        }
        last_collected_gold = collected_gold;

        collected_bomb = Math.floor(collected_bomb);
        delta_collected_bomb = collected_bomb - last_collected_bomb;
        last_collected_bomb = collected_bomb;
    },
    on_load: function(){
        const feedback2 = document.getElementById('gold_text2');
        if(collected_gold < 2){
            feedback2.textContent = collected_gold + " Gold Coin so far!";
        }else{
            feedback2.textContent = collected_gold + " Gold Coins so far!";
        }

        const feedback3 = document.getElementById('gold_text3');
        if(collected_bomb < 2){
            feedback3.textContent = collected_bomb + " Gold Coin for healthcare!";
        }else{
            feedback3.textContent = collected_bomb + " Gold Coins for healthcare!";
        }

        const feedback4 = document.getElementById('gold_text4');
        const total_balance = initial_gold + collected_gold - collected_bomb
        if(Math.abs(total_balance) < 2){
            feedback4.textContent = "Your total balance: " + total_balance + " Gold Coin";
        }else{
            feedback4.textContent = "Your total balance: " + total_balance + " Gold Coins";
        }
    }
};

//----------------------------------------------------------------------------//
// break trials
//----------------------------------------------------------------------------//
var break_trial_start = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<div>
            <h2>Great job so far!</h2>
            <h1>Let's take a 10-second break.</h1>
        </div>`,
    choices: "NO_KEYS",
    trial_duration: 1000,
    data: {
        task: 'break',
    }
};
function make_break_trial(i){
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<div> <h1>` + i + `</h1> </div>`,
        choices: "NO_KEYS",
        trial_duration: 1000,
        data: {
            task: 'break',
        }
    };
    return trial
}
var break_timeline = [break_trial_start];
for (let j = 10; j > 0; j--) {
    break_timeline.push(make_break_trial(j));
}
var break_trial = {
    timeline: break_timeline,
    repetitions: 1
};

//----------------------------------------------------------------------------//
// One-step moving in the grid
//----------------------------------------------------------------------------//
function make_movegrid_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agent() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to actions.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegrid',
            agent_TPos: function(){return agent_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            agent_topPos  = topPos0;
            agent_leftPos = leftPos0;
            agent_TPos    = TPos0;
        },
        on_load: function(){
            showactions(room_index, "")
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move(options)
        }
    };
    return trial
}

function make_movegridresult_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agent() + 
                `<div id="action_row" class="row"></div>` + 
                `<div><h3></h3></div>` +
            `</div>`,
        trial_duration: 1000,
        choices: "NO_KEYS",
        data: {
            task: 'movegridresult',
            agent_TPos: function(){return agent_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_load: function(){
            showactions(room_index, "")
            updateagent()
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// One-step  moving in the grid practice
//----------------------------------------------------------------------------//
function make_movegrid_practice_trial(room_index, i_practice) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agent() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to actions.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegrid',
            agent_TPos: function(){return agent_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            agent_topPos  = topPos0;
            agent_leftPos = leftPos0;
            agent_TPos    = TPos0;
        },
        on_load: function(){
            showactions(room_index, "")
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move_practice(options, i_practice)
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// One-step gold collecting in the grid
//----------------------------------------------------------------------------//
function make_seegridgoal_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agent() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3 style="color: red;">Wait until the goal appears!</h3></div>` +
            `</div>`,
        choices: "NO_KEYS",
        trial_duration: 750,
        data: {
            task: 'seegrid',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            agent_topPos  = topPos0;
            agent_leftPos = leftPos0;
            agent_TPos    = TPos0;
            goal_topPos   = topPos0;
            goal_leftPos  = leftPos0;
            goal_TPos     = TPos0;
        },
        on_load: function(){
            showactions(room_index, "")
        }
    };
    return trial
}

function make_movegridgoal_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentgoal() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to the most appropriate action to collect the gold.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegridgoal',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            sample_goal();
        },
        on_load: function(){
            showactions(room_index, "")
            updategoal()
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move(options)
            if(areArraysEqual(agent_TPos, goal_TPos)){
                data.success = true;
            } else {
                data.success = false;
            }
        }
    };
    return trial
}

function make_movegridgoalresult_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentgoal() + 
                `<div id="action_row" class="row"></div>` + 
                `<div><h3 id="feedback"></h3></div>` +
            `</div>`,
        trial_duration: 1000,
        choices: "NO_KEYS",
        data: {
            task: 'movegridgoalresult',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_load: function(){
            showactions(room_index, "");
            updateagent();
            updategoal();
            
            var success = jsPsych.data.get().last(1).values()[0].success;
            const feedback = document.getElementById('feedback');
            if(success){
                feedback.textContent = "Yayy! You got the gold! :)";
                feedback.style.color = "green";
            }else{
                feedback.textContent = "No! You missed it... :(";
                feedback.style.color = "red";
            }
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// One-step bomb avoiding in the grid
//----------------------------------------------------------------------------//
function make_seegridgoalbomb_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentbomb() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3 style="color: red;">Wait until a bomb disappears!</h3></div>` +
            `</div>`,
        choices: "NO_KEYS",
        trial_duration: 750,
        data: {
            task: 'seegridbomb',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            agent_topPos  = topPos0;
            agent_leftPos = leftPos0;
            agent_TPos    = TPos0;
            goal_topPos   = topPos0;
            goal_leftPos  = leftPos0;
            goal_TPos     = TPos0;
        },
        on_load: function(){
            showactions(room_index, "")
        }
    };
    return trial
}

function make_movegridgoalbomb_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentbombgoal() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to the most appropriate action to avoid the remaining bombs.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegridgoalbomb',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            sample_goal();
        },
        on_load: function(){
            showactions(room_index, "")
            updategoal()
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move(options)
            if(areArraysEqual(agent_TPos, goal_TPos)){
                data.success = true;
            } else {
                data.success = false;
            }
        }
    };
    return trial
}

function make_movegridgoalresultbomb_trial(room_index) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentbombgoal() + 
                `<div id="action_row" class="row"></div>` + 
                `<div><h3 id="feedback"></h3></div>` +
            `</div>`,
        trial_duration: 1000,
        choices: "NO_KEYS",
        data: {
            task: 'movegridgoalresultbomb',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_load: function(){
            showactions(room_index, "");
            updateagent();
            updategoal();
            
            var success = jsPsych.data.get().last(1).values()[0].success;
            const feedback = document.getElementById('feedback');
            if(success){
                feedback.textContent = "Yayy! You avoid the bombs! :)";
                feedback.style.color = "green";
            }else{
                feedback.textContent = "No! You ended up on a bomb... :(";
                feedback.style.color = "red";
            }
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// One-step moving in the grid practice
//----------------------------------------------------------------------------//
function make_movegridgoal_practice_trial(room_index, i_practiceG, i_practiceA) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentgoal() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to the most appropriate action to collect the gold.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegridgoal',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            sample_goal_practice(i_practiceG);
        },
        on_load: function(){
            showactions(room_index, "")
            updategoal()
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move_practice(options, i_practiceA)
            if(areArraysEqual(agent_TPos, goal_TPos)){
                data.success = true;
            } else {
                data.success = false;
            }
        }
    };
    return trial
}

function make_movegridgoalbomb_practice_trial(room_index, i_practiceG, i_practiceA) {
    var trial = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                initialize_agentbombgoal() + 
                `<div id="action_row" class="row"></div>` +
                `<div><h3>Press the key corresponding to the most appropriate action to avoid the remaining bombs.</h3></div>` +
            `</div>`,
        post_trial_gap: 75,
        choices: Object.keys(allrooms[room_index]),
        data: {
            task: 'movegridgoalbomb',
            agent_TPos: function(){return agent_TPos},
            goal_TPos: function(){return goal_TPos},
            room_index: room_index,
            possible_options: allrooms[room_index],
            n_act: Object.keys(allrooms[room_index]).length
        },
        on_start: function(){
            sample_goal_practice(i_practiceG);
        },
        on_load: function(){
            showactions(room_index, "")
            updategoal()
        },
        on_finish: function(data){
            options = data.possible_options[data.response]
            move_practice(options, i_practiceA)
            if(areArraysEqual(agent_TPos, goal_TPos)){
                data.success = true;
            } else {
                data.success = false;
            }
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// Making selection trials
//----------------------------------------------------------------------------//
function make_roomselection_trial(room_index1, room_index2, iftest = true) {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
            <div class="selection-spacer"><h4>You can choose one of these two rooms:</h4></div>
            <div class="selection-container">` +
                `<div class="selection-box">` + 
                    `<div style="display: flex; flex-direction: column; align-items: center; text-align: center">` + 
                        `<p><strong>Room 1</strong></p>` +
                        initialize_agent_selection(1) + 
                        `<div id="action_row1" class="row"></div>` +
                    `</div>`+
                `</div>` +
                `<div class="selection-box">` + 
                    `<div style="display: flex; flex-direction: column; align-items: center; text-align: center">` + 
                        `<p><strong>Room 2</strong></p>` +
                        initialize_agent_selection(2) + 
                        `<div id="action_row2" class="row"></div>` +
                    `</div>`+
                `</div>` +
            `</div>
            <div class="selection-spacer"><h4>Which one do you choose?</h4></div>
            `,
        choices: ["Room 1", "Room 2"],
        post_trial_gap: 75,
        trial_duration: 10000,
        data: {
            room_index1: room_index1,
            room_index2: room_index2
        },
        on_load: function(){
            selection_showactions(room_index1,"1")
            selection_showactions(room_index2,"2")
        },
        on_finish: function(data){
            if(data.response == 0){
                data.chosen_room_index = data.room_index1;
                data.timeout = false;
            } else if(data.response == 1) {
                data.chosen_room_index = data.room_index2;
                data.timeout = false;
            } else {
                data.chosen_room_index = -1;
                data.timeout = true;
            }
            if(iftest){
                selection_trial ++;
                data.selection_trial = selection_trial;
                data.task = 'roomselection';
                if(!data.timeout){
                    // collected_gold += sample_bonus(data.chosen_room_index)
                    collected_gold += bonus_probability(data.chosen_room_index)
                }
                data.collected_bomb = collected_bomb;
                data.collected_gold = collected_gold;
            }else{
                data.task = 'roomselection_practice';
            }
        }
    };
    return trial
}

function make_showselectedroom_trial() {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `<div></div>`,
        choices: [],
        data: {
            task: 'showselectedroom',
        },
        on_start: function(trial) {
            var iftimeout = jsPsych.data.get().last(1).values()[0].timeout;
            if(iftimeout){
                trial.stimulus = `<div><h1 style="color: red">You lost your chance to choose a room in 10 seconds...</h1></div>`;
                trial.choices = ['Continue']
            }else{
                trial.stimulus = `
                    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                        `<div><h3 id="feedback"></h3></div>` +    
                        initialize_agent() + 
                        `<div id="action_row" class="row"></div>` + 
                    `</div>`;
                trial.trial_duration = 1000;
            }
        },
        on_load: function(){
            var iftimeout = jsPsych.data.get().last(1).values()[0].timeout;
            if (!iftimeout){
                var room_index = jsPsych.data.get().last(1).values()[0].chosen_room_index;
                showactions(room_index,"")
                var chosen_room = jsPsych.data.get().last(1).values()[0].response + 1;
                feedback.textContent = "You chose Room " + chosen_room;
            }
        },
    };
    return trial
}

function make_roomselpracfeedback_trial(room_index1, room_index2) {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: function(){
            var iftimeout = jsPsych.data.get().last(2).values()[0].timeout;
            if(iftimeout){
                return `<div></div>`
            }else{
                var room_index = jsPsych.data.get().last(2).values()[0].chosen_room_index;
                var pmax = Math.max(bonus_probability(room_index1), bonus_probability(room_index2));
                var pchosen = bonus_probability(room_index)
                if((pchosen < pmax) || (pchosen < 0.5)){
                    return `<div class="gold-coins-container">
                        <img class="gold-coins-image" src="img/Cross.png" alt="Coin Image">
                        <div class="gold-coins-text">
                            <h1>No... We missed the gold!</h1>
                        </div>
                        <img class="gold-coins-image" src="img/Cross.png" alt="Coin Image">
                    </div>
                    <h1 style="color: red;">This feedback is ONLY for the practice trials.</h1>`
                }else{
                    return `<div class="gold-coins-container">
                        <img class="gold-coins-image" src="img/Goal.png" alt="Coin Image">
                        <div class="gold-coins-text">
                            <h1>Yayy! We collected the gold!</h1>
                        </div>
                        <img class="gold-coins-image" src="img/Goal.png" alt="Coin Image">
                    </div>
                    <h1 style="color: red;">This feedback is ONLY for the practice trials.</h1>`
                }
            }
        },
        choices: [],
        data: {
            task: 'roomselectionpracticefeedback',
        },
        on_start: function(trial) {
            var iftimeout = jsPsych.data.get().last(2).values()[0].timeout;
            if(iftimeout){
                trial.trial_duration = 10;
            }else{
                trial.choices = ['Continue']
            }
        }
    };
    return trial
}

//----------------------------------------------------------------------------//
// Making selection trials bomb
//----------------------------------------------------------------------------//
function make_roomselectionbomb_trial(room_index1, room_index2, iftest = true) {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
            <div class="selection-spacer"><h4>You can choose one of these two rooms:</h4></div>
            <div class="selection-container">` +
                `<div class="selection-box">` + 
                    `<div style="display: flex; flex-direction: column; align-items: center; text-align: center">` + 
                        `<p><strong>Room 1</strong></p>` +
                        initialize_agentbomb_selection(1) + 
                        `<div id="action_row1" class="row"></div>` +
                    `</div>`+
                `</div>` +
                `<div class="selection-box">` + 
                    `<div style="display: flex; flex-direction: column; align-items: center; text-align: center">` + 
                        `<p><strong>Room 2</strong></p>` +
                        initialize_agentbomb_selection(2) + 
                        `<div id="action_row2" class="row"></div>` +
                    `</div>`+
                `</div>` +
            `</div>
            <div class="selection-spacer"><h4>Which one do you choose?</h4></div>
            `,
        choices: ["Room 1", "Room 2"],
        post_trial_gap: 75,
        trial_duration: 10000,
        data: {
            room_index1: room_index1,
            room_index2: room_index2
        },
        on_load: function(){
            selection_showactions(room_index1,"1")
            selection_showactions(room_index2,"2")
        },
        on_finish: function(data){
            if(data.response == 0){
                data.chosen_room_index = data.room_index1;
                data.timeout = false;
            } else if(data.response == 1) {
                data.chosen_room_index = data.room_index2;
                data.timeout = false;
            } else {
                data.chosen_room_index = -1;
                data.timeout = true;
            }
            if(iftest){
                selection_trial ++;
                data.selection_trial = selection_trial;
                data.task = 'roomselectionbomb';
                if(!data.timeout){
                    // collected_bomb += 1 - sample_bonus(data.chosen_room_index)
                    collected_bomb += 1 - bonus_probability(data.chosen_room_index)
                }
                data.collected_bomb = collected_bomb;
                data.collected_gold = collected_gold;
            }else{
                data.task = 'roomselection_practicebomb';
            }
        }
    };
    return trial
}

function make_showselectedroombomb_trial() {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `<div></div>`,
        choices: [],
        data: {
            task: 'showselectedroombomb',
        },
        on_start: function(trial) {
            var iftimeout = jsPsych.data.get().last(1).values()[0].timeout;
            if(iftimeout){
                trial.stimulus = `<div><h1 style="color: red">You lost your chance to choose a room in 10 seconds...</h1></div>`;
                trial.choices = ['Continue']
            }else{
                trial.stimulus = `
                    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">` + 
                        `<div><h3 id="feedback"></h3></div>` +    
                        initialize_agentbomb() + 
                        `<div id="action_row" class="row"></div>` + 
                    `</div>`;
                trial.trial_duration = 1000;
            }
        },
        on_load: function(){
            var iftimeout = jsPsych.data.get().last(1).values()[0].timeout;
            if (!iftimeout){
                var room_index = jsPsych.data.get().last(1).values()[0].chosen_room_index;
                showactions(room_index,"")
                var chosen_room = jsPsych.data.get().last(1).values()[0].response + 1;
                feedback.textContent = "You chose Room " + chosen_room;
            }
        },
    };
    return trial
}

function make_roomselpracfeedbackbomb_trial(room_index1, room_index2) {
    var trial = {
        type: jsPsychHtmlButtonResponse,
        stimulus: function(){
            var iftimeout = jsPsych.data.get().last(2).values()[0].timeout;
            if(iftimeout){
                return `<div></div>`
            }else{
                var room_index = jsPsych.data.get().last(2).values()[0].chosen_room_index;
                var pmax = Math.max(bonus_probability(room_index1), bonus_probability(room_index2));
                var pchosen = bonus_probability(room_index)
                if((pchosen < pmax) || (pchosen < 0.5)){
                    return `<div class="gold-coins-container">
                        <img class="gold-coins-image" src="img/Bomb.png" alt="Coin Image">
                        <div class="gold-coins-text">
                            <h1>No... We ended up on a bomb!</h1>
                        </div>
                        <img class="gold-coins-image" src="img/Bomb.png" alt="Coin Image">
                    </div>
                    <h1 style="color: red;">This feedback is ONLY for the practice trials.</h1>`
                }else{
                    return `<div class="gold-coins-container">
                        <img class="gold-coins-image" src="img/Check.png" alt="Coin Image">
                        <div class="gold-coins-text">
                            <h1>Yayy! We avoided the remaining bombs!</h1>
                        </div>
                        <img class="gold-coins-image" src="img/Check.png" alt="Coin Image">
                    </div>
                    <h1 style="color: red;">This feedback is ONLY for the practice trials.</h1>`
                }
            }
        },
        choices: [],
        data: {
            task: 'roomselectionpracticefeedbackbomb',
        },
        on_start: function(trial) {
            var iftimeout = jsPsych.data.get().last(2).values()[0].timeout;
            if(iftimeout){
                trial.trial_duration = 10;
            }else{
                trial.choices = ['Continue']
            }
        }
    };
    return trial
}
