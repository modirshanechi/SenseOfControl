var jsPsychGold1stepComprehension = (function (jspsych) {
  'use strict';

  const info = {
    name: 'gold1-comprehension',
    description: '',
    parameters: {
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      }
    }
  }
  class Gold1ComprehensionPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      // Plug-in setup
      var plugin_id_name = "jspsych-survey-multi-choice";
      var plugin_id_selector = '#' + plugin_id_name;
      var _join = function( /*args*/ ) {
        var arr = Array.prototype.slice.call(arguments, _join.length);
        return arr.join('-');
      }

      // ---------------------------------- //
      // Section 1: Define HTML             //
      // ---------------------------------- //

      // Initialize HTML
      var html = "";

      // inject CSS for trial
      html += '<style id="jspsych-survey-multi-choice-css">';
      html += ".jspsych-survey-multi-choice-question { margin-top: 0em; margin-bottom: 1.0em; text-align: left; padding-left: 8em}"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-text { text-align: left; margin: 0em 0em 0.5em 0em }"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-option { display: inline-block;  margin: 0em 1em 0em 1em; vertical-align: max-width: 5em}"+
      "label.jspsych-survey-multi-choice-text input[type='radio'] {margin-right: 1em;}"+
      ".invalid { display:inline-block; border: 1px solid; border-radius: 4px; margin: 0.25em 1em 0em 1em; padding: 0.5px 4px 0.5px 4px; color: #D8000C; background-color: #FFBABA; font-size: 14px; animation: flash 0.1s}"+
      ".valid { display: none }"+
      "@keyframes flash { from { opacity: 45%; } to { opacity: 100%; } }"
      html += '</style>';

      // ---------------------------------- //
      // Initialize check                   //
      // ---------------------------------- //

      // form element
      var trial_form_id = _join(plugin_id_name, "form");
      display_element.innerHTML += '<form id="'+trial_form_id+'"></form>';

      // Show preamble text
      html += '<div id="jspsych-survey-multi-choice-preamble" class="jspsych-survey-multi-choice-preamble"><h3>Please answer the questions below:</h3></div>';

      // Initialize form element
      html += '<form id="jspsych-survey-multi-choice-form">';

      
      // ---------------------------------- //
      // Comprehension Question #2         //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-1" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>1. What are you supposed to do in this experiment?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-1-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-0">To <strong>collect</strong> gold!</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-1-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-1">To <strong>move</strong> in the rooms.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-1-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-2">To <strong>watch</strong> the computer collecting gold.</label>';
      html += '</div>';

      // Option 4:
      html += '<div id="jspsych-survey-multi-choice-option-1-3" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-3" value="4" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-3">To <strong>choose</strong> a room for gold collecting.</label>';
      html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q2-error"></p>'
      html += '</div>';

      // ---------------------------------- //
      // Comprehension Question #4          //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-3" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="bonus">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>2. Which of the following statements is correct?</strong></p>';

      // Option 1:
      html += '<div id="jspsych-survey-multi-choice-option-3-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-0" value=1 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-0">Some actions have <strong>more than one</strong> endpoint.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-3-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-1" value=2 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-1">The only difference between different rooms is in the <strong>number</strong> available actions.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-3-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-2" value=3 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-2">It is <strong>always possible</strong> to collect the gold.</label>';
      html += '</div>';


      // Close item
      html += '<br><p id="Q4-error"></p>'
      html += '</div>';

      // ---------------------------------- //
      // Comprehension Question #1        //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-0" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>3. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-0-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-0-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-1">Yes, <strong>Room 2</strong></label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-0-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-2"><strong>No</strong>, both rooms are equally good.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-0-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-3"><strong>No</strong>, both rooms are equally good.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q1-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide1.png" style="max-width: 40%"></img></div>';



      // ---------------------------------- //
      // Finalize HTML                      //
      // ---------------------------------- //

      // add submit button
      html += '<input type="submit" id="'+plugin_id_name+'-next" class="'+plugin_id_name+' jspsych-btn"' + (trial.button_label ? ' value="'+trial.button_label + '"': '') + 'style="margin-bottom: 5px;" disabled></input>';

      // End HTML
      html += '</form>';

      // Display HTML
      display_element.innerHTML = html;

      // ---------------------------------- //
      // Section 2: jsPsych Functions       //
      // ---------------------------------- //

      // Define error messages
      const Q1 = document.getElementById("Q1-error");
      const Q2 = document.getElementById("Q2-error");
      // const Q3 = document.getElementById("Q3-error");
      const Q4 = document.getElementById("Q4-error");
      var count = 0;

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-0').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-0 input:checked').value;

        // Validation
        if (val === "2") {

          // Update text
          Q1.innerHTML = "";
          Q1.className = "valid"

        } else {

          // Update text
          Q1.innerHTML = "That's incorrect. <br>Hint: Think about potential positions of the gold coin and how we can collect it for you.";
          Q1.className = "invalid"

          // Restart animation
          Q1.style.animation = 'none';
          Q1.offsetHeight; /* trigger reflow */
          Q1.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-1').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-1 input:checked').value;

        // Validation
        if (val === "4") {

          // Update text
          Q2.innerHTML = "";
          Q2.className = "valid"

        } else {

          // Update text
          Q2.innerHTML = "That's incorrect. <br>Hint: WE collect the gold for you, but in which rooms?";
          Q2.className = "invalid"

          // Restart animation
          Q2.style.animation = 'none';
          Q2.offsetHeight; /* trigger reflow */
          Q2.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // // Detect changes on second comprehension item
      // display_element.querySelector('#jspsych-survey-multi-choice-2').addEventListener('change', function(){

      //   // On change, find which item is checked.
      //   var val = display_element.querySelector('#jspsych-survey-multi-choice-2 input:checked').value;

      //   // Validation
      //   if (val === "3") {

      //     // Update text
      //     Q3.innerHTML = "";
      //     Q3.className = "valid"

      //   } else {

      //     // Update text
      //     Q3.innerHTML = "That's incorrect. <br>Hint: think about differences betweem different rooms.";
      //     Q3.className = "invalid"

      //     // Restart animation
      //     Q3.style.animation = 'none';
      //     Q3.offsetHeight; /* trigger reflow */
      //     Q3.style.animation = null;

      //     // Increment error count
      //     count += 1;

      //   }

      // });

      // Detect changes on third comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-3').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-3 input:checked').value;

        // Validation
        if (val === "1") {

          // Update text
          Q4.innerHTML = "";
          Q4.className = "valid"

        } else {

          // Update text
          Q4.innerHTML = "That's incorrect. Think about the practice trials we did together!";
          Q4.className = "invalid"

          // Restart animation
          Q4.style.animation = 'none';
          Q4.offsetHeight; /* trigger reflow */
          Q4.style.animation = null;

          // Increment error count
          count += 1;

        }

      })

      // Detect if all correct answers
      display_element.addEventListener('change', function(){
        // if (Q1.className === 'valid' && Q2.className === 'valid' && Q3.className === 'valid' && Q4.className === 'valid') {
        if (Q1.className === 'valid' && Q2.className === 'valid' && Q4.className === 'valid') {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = false;
        } else {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = true;
        }
      })

      // Detect submit button press
      document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault();

        // Measure response time
        var endTime = performance.now();
        var response_time = endTime - startTime;

        // Gather responses
        var question_data = {};
        // for (var i=0; i<4; i++) {
        for (let i of [0, 1, 3]){

          // Find matching question.
          var match = display_element.querySelector('#jspsych-survey-multi-choice-'+i);
          var name = match.attributes['data-name'].value;
          var val = match.querySelector("input[type=radio]:checked").value;

          // Store response
          var obje = {};
          obje[name] = val;
          Object.assign(question_data, obje);

        }

        // Save data
        var trial_data = {
          "rt": response_time,
          "responses": JSON.stringify(question_data),
          "errors": count
        };
        display_element.innerHTML += '';

        // next trial
        jsPsych.finishTrial(trial_data);
      });

      var startTime = performance.now();

    }
  }
  Gold1ComprehensionPlugin.info = info;

  return Gold1ComprehensionPlugin;

})(jsPsychModule);


// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
var jsPsychGold1stepComprehension2 = (function (jspsych) {
  'use strict';

  const info = {
    name: 'gold1-comprehension2',
    description: '',
    parameters: {
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      }
    }
  }
  class Gold1ComprehensionPlugin2 {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      // Plug-in setup
      var plugin_id_name = "jspsych-survey-multi-choice";
      var plugin_id_selector = '#' + plugin_id_name;
      var _join = function( /*args*/ ) {
        var arr = Array.prototype.slice.call(arguments, _join.length);
        return arr.join('-');
      }

      // ---------------------------------- //
      // Section 1: Define HTML             //
      // ---------------------------------- //

      // Initialize HTML
      var html = "";

      // inject CSS for trial
      html += '<style id="jspsych-survey-multi-choice-css">';
      html += ".jspsych-survey-multi-choice-question { margin-top: 0em; margin-bottom: 1.0em; text-align: left; padding-left: 8em}"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-text { text-align: left; margin: 0em 0em 0.5em 0em }"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-option { display: inline-block;  margin: 0em 1em 0em 1em; vertical-align: max-width: 5em}"+
      "label.jspsych-survey-multi-choice-text input[type='radio'] {margin-right: 1em;}"+
      ".invalid { display:inline-block; border: 1px solid; border-radius: 4px; margin: 0.25em 1em 0em 1em; padding: 0.5px 4px 0.5px 4px; color: #D8000C; background-color: #FFBABA; font-size: 14px; animation: flash 0.1s}"+
      ".valid { display: none }"+
      "@keyframes flash { from { opacity: 45%; } to { opacity: 100%; } }"
      html += '</style>';

      // ---------------------------------- //
      // Initialize check                   //
      // ---------------------------------- //

      // form element
      var trial_form_id = _join(plugin_id_name, "form");
      display_element.innerHTML += '<form id="'+trial_form_id+'"></form>';

      // Show preamble text
      html += '<div id="jspsych-survey-multi-choice-preamble" class="jspsych-survey-multi-choice-preamble"><h3>Please answer the questions below:</h3></div>';

      // Initialize form element
      html += '<form id="jspsych-survey-multi-choice-form">';

      
      // ---------------------------------- //
      // Comprehension Question #2         //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-1" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>1. Consider the room below. <br>How could we increase the chance of gold-collecting in this room?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-1-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-0">By making actions <strong>probabilitstic</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-1-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-1">By adding <strong>more actions</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-1-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-2">The chance of collecting the gold is <strong>already maximum</strong> in this room.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-1-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-3">To <strong>choose</strong> a room where the gold will be collected.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q2-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide2.png" style="max-width: 45%"></img></div>';


      // ---------------------------------- //
      // Comprehension Question #4          //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-3" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="bonus">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>2. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1:
      html += '<div id="jspsych-survey-multi-choice-option-3-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-0" value=1 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-3-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-1" value=2 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-1">Yes, <strong>Room 2</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-3-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-2" value=3 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-2"><strong>No</strong>, both rooms are equally good.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-3-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-3" value=4 required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-3">It is always more likely that the gold appears in the top three loca <strong>more often</strong> in some locations than the others.</label>';
      // html += '</div>';


      // Close item
      html += '<br><p id="Q4-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide3.png" style="max-width: 45%"></img></div>';

      // ---------------------------------- //
      // Comprehension Question #1        //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-0" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>3. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-0-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-0-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-1">Yes, <strong>Room 2</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-0-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-2"><strong>No</strong>, both rooms are equally good.</label></label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-0-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-3"><strong>No</strong>, both rooms are equally good.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q1-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide4.png" style="max-width: 45%"></img></div>';



      // ---------------------------------- //
      // Finalize HTML                      //
      // ---------------------------------- //

      // add submit button
      html += '<input type="submit" id="'+plugin_id_name+'-next" class="'+plugin_id_name+' jspsych-btn"' + (trial.button_label ? ' value="'+trial.button_label + '"': '') + 'style="margin-bottom: 5px;" disabled></input>';

      // End HTML
      html += '</form>';

      // Display HTML
      display_element.innerHTML = html;

      // ---------------------------------- //
      // Section 2: jsPsych Functions       //
      // ---------------------------------- //

      // Define error messages
      const Q1 = document.getElementById("Q1-error");
      const Q2 = document.getElementById("Q2-error");
      // const Q3 = document.getElementById("Q3-error");
      const Q4 = document.getElementById("Q4-error");
      var count = 0;

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-0').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-0 input:checked').value;

        // Validation
        if (val === "3") {

          // Update text
          Q1.innerHTML = "";
          Q1.className = "valid"

        } else {

          // Update text
          Q1.innerHTML = "That's incorrect. <br>Hint: Remember that the gold coin can appear in ANY of the 8 empty squares, with EQUAL probability.";
          Q1.className = "invalid"

          // Restart animation
          Q1.style.animation = 'none';
          Q1.offsetHeight; /* trigger reflow */
          Q1.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-1').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-1 input:checked').value;

        // Validation
        if (val === "3") {

          // Update text
          Q2.innerHTML = "";
          Q2.className = "valid"

        } else {

          // Update text
          Q2.innerHTML = "That's incorrect. <br>Hint: Think about potential positions of the gold coin and how we can collect it for you.";
          Q2.className = "invalid"

          // Restart animation
          Q2.style.animation = 'none';
          Q2.offsetHeight; /* trigger reflow */
          Q2.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // // Detect changes on second comprehension item
      // display_element.querySelector('#jspsych-survey-multi-choice-2').addEventListener('change', function(){

      //   // On change, find which item is checked.
      //   var val = display_element.querySelector('#jspsych-survey-multi-choice-2 input:checked').value;

      //   // Validation
      //   if (val === "3") {

      //     // Update text
      //     Q3.innerHTML = "";
      //     Q3.className = "valid"

      //   } else {

      //     // Update text
      //     Q3.innerHTML = "That's incorrect. <br>Hint: think about differences betweem different rooms.";
      //     Q3.className = "invalid"

      //     // Restart animation
      //     Q3.style.animation = 'none';
      //     Q3.offsetHeight; /* trigger reflow */
      //     Q3.style.animation = null;

      //     // Increment error count
      //     count += 1;

      //   }

      // });

      // Detect changes on third comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-3').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-3 input:checked').value;

        // Validation
        if (val === "1") {

          // Update text
          Q4.innerHTML = "";
          Q4.className = "valid"

        } else {

          // Update text
          Q4.innerHTML = "That's incorrect. <br>Hint: Think about the chance of collecting the gold coin if it appears in a reachable location.";
          Q4.className = "invalid"

          // Restart animation
          Q4.style.animation = 'none';
          Q4.offsetHeight; /* trigger reflow */
          Q4.style.animation = null;

          // Increment error count
          count += 1;

        }

      })

      // Detect if all correct answers
      display_element.addEventListener('change', function(){
        // if (Q1.className === 'valid' && Q2.className === 'valid' && Q3.className === 'valid' && Q4.className === 'valid') {
        if (Q1.className === 'valid' && Q2.className === 'valid' && Q4.className === 'valid') {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = false;
        } else {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = true;
        }
      })

      // Detect submit button press
      document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault();

        // Measure response time
        var endTime = performance.now();
        var response_time = endTime - startTime;

        // Gather responses
        var question_data = {};
        // for (var i=0; i<4; i++) {
        for (let i of [0, 1, 3]){

          // Find matching question.
          var match = display_element.querySelector('#jspsych-survey-multi-choice-'+i);
          var name = match.attributes['data-name'].value;
          var val = match.querySelector("input[type=radio]:checked").value;

          // Store response
          var obje = {};
          obje[name] = val;
          Object.assign(question_data, obje);

        }

        // Save data
        var trial_data = {
          "rt": response_time,
          "responses": JSON.stringify(question_data),
          "errors": count
        };
        display_element.innerHTML += '';

        // next trial
        jsPsych.finishTrial(trial_data);
      });

      var startTime = performance.now();

    }
  }
  Gold1ComprehensionPlugin2.info = info;

  return Gold1ComprehensionPlugin2;

})(jsPsychModule);


// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
var jsPsychGold1stepComprehension3 = (function (jspsych) {
  'use strict';

  const info = {
    name: 'gold1-Comprehension3',
    description: '',
    parameters: {
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      }
    }
  }
  class Gold1ComprehensionPlugin3 {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      // Plug-in setup
      var plugin_id_name = "jspsych-survey-multi-choice";
      var plugin_id_selector = '#' + plugin_id_name;
      var _join = function( /*args*/ ) {
        var arr = Array.prototype.slice.call(arguments, _join.length);
        return arr.join('-');
      }

      // ---------------------------------- //
      // Section 1: Define HTML             //
      // ---------------------------------- //

      // Initialize HTML
      var html = "";

      // inject CSS for trial
      html += '<style id="jspsych-survey-multi-choice-css">';
      html += ".jspsych-survey-multi-choice-question { margin-top: 0em; margin-bottom: 1.0em; text-align: left; padding-left: 8em}"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-text { text-align: left; margin: 0em 0em 0.5em 0em }"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-option { display: inline-block;  margin: 0em 1em 0em 1em; vertical-align: max-width: 5em}"+
      "label.jspsych-survey-multi-choice-text input[type='radio'] {margin-right: 1em;}"+
      ".invalid { display:inline-block; border: 1px solid; border-radius: 4px; margin: 0.25em 1em 0em 1em; padding: 0.5px 4px 0.5px 4px; color: #D8000C; background-color: #FFBABA; font-size: 14px; animation: flash 0.1s}"+
      ".valid { display: none }"+
      "@keyframes flash { from { opacity: 45%; } to { opacity: 100%; } }"
      html += '</style>';

      // ---------------------------------- //
      // Initialize check                   //
      // ---------------------------------- //

      // form element
      var trial_form_id = _join(plugin_id_name, "form");
      display_element.innerHTML += '<form id="'+trial_form_id+'"></form>';

      // Show preamble text
      html += '<div id="jspsych-survey-multi-choice-preamble" class="jspsych-survey-multi-choice-preamble"><h3>Please answer the questions below:</h3></div>';

      // Initialize form element
      html += '<form id="jspsych-survey-multi-choice-form">';

      
      // ---------------------------------- //
      // Comprehension Question #2         //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-1" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>1. Consider the room below. <br>How could we increase the chance of bomb-avoiding in this room?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-1-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-0">By making actions <strong>probabilitstic</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-1-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-1">By adding <strong>more actions</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-1-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-2">The chance of avoiding bombs is <strong>already maximum</strong> in this room.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-1-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-3">To <strong>choose</strong> a room where the gold will be collected.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q2-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide6.png" style="max-width: 45%"></img></div>';


      // ---------------------------------- //
      // Comprehension Question #4          //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-3" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="bonus">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>2. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1:
      html += '<div id="jspsych-survey-multi-choice-option-3-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-0" value=1 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-3-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-1" value=2 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-1">Yes, <strong>Room 2</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-3-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-2" value=3 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-2"><strong>No</strong>, both rooms are equally good.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-3-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-3" value=4 required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-3">It is always more likely that the gold appears in the top three loca <strong>more often</strong> in some locations than the others.</label>';
      // html += '</div>';


      // Close item
      html += '<br><p id="Q4-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide7.png" style="max-width: 45%"></img></div>';

      // ---------------------------------- //
      // Comprehension Question #1        //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-0" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>3. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-0-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-0-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-1">Yes, <strong>Room 2</strong>.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-0-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-2"><strong>No</strong>, both rooms are equally good.</label></label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-0-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-3"><strong>No</strong>, both rooms are equally good.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q1-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide8.png" style="max-width: 45%"></img></div>';



      // ---------------------------------- //
      // Finalize HTML                      //
      // ---------------------------------- //

      // add submit button
      html += '<input type="submit" id="'+plugin_id_name+'-next" class="'+plugin_id_name+' jspsych-btn"' + (trial.button_label ? ' value="'+trial.button_label + '"': '') + 'style="margin-bottom: 5px;" disabled></input>';

      // End HTML
      html += '</form>';

      // Display HTML
      display_element.innerHTML = html;

      // ---------------------------------- //
      // Section 2: jsPsych Functions       //
      // ---------------------------------- //

      // Define error messages
      const Q1 = document.getElementById("Q1-error");
      const Q2 = document.getElementById("Q2-error");
      // const Q3 = document.getElementById("Q3-error");
      const Q4 = document.getElementById("Q4-error");
      var count = 0;

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-0').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-0 input:checked').value;

        // Validation
        if (val === "3") {

          // Update text
          Q1.innerHTML = "";
          Q1.className = "valid"

        } else {

          // Update text
          Q1.innerHTML = "That's incorrect. <br>Hint: Remember that ANY of the 8 bombs can disappear, with EQUAL probability.";
          Q1.className = "invalid"

          // Restart animation
          Q1.style.animation = 'none';
          Q1.offsetHeight; /* trigger reflow */
          Q1.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-1').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-1 input:checked').value;

        // Validation
        if (val === "3") {

          // Update text
          Q2.innerHTML = "";
          Q2.className = "valid"

        } else {

          // Update text
          Q2.innerHTML = "That's incorrect. <br>Hint: Think about potential safe positions (without bomb) and how we can reach it for you.";
          Q2.className = "invalid"

          // Restart animation
          Q2.style.animation = 'none';
          Q2.offsetHeight; /* trigger reflow */
          Q2.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // // Detect changes on second comprehension item
      // display_element.querySelector('#jspsych-survey-multi-choice-2').addEventListener('change', function(){

      //   // On change, find which item is checked.
      //   var val = display_element.querySelector('#jspsych-survey-multi-choice-2 input:checked').value;

      //   // Validation
      //   if (val === "3") {

      //     // Update text
      //     Q3.innerHTML = "";
      //     Q3.className = "valid"

      //   } else {

      //     // Update text
      //     Q3.innerHTML = "That's incorrect. <br>Hint: think about differences betweem different rooms.";
      //     Q3.className = "invalid"

      //     // Restart animation
      //     Q3.style.animation = 'none';
      //     Q3.offsetHeight; /* trigger reflow */
      //     Q3.style.animation = null;

      //     // Increment error count
      //     count += 1;

      //   }

      // });

      // Detect changes on third comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-3').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-3 input:checked').value;

        // Validation
        if (val === "1") {

          // Update text
          Q4.innerHTML = "";
          Q4.className = "valid"

        } else {

          // Update text
          Q4.innerHTML = "That's incorrect. <br>Hint: Think about the chance of avoiding the bombs if the safe location is reachable.";
          Q4.className = "invalid"

          // Restart animation
          Q4.style.animation = 'none';
          Q4.offsetHeight; /* trigger reflow */
          Q4.style.animation = null;

          // Increment error count
          count += 1;

        }

      })

      // Detect if all correct answers
      display_element.addEventListener('change', function(){
        // if (Q1.className === 'valid' && Q2.className === 'valid' && Q3.className === 'valid' && Q4.className === 'valid') {
        if (Q1.className === 'valid' && Q2.className === 'valid' && Q4.className === 'valid') {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = false;
        } else {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = true;
        }
      })

      // Detect submit button press
      document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault();

        // Measure response time
        var endTime = performance.now();
        var response_time = endTime - startTime;

        // Gather responses
        var question_data = {};
        // for (var i=0; i<4; i++) {
        for (let i of [0, 1, 3]){

          // Find matching question.
          var match = display_element.querySelector('#jspsych-survey-multi-choice-'+i);
          var name = match.attributes['data-name'].value;
          var val = match.querySelector("input[type=radio]:checked").value;

          // Store response
          var obje = {};
          obje[name] = val;
          Object.assign(question_data, obje);

        }

        // Save data
        var trial_data = {
          "rt": response_time,
          "responses": JSON.stringify(question_data),
          "errors": count
        };
        display_element.innerHTML += '';

        // next trial
        jsPsych.finishTrial(trial_data);
      });

      var startTime = performance.now();

    }
  }
  Gold1ComprehensionPlugin3.info = info;

  return Gold1ComprehensionPlugin3;

})(jsPsychModule);


// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

var jsPsychGold1stepComprehensionBG = (function (jspsych) {
  'use strict';

  const info = {
    name: 'gold1-comprehensionBG',
    description: '',
    parameters: {
      button_label: {
        type: jspsych.ParameterType.STRING,
        pretty_name: 'Button label',
        default:  'Continue',
        description: 'Label of the button.'
      }
    }
  }
  class Gold1ComprehensionPluginBG {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {

      // Plug-in setup
      var plugin_id_name = "jspsych-survey-multi-choice";
      var plugin_id_selector = '#' + plugin_id_name;
      var _join = function( /*args*/ ) {
        var arr = Array.prototype.slice.call(arguments, _join.length);
        return arr.join('-');
      }

      // ---------------------------------- //
      // Section 1: Define HTML             //
      // ---------------------------------- //

      // Initialize HTML
      var html = "";

      // inject CSS for trial
      html += '<style id="jspsych-survey-multi-choice-css">';
      html += ".jspsych-survey-multi-choice-question { margin-top: 0em; margin-bottom: 1.0em; text-align: left; padding-left: 8em}"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-text { text-align: left; margin: 0em 0em 0.5em 0em }"+
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-option { display: inline-block;  margin: 0em 1em 0em 1em; vertical-align: max-width: 5em}"+
      "label.jspsych-survey-multi-choice-text input[type='radio'] {margin-right: 1em;}"+
      ".invalid { display:inline-block; border: 1px solid; border-radius: 4px; margin: 0.25em 1em 0em 1em; padding: 0.5px 4px 0.5px 4px; color: #D8000C; background-color: #FFBABA; font-size: 14px; animation: flash 0.1s}"+
      ".valid { display: none }"+
      "@keyframes flash { from { opacity: 45%; } to { opacity: 100%; } }"
      html += '</style>';

      // ---------------------------------- //
      // Initialize check                   //
      // ---------------------------------- //

      // form element
      var trial_form_id = _join(plugin_id_name, "form");
      display_element.innerHTML += '<form id="'+trial_form_id+'"></form>';

      // Show preamble text
      html += '<div id="jspsych-survey-multi-choice-preamble" class="jspsych-survey-multi-choice-preamble"><h3>Please answer the questions below:</h3></div>';

      // Initialize form element
      html += '<form id="jspsych-survey-multi-choice-form">';

      
      // ---------------------------------- //
      // Comprehension Question #2         //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-1" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>1. What are you supposed to do in this experiment?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-1-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-0">To <strong>avoid</strong> bombs!</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-1-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-1">To <strong>move</strong> in the rooms.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-1-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-2">To <strong>watch</strong> the computer avoiding bombs.</label>';
      html += '</div>';

      // Option 4:
      html += '<div id="jspsych-survey-multi-choice-option-1-3" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-1" id="jspsych-survey-multi-choice-response-1-3" value="4" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-1-3">To <strong>choose</strong> a room for bomb avoiding.</label>';
      html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q2-error"></p>'
      html += '</div>';

      // ---------------------------------- //
      // Comprehension Question #4          //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-3" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="bonus">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>2. Which of the following statements is correct?</strong></p>';

      // Option 1:
      html += '<div id="jspsych-survey-multi-choice-option-3-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-0" value=1 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-0">Some actions have <strong>more than one</strong> endpoint.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-3-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-1" value=2 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-1">The only difference between different rooms is in the <strong>number</strong> available actions.</label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-3-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-3" id="jspsych-survey-multi-choice-response-3-2" value=3 required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-3-2">It is <strong>always possible</strong> to avoid the bombs.</label>';
      html += '</div>';


      // Close item
      html += '<br><p id="Q4-error"></p>'
      html += '</div>';

      // ---------------------------------- //
      // Comprehension Question #1        //
      // ---------------------------------- //

      // Initialize item
      html += '<div id="jspsych-survey-multi-choice-0" class="jspsych-survey-multi-choice-question jspsych-survey-multi-choice-horizontal" data-name="outcome">';

      // Add question text
      html += '<p class="jspsych-survey-multi-choice-text survey-multi-choice"><strong>3. Is any of the two rooms below better than the other one?</strong></p>';

      // Option 1: 
      html += '<div id="jspsych-survey-multi-choice-option-0-0" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-0" value="1" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-0">Yes, <strong>Room 1</strong>.</label>';
      html += '</div>';

      // Option 2:
      html += '<div id="jspsych-survey-multi-choice-option-0-1" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-1" value="2" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-1">Yes, <strong>Room 2</strong></label>';
      html += '</div>';

      // Option 3:
      html += '<div id="jspsych-survey-multi-choice-option-0-2" class="jspsych-survey-multi-choice-option">';
      html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-2" value="3" required>';
      html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-2"><strong>No</strong>, both rooms are equally good.</label>';
      html += '</div>';

      // // Option 4:
      // html += '<div id="jspsych-survey-multi-choice-option-0-3" class="jspsych-survey-multi-choice-option">';
      // html += '<input type="radio" name="jspsych-survey-multi-choice-response-0" id="jspsych-survey-multi-choice-response-0-3" value="4" required>';
      // html += '<label class="jspsych-survey-multi-choice-text" for="jspsych-survey-multi-choice-response-0-3"><strong>No</strong>, both rooms are equally good.</label>';
      // html += '</div>';

      // Close item
      html += '<br><p class="error" id="Q1-error"></p>'
      html += '</div>';

      // Show preamble text
      html += '<div class="jspsych-survey-multi-choice-preamble"><img src="./' + comp_path +  'Slide1.png" style="max-width: 40%"></img></div>';



      // ---------------------------------- //
      // Finalize HTML                      //
      // ---------------------------------- //

      // add submit button
      html += '<input type="submit" id="'+plugin_id_name+'-next" class="'+plugin_id_name+' jspsych-btn"' + (trial.button_label ? ' value="'+trial.button_label + '"': '') + 'style="margin-bottom: 5px;" disabled></input>';

      // End HTML
      html += '</form>';

      // Display HTML
      display_element.innerHTML = html;

      // ---------------------------------- //
      // Section 2: jsPsych Functions       //
      // ---------------------------------- //

      // Define error messages
      const Q1 = document.getElementById("Q1-error");
      const Q2 = document.getElementById("Q2-error");
      // const Q3 = document.getElementById("Q3-error");
      const Q4 = document.getElementById("Q4-error");
      var count = 0;

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-0').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-0 input:checked').value;

        // Validation
        if (val === "2") {

          // Update text
          Q1.innerHTML = "";
          Q1.className = "valid"

        } else {

          // Update text
          Q1.innerHTML = "That's incorrect. <br>Hint: Think about potential safe locations and how we can get there for you.";
          Q1.className = "invalid"

          // Restart animation
          Q1.style.animation = 'none';
          Q1.offsetHeight; /* trigger reflow */
          Q1.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // Detect changes on first comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-1').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-1 input:checked').value;

        // Validation
        if (val === "4") {

          // Update text
          Q2.innerHTML = "";
          Q2.className = "valid"

        } else {

          // Update text
          Q2.innerHTML = "That's incorrect. <br>Hint: WE avoid the bombs for you, but in which rooms?";
          Q2.className = "invalid"

          // Restart animation
          Q2.style.animation = 'none';
          Q2.offsetHeight; /* trigger reflow */
          Q2.style.animation = null;

          // Increment error count
          count += 1;

        }

      });

      // // Detect changes on second comprehension item
      // display_element.querySelector('#jspsych-survey-multi-choice-2').addEventListener('change', function(){

      //   // On change, find which item is checked.
      //   var val = display_element.querySelector('#jspsych-survey-multi-choice-2 input:checked').value;

      //   // Validation
      //   if (val === "3") {

      //     // Update text
      //     Q3.innerHTML = "";
      //     Q3.className = "valid"

      //   } else {

      //     // Update text
      //     Q3.innerHTML = "That's incorrect. <br>Hint: think about differences betweem different rooms.";
      //     Q3.className = "invalid"

      //     // Restart animation
      //     Q3.style.animation = 'none';
      //     Q3.offsetHeight; /* trigger reflow */
      //     Q3.style.animation = null;

      //     // Increment error count
      //     count += 1;

      //   }

      // });

      // Detect changes on third comprehension item
      display_element.querySelector('#jspsych-survey-multi-choice-3').addEventListener('change', function(){

        // On change, find which item is checked.
        var val = display_element.querySelector('#jspsych-survey-multi-choice-3 input:checked').value;

        // Validation
        if (val === "1") {

          // Update text
          Q4.innerHTML = "";
          Q4.className = "valid"

        } else {

          // Update text
          Q4.innerHTML = "That's incorrect. Think about the practice trials we did together!";
          Q4.className = "invalid"

          // Restart animation
          Q4.style.animation = 'none';
          Q4.offsetHeight; /* trigger reflow */
          Q4.style.animation = null;

          // Increment error count
          count += 1;

        }

      })

      // Detect if all correct answers
      display_element.addEventListener('change', function(){
        // if (Q1.className === 'valid' && Q2.className === 'valid' && Q3.className === 'valid' && Q4.className === 'valid') {
        if (Q1.className === 'valid' && Q2.className === 'valid' && Q4.className === 'valid') {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = false;
        } else {
          document.getElementById("jspsych-survey-multi-choice-next").disabled = true;
        }
      })

      // Detect submit button press
      document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault();

        // Measure response time
        var endTime = performance.now();
        var response_time = endTime - startTime;

        // Gather responses
        var question_data = {};
        // for (var i=0; i<4; i++) {
        for (let i of [0, 1, 3]){

          // Find matching question.
          var match = display_element.querySelector('#jspsych-survey-multi-choice-'+i);
          var name = match.attributes['data-name'].value;
          var val = match.querySelector("input[type=radio]:checked").value;

          // Store response
          var obje = {};
          obje[name] = val;
          Object.assign(question_data, obje);

        }

        // Save data
        var trial_data = {
          "rt": response_time,
          "responses": JSON.stringify(question_data),
          "errors": count
        };
        display_element.innerHTML += '';

        // next trial
        jsPsych.finishTrial(trial_data);
      });

      var startTime = performance.now();

    }
  }
  Gold1ComprehensionPluginBG.info = info;

  return Gold1ComprehensionPluginBG;

})(jsPsychModule);
