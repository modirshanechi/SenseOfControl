/**
* Locus of control
*
* Hanna Levenson (1974) 
* Activism and Powerful Others: 
*   Distinctions within the Concept of Internal-External Control, 
* Journal of Personality Assessment, 38:4, 377-383
*   DOI: 10.1080/00223891.1974.10119988
*
**/

var loc74_1 = {
  type: jsPsychSurveyTemplate,
  items: [
    "Whether or not I get to be a leader depends mostly on my ability.", // 1
    "To a great extent my life is controlled by accidental happenings.", // 2
    "I feel like what happens in my life is mostly determined by powerful people.", // 3
    "I know how to read and write.", // attention check
    "Whether or not I get into a car accident depends mostly on how good a driver I am.", // 4
    "When I make plans, I am almost certain to  make them work.", // 5
    "Often there is no chance of protecting my personal interest from bad luck happenings.", // 6
    "When I get what I want, it's usually because I'm lucky.", // 7
    "Although I might have good ability, I will not be given leadership responsibility without appealing to those in positions of power.", // 8
    "How many friends I have depends on how nice a person I am.", // 9
    "I have often found that what is going to happen will happen.", // 10
    "My life is chiefly controlled by powerful others.", // 11
    "Whether or not I get into a car accident is mostly a matter of luck." // 12
  ],
  scale:[
    "Strongly<br>Disagree",
    "Disagree",
    "Somewhat<br>Disagree",
    "Somewhat<br>Agree",
    "Agree",
    "Strongly<br>Agree",
    "<strong>Skip</strong><br>the question",
  ],
  instructions: '<h2>Survey</h2>' +
                'For each of the statements please indicate your level of agreement or disagreement by clicking one of the scale categories.' +
                '<br><font color="#c87606">Your answers will not affect your payment or bonus.</font>',
  survey_width: 1100,
  item_width: 45,
  scale_repeat: 7,
  infrequency_items: [
    false, false, false, // 01-03
    true, // attention check
    false, false, false, // 04-05
    false, false, false, false, false, false  // 07-12
  ],
  randomize_question_order: false,
  if_skip_first: false,
  if_skip_last: true,
  data: {
    task: 'LOC1',
    IPC: [
      "I", "C", "P", // 01-03
      "Attention", // attention check
      "I", "I", "C", // 04-05
      "C", "P", "I", "C", "P", "C"  // 07-12
    ]
  },
}

var loc74_2 = {
  type: jsPsychSurveyTemplate,
  items: [
    "People like myself have very little chance of protecting our personal interests when they conflict with those of strong pressure groups.", // 13
    "It's not always wise for me to plan too far ahead because many things turn out to be a matter of good or bad fortune.", // 14
    "Getting what I want requires pleasing those people above me.", // 15
    "Whether or not I get to be a leader depends on whether I'm lucky enough to be in the right place at the right time.", // 16
    "If important people were to decide they didn't like me, I probably wouldn't make many friends.", // 17
    "I can pretty much determine what will happen in my life.", // 18
    "I am usually able to protect my personal interests.", // 19
    "Whether or not I get into a car accident depends mostly on the other driver.", // 20
    "I find myself living on Mars.", // attention check
    "When I get what I want, it's usually because I worked hard for it.", // 21
    "In order to have my plans work, I make sure that they fit in with the desires of people who have power over me.", // 22
    "My life is determined by my own actions.", // 23
    "It's chiefly a matter of fate whether or not I have a few friends or many friends." // 24
  ],
  scale:[
    "Strongly<br>Disagree",
    "Disagree",
    "Somewhat<br>Disagree",
    "Somewhat<br>Agree",
    "Agree",
    "Strongly<br>Agree",
    "<strong>Skip</strong><br>the question",
  ],
  instructions: '<h2>Survey</h2>' +
                'For each of the statements please indicate your level of agreement or disagreement by clicking one of the scale categories.' +
                '<br><font color="#c87606">Your answers will not affect your payment or bonus.</font>',
  survey_width: 1100,
  item_width: 45,
  scale_repeat: 7,
  infrequency_items: [
    false, false, false, false, false, false, // 13-18
    false, false, // 19-20
    true, // attention check
    false, false, false, false  // 21-24
  ],
  randomize_question_order: false,
  if_skip_first: false,
  if_skip_last: true,
  data: {
    task: 'LOC2',
    IPC: [
      "P", "C", "P", "C", "P", "I", // 13-18
      "I", "P", // 19-20
      "Attention", // attention check
      "I", "P", "I", "C"  // 21-24
    ]
  },
}
