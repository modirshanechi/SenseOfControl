/**
* Intolerance of Uncertainty Questionnaire (IUS-12)
*
* Carleton, R. N., Norton, M. P. J., & Asmundson, G. J. (2007). Fearing the unknown:
* A short version of the Intolerance of Uncertainty Scale. Journal of anxiety disorders,
* 21(1), 105-117.
*
* Bottesi et al. (2020). A short-form version of the Intolerance of Uncertainty Scale:
* Initial development of the IUS-5. 10.31234/osf.io/b62wf
*
**/

var ius12 = {
  type: jsPsychSurveyTemplate,
  items: [
    "Unforeseen events upset me greatly.", // 1
    "It frustrates me not having all the information I need.", // 2
    "Uncertainty keeps me from living a full life.", // 3
    "One should always look ahead so as to avoid surprises.", // 4
    "A small unforeseen event can spoil everything, even with the best of planning.", // 5
    "When it's time to act, uncertainty paralyses me.", // 6
    "When I am uncertain I can't function very well.", // 7
    "I always answer these survey questions honestly.", // attention check
    "I always want to know what the future has in store for me.", // 8
    "I can't stand being taken by surprise.", // 9
    "The smallest doubt can stop me from acting.", // 10
    "I should be able to organize everything in advance.", // 11
    "I must get away from all uncertain situations." // 12
  ],
  scale:[
    "Not at all<br>characteristic<br>of me",
    "A little<br>characteristic<br>of me",
    "Somewhat<br>characteristic<br>of me",
    "Very<br>characteristic<br>of me",
    "Entirely<br>characteristic<br>of me",
    "<strong>Skip</strong><br>the question",
  ],
  instructions: '<h2>Survey</h2>' +
                'Read each statement carefully and select which best describes you.' + 
                '<br><font color="#c87606">Your answers will not affect your payment or bonus.</font>',
  survey_width: 1100,
  item_width: 30,
  scale_repeat: 7,
  infrequency_items: [
    false, false, false, false, false, false, // 1-6
    false, // 7-7
    true, // attention check
    false, false, false, false, false // 8:12
  ],
  randomize_question_order: false,
  if_skip_first: false,
  if_skip_last: true,
  data: {
    task: 'IUS',
  },
}
