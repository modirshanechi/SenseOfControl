//----------------------------------------------------------------------------//
// general functions
//----------------------------------------------------------------------------//
function areArraysEqual(arr1, arr2) {
    return arr1.length === arr2.length && arr1.every((value, index) => value === arr2[index]);
}

//----------------------------------------------------------------------------//
// base information
//----------------------------------------------------------------------------//
const topPos0 = 50;
const leftPos0 = 50;
const TPos0    = [0,0,0, 0,1,0, 0,0,0];
const roomdim  = 9;

var agent_topPos = topPos0;
var agent_leftPos = leftPos0;
var agent_TPos = TPos0;

var goal_topPos = topPos0;
var goal_leftPos = leftPos0;
var goal_TPos = TPos0;


function initialize_agent() {
    return `
    <div class="container">
        <img src="img/BaseAction.png" alt="Base Action Image" class="base-image">
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image">    
    </div>`;
}

function initialize_agent_selection(agentid) {
    return `<div class="selection-container-small">
        <img src="img/BaseAction.png" alt="Base Action Image" class="base-image">
        <img id="agent` + agentid + `" src="img/Agent.png" alt="Agent Image" class="agent-image">    
    </div>`;
}

function initialize_agentgoal() {
    return `
    <div class="container">
        <img src="img/BaseAction.png" alt="Base Action Image" class="base-image">
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image"> 
        <img id="goal" src="img/Goal.png" alt="Goal Image" class="agent-image"> 
    </div>`;
}

//----------------------------------------------------------------------------//
// movement
//----------------------------------------------------------------------------//
const gridSize = 3;
const step = 100 / gridSize;
function move(options) {
    const availableMoves = [];
    options.forEach((val, index) => {
        if (val === 1) availableMoves.push(index);
    });
    const randomMove = availableMoves[Math.floor(Math.random() * availableMoves.length)];
    agent_TPos = [0,0,0, 0,0,0, 0,0,0]
    agent_TPos[randomMove] = 1;

    if (randomMove < 3 && agent_topPos > 16.67) agent_topPos -= step;
    if (randomMove >= 6 && agent_topPos < 83.33) agent_topPos += step;
    if (randomMove % 3 === 0 && agent_leftPos > 16.67) agent_leftPos -= step;
    if (randomMove % 3 === 2 && agent_leftPos < 83.33) agent_leftPos += step;
}

function move_practice(options,i_practice) {
    const availableMoves = [];
    options.forEach((val, index) => {
        if (val === 1) availableMoves.push(index);
    });
    const randomMove = availableMoves[(i_practice % availableMoves.length)];
    agent_TPos = [0,0,0, 0,0,0, 0,0,0]
    agent_TPos[randomMove] = 1;

    if (randomMove < 3 && agent_topPos > 16.67) agent_topPos -= step;
    if (randomMove >= 6 && agent_topPos < 83.33) agent_topPos += step;
    if (randomMove % 3 === 0 && agent_leftPos > 16.67) agent_leftPos -= step;
    if (randomMove % 3 === 2 && agent_leftPos < 83.33) agent_leftPos += step;
}

function updateagent(){
    var agent = document.getElementById('agent');
    agent.style.top  = agent_topPos + '%';
    agent.style.left = agent_leftPos + '%'; 
}

//----------------------------------------------------------------------------//
// Goal 
//----------------------------------------------------------------------------//
function sample_goal() {
    const availableMoves = [];
    options = [1,1,1, 1,0,1, 1,1,1]
    options.forEach((val, index) => {
        if (val === 1) availableMoves.push(index);
    });
    const randomMove = availableMoves[Math.floor(Math.random() * availableMoves.length)];
    goal_TPos = [0,0,0, 0,0,0, 0,0,0]
    goal_TPos[randomMove] = 1;

    if (randomMove < 3 && goal_topPos > 16.67) goal_topPos -= step;
    if (randomMove >= 6 && goal_topPos < 83.33) goal_topPos += step;
    if (randomMove % 3 === 0 && goal_leftPos > 16.67) goal_leftPos -= step;
    if (randomMove % 3 === 2 && goal_leftPos < 83.33) goal_leftPos += step;
}
function sample_goal_practice(i_practice) {
    const availableMoves = [];
    options = [1,1,1, 1,0,1, 1,1,1]
    options.forEach((val, index) => {
        if (val === 1) availableMoves.push(index);
    });
    const randomMove = availableMoves[(i_practice % availableMoves.length)];
    goal_TPos = [0,0,0, 0,0,0, 0,0,0]
    goal_TPos[randomMove] = 1;

    if (randomMove < 3 && goal_topPos > 16.67) goal_topPos -= step;
    if (randomMove >= 6 && goal_topPos < 83.33) goal_topPos += step;
    if (randomMove % 3 === 0 && goal_leftPos > 16.67) goal_leftPos -= step;
    if (randomMove % 3 === 2 && goal_leftPos < 83.33) goal_leftPos += step;
}

function updategoal(){
    var goal = document.getElementById('goal');
    goal.style.top  = goal_topPos + '%';
    goal.style.left = goal_leftPos + '%'; 
}

//----------------------------------------------------------------------------//
// sample bonus for a selected room
//----------------------------------------------------------------------------//
function bonus_probability(room_index){
    let Room = allrooms[room_index];
    let keys = Object.keys(Room);
    let dimension = roomdim;

    // Step 1: Normalize each 9-D vector by dividing by the sum of its elements
    let normalizedVectors = {};
    keys.forEach(key => {
        let vector = Room[key];
        let sum = vector.reduce((acc, val) => acc + val, 0);
        normalizedVectors[key] = sum === 0 ? vector : vector.map(val => val / sum);
    });

    // Step 2: Compute element-wise maximum over all normalized vectors
    let maxVector = Array(dimension).fill(0);
    keys.forEach(key => {
        normalizedVectors[key].forEach((val, index) => {
            maxVector[index] = Math.max(maxVector[index], val);
        });
    });

    // Step 3: Compute probability
    let sumMaxVector = maxVector.reduce((acc, val) => acc + val, 0);
    let probability = sumMaxVector / (dimension-1); // Value between 0 and 1

    return probability;
}

function sample_bonus(room_index){
    let probability = bonus_probability(room_index);
    let bernoulliSample = Math.random() < probability ? 1 : 0;

    return bernoulliSample;
}

//----------------------------------------------------------------------------//
// action visualization
//----------------------------------------------------------------------------//
function showactions(room_index, rowid) {
    Room = allrooms[room_index]
    n = Object.keys(Room).length
    
    const row = document.getElementById('action_row' + rowid);
    const containerHeight = 75
    const containerWidth  = 75
    const totalWidth = row.gap * (n - 1) + containerHeight * n;
    // const containerHeight = containerWidth / aspectRatio; // Calculate height based on width and ratio

    // Set the row's fixed width
    row.style.width = `${totalWidth}px`;
    
    Object.entries(Room).forEach(([action, T]) => {
        // Create a container div
        const container = document.createElement('div');
        container.className = 'container-row';

        // Set the dynamically calculated width and height
        container.style.width  = `${containerWidth}px`;
        container.style.height = `${containerHeight}px`;

        // Create the base image
        const baseImage = document.createElement('img');
        baseImage.src = 'img/BaseAction.png';
        baseImage.alt = 'Base Action Image';
        baseImage.className = 'base-image';
        container.appendChild(baseImage);
        
        // Create the target images
        for (let j = 0; j < gridSize * gridSize; j++) {
            if (T[j] == 1) {
                const targetImage = document.createElement('img');
                targetImage.src = 'img/Target.png'; // Path to the target image
                targetImage.alt = 'Target Image';
                targetImage.className = 'target-image';
    
                // Calculate position in the grid (row and column)
                const row = Math.floor(j / gridSize);
                const col = j % gridSize;
    
                // Position the target image
                targetImage.style.position = 'absolute';
                targetImage.style.width = `${(100 / gridSize) * 0.8}%`;
                targetImage.style.height = `${(100 / gridSize) * 0.8}%`;
                targetImage.style.left = `${(col * 100) / gridSize + (100 / gridSize) * 0.1}%`;
                targetImage.style.top = `${(row * 100) / gridSize + (100 / gridSize) * 0.1}%`;

    
                // Append the target image to the base image
                container.appendChild(targetImage);
            }
        }

        // Create the agent image
        const agentImage = document.createElement('img');
        agentImage.src = 'img/Agent.png';
        agentImage.alt = 'Agent Image';
        agentImage.className = 'agent-image';
        container.appendChild(agentImage);

        // Create a label for the index
        const label = document.createElement('div');
        label.className = 'image-label';
        label.textContent = `Action ${action}`; // Set the index text
        container.appendChild(label);

        // Append the container to the row
        row.appendChild(container);
    });
}

function selection_showactions(room_index, rowid) {
    Room = allrooms[room_index]
    n = Object.keys(Room).length
    
    const row = document.getElementById('action_row' + rowid);
    if(n < 5){
        var containerHeight = 75
        var containerWidth  = 75
    }else{
        var containerHeight = 50
        var containerWidth  = 50
    }
    
    const gap = 10
    const totalWidth = gap * (n - 1) + containerHeight * n;

    // Set the row's fixed width
    row.style.width = `${totalWidth}px`;
    
    Object.entries(Room).forEach(([action, T]) => {
        // Create a container div
        const container = document.createElement('div');
        container.className = 'container-row';

        // Set the dynamically calculated width and height
        container.style.width  = `${containerWidth}px`;
        container.style.height = `${containerHeight}px`;

        // Create the base image
        const baseImage = document.createElement('img');
        baseImage.src = 'img/BaseAction.png';
        baseImage.alt = 'Base Action Image';
        baseImage.className = 'base-image';
        container.appendChild(baseImage);
        
        // Create the target images
        for (let j = 0; j < gridSize * gridSize; j++) {
            if (T[j] == 1) {
                const targetImage = document.createElement('img');
                targetImage.src = 'img/Target.png'; // Path to the target image
                targetImage.alt = 'Target Image';
                targetImage.className = 'target-image';
    
                // Calculate position in the grid (row and column)
                const row = Math.floor(j / gridSize);
                const col = j % gridSize;
    
                // Position the target image
                targetImage.style.position = 'absolute';
                targetImage.style.width = `${(100 / gridSize) * 0.8}%`; 
                targetImage.style.height = `${(100 / gridSize) * 0.8}%`;
                targetImage.style.left = `${(col * 100) / gridSize + (100 / gridSize) * 0.1}%`; // Centered horizontally
                targetImage.style.top = `${(row * 100) / gridSize + (100 / gridSize) * 0.1}%`; // Centered vertically

    
                // Append the target image to the base image
                container.appendChild(targetImage);
            }
        }

        // Create the agent image
        const agentImage = document.createElement('img');
        agentImage.src = 'img/Agent.png';
        agentImage.alt = 'Agent Image';
        agentImage.className = 'agent-image';
        container.appendChild(agentImage);

        // Create a label for the index
        const label = document.createElement('div');
        label.className = 'image-label';
        label.textContent = `Action ${action}`; // Set the index text
        container.appendChild(label);
        if(n > 4){
            label.style.fontSize = '0.5em';
        }

        // Append the container to the row
        row.appendChild(container);
    });
}
