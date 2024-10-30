// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

let sentimentModel;
let userInput;

async function setup() {
  // Create a canvas and a text input field
  createCanvas(200, 200);
  userInput = createElement('textArea', 'Write your text here.');

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  // Initialize the sentiment analysis model
  sentimentModel = await pipeline('sentiment-analysis');

  // Create a button to trigger sentiment analysis (after model is loaded)
  let button = createButton('analyze');
  button.mousePressed(analyzeText);
}

// Asynchronous function to analyze user input text
async function analyzeText() {
  // Analyze the sentiment of the input text
  let results = await sentimentModel(userInput.value());
  console.log(results);

  // Extract label and score using destructuring
  let { label, score } = results[0];

  // Set background color based on sentiment label and confidence score
  if (label == 'POSITIVE') {
    background(0, 255 * score, 0);
  } else {
    background(255 * score, 0, 0);
  }
}
