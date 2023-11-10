// Import the pipeline function from the transformers library via CDN link
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Declare variables to hold the instances of the summarizer, generator, and classifier
let summarizer, generator, classifier;

// Add an event listener to the window object that will execute the following function once the window is fully loaded
window.addEventListener('load', () => {
  // Add click event listeners to buttons with specific IDs, binding them to their respective functions
  document.getElementById('summarize').addEventListener('click', summarize);
  document.getElementById('generate').addEventListener('click', generate);
  document.getElementById('classify').addEventListener('click', classify);
});

// Grab DOM elements for updating status and output messages to the user
const statusElement = document.getElementById('status');
const outputElement = document.getElementById('output');

// Define an asynchronous function to summarize text
async function summarize() {
  // Retrieve the input text from the DOM
  let text = document.getElementById('inputText').value;
  // If the summarizer model has not been loaded yet, load it and update the status
  if (!summarizer) {
    updateStatus('Loading summarization model...');
    summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
  }
  // Update status and run the summarizer model on the input text
  updateStatus('Summarizing...');
  let output = await summarizer(text, {
    max_new_tokens: 100, // Set the maximum number of new tokens for the summary
  });
  // Log and display the summarization results
  console.log(output);
  updateOutput(output[0].summary_text);
}

// Define an asynchronous function to generate text
async function generate() {
  // Retrieve the input text from the DOM
  let text = document.getElementById('inputText').value;
  // If the generator model has not been loaded yet, load it and update the status
  if (!generator) {
    updateStatus('Loading generation model...');
    generator = await pipeline(
      'text2text-generation',
      'Xenova/LaMini-Flan-T5-783M'
    );
  }
  // Update status and run the generator model on the input text
  updateStatus('Generating...');
  let output = await generator(text, {
    max_new_tokens: 100, // Set the maximum number of new tokens for the generated text
  });
  // Log and display the generation results
  console.log(output);
  updateOutput(output[0]);
}

// Define an asynchronous function to classify text
async function classify() {
  // Retrieve the input text from the DOM
  let text = document.getElementById('inputText').value;
  // If the classifier model has not been loaded yet, load it and update the status
  if (!classifier) {
    updateStatus('Loading classification model...');
    classifier = await pipeline('text-classification', 'Xenova/toxic-bert');
  }
  // Update status and run the classifier model on the input text
  updateStatus('Classifying...');
  let results = await classifier(text, { topk: null });

  // Construct a string to display each classification result with its corresponding score
  let output = '';
  for (let element of results) {
    output += `${element.label} : ${element.score}<br>`;
  }
  // Update the DOM with the classification results
  updateOutput(output);
}

// Function to update the status text in the DOM
function updateStatus(message) {
  statusElement.textContent = 'Status: ' + message;
}

// Function to update the output text in the DOM and set the status back to 'Ready' once done
function updateOutput(message) {
  outputElement.innerHTML = message;
  updateStatus('Ready');
}
