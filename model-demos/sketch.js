import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

let summarizer, generator, classifier;

window.addEventListener('load', () => {
  document.getElementById('summarize').addEventListener('click', summarize);
  document.getElementById('generate').addEventListener('click', generate);
  document.getElementById('classify').addEventListener('click', classify);
});

const statusElement = document.getElementById('status');
const outputElement = document.getElementById('output');

async function summarize() {
  let text = document.getElementById('inputText').value;
  if (!summarizer) {
    updateStatus('Loading summarization model...');
    summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
  }
  updateStatus('Summarizing...');
  let output = await summarizer(text, {
    max_new_tokens: 100,
  });
  console.log(output);
  updateOutput(output[0].summary_text);
}

async function generate() {
  let text = document.getElementById('inputText').value;
  if (!generator) {
    updateStatus('Loading generation model...');
    generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
  }
  updateStatus('Generating...');
  let output = await generator(text, {
    max_new_tokens: 100,
  });
  console.log(output);
  updateOutput(output[0]);
}

async function classify() {
  let text = document.getElementById('inputText').value;
  if (!classifier) {
    updateStatus('Loading classification model...');
    classifier = await pipeline('text-classification', 'Xenova/toxic-bert');
  }
  updateStatus('Classifying...');
  let results = await classifier(text, { topk: null });

  let output = '';
  for (let element of results) {
    output += `${element.label} : ${element.score}<br>`;
  }
  updateOutput(output);
}

function updateStatus(message) {
  statusElement.textContent = 'Status: ' + message;
}

function updateOutput(message) {
  outputElement.innerHTML = message;
  updateStatus('Ready');
}
