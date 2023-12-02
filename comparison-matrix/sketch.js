// Importing pipeline from transformers.js for advanced text processing and feature extraction
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Function to get embeddings for a list of sentences using the transformers.js library
async function getEmbeddings(sentences) {
  // Initialize the text feature extraction pipeline
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  // Store embeddings for each sentence
  let embeddings = [];
  for (let sentence of sentences) {
    let output = await extractor(sentence, { pooling: 'mean', normalize: true });
    embeddings.push(output.data);
  }
  return embeddings;
}
// Store the similarity comparisons between sentences
let comparison;

// List of sentences to compare
const sentences = [
  'What color is the sky?',
  'What is an apple?',
  'The sky is blue.',
  'What does the fox say?',
  'An apple is a fruit.',
  'I have no idea.',
];

// p5.js sketch for visualizing the sentence similarities
function sketch(p) {
  let cellSize;
  let cols = sentences.length;
  let rows = sentences.length;
  let whichsentences;

  // Setup the canvas and HTML elements
  p.setup = async function () {
    p.createCanvas(400, 400);
    // Retrieve embeddings for the sentences
    const embeddings = await getEmbeddings(sentences);

    // Initialize and fill the comparison matrix with cosine similarity values
    comparison = [];
    for (let i = 0; i < sentences.length; i++) {
      comparison[i] = [];
      for (let j = 0; j < sentences.length; j++) {
        comparison[i][j] = cosineSimilarity(embeddings[i], embeddings[j]);
      }
    }
    console.log(comparison);
    cellSize = p.width / sentences.length;
    p.background(0);
    whichsentences = p.createP('');
  };

  // Draw the similarity matrix and handle mouse interactions
  p.draw = function () {
    p.background(0);
    if (comparison) {
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          let colorValue = comparison[i][j] * 255;
          p.fill(colorValue);
          p.rect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
      }

      // Display sentences and similarity score based on mouse position
      let row = Math.floor(p.mouseY / cellSize);
      let col = Math.floor(p.mouseX / cellSize);
      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        whichsentences.html(
          `[${col}]: "${sentences[col]}<br>[${row}]: "${sentences[row]}"<br>${comparison[col][row]}`
        );
      }
    }
  };
}

// Create a new p5.js instance with the sketch
new p5(sketch);

// Function to calculate dot product of two vectors
function dotProduct(vecA, vecB) {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}

// Function to calculate the magnitude of a vector
function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

// Function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}
