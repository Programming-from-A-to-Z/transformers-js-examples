// Programming A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F24

// Function to load transformers.js dynamically
async function loadTransformers() {
  try {
    console.log('Loading transformers.js...');
    const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    const { pipeline } = module;
    console.log('Transformers.js loaded successfully.');
    return pipeline;
  } catch (error) {
    console.error('Failed to load transformers.js', error);
  }
}

// Function to get embeddings for a list of sentences using the transformers.js library
async function getEmbeddings(sentences) {
  let pipeline = await loadTransformers();
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
let cellSize;
let cols = sentences.length;
let rows = sentences.length;
let whichsentences;

async function setup() {
  createCanvas(400, 400);
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
  cellSize = width / sentences.length;
  background(0);
  whichsentences = createP('');
}

// Draw the similarity matrix and handle mouse interactions
function draw() {
  background(0);
  if (comparison) {
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        let colorValue = comparison[i][j] * 255;
        fill(colorValue);
        rect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }

    // Display sentences and similarity score based on mouse position
    let row = Math.floor(mouseY / cellSize);
    let col = Math.floor(mouseX / cellSize);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      whichsentences.html(
        `[${col}]: "${sentences[col]}"<br>[${row}]: "${sentences[row]}"<br>${comparison[col][row]}`
      );
    }
  }
}

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
