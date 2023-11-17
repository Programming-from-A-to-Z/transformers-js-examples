import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

async function getEmbeddings(sentences) {
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  let embeddings = [];
  for (let sentence of sentences) {
    let output = await extractor(sentence, { pooling: 'mean', normalize: true });
    embeddings.push(output.data);
  }
  return embeddings;
}

// Cosine Similarity Functions
function dotProduct(vecA, vecB) {
  let dot = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
  if (!dot) {
    console.log(vecA.length);
    console.log(vecB.length);
    console.log(vecA);
    console.log(vecB);
    console.log(dot);
  }
  return dot;
}

function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

function cosineSimilarity(vecA, vecB) {
  return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}

let comparison;

const sentences = [
  'The train goes choo choo!',
  'Good evening!',
  'The cat meows',
  'Good morning!',
  'The dog barks',
];

document.addEventListener('DOMContentLoaded', async () => {
  const embeddings = await getEmbeddings(sentences);
  comparison = [];
  for (let i = 0; i < sentences.length; i++) {
    comparison[i] = [];
    for (let j = 0; j < sentences.length; j++) {
      comparison[i][j] = cosineSimilarity(embeddings[i], embeddings[j]);
    }
  }
  console.log(comparison);
});

// This function sets up the p5.js instance
function sketch(p) {
  let cellSize;
  let cols = sentences.length;
  let rows = sentences.length;
  let whichsentences;

  p.setup = function () {
    p.createCanvas(400, 400);
    cellSize = p.width / sentences.length;
    p.background(0);
    whichsentences = p.createP('');
  };

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

new p5(sketch);
