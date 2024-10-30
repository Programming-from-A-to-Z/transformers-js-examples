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

// Function to load transformers.js dynamically
async function loadTransformers() {
  console.log('Loading transformers.js...');
  const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
  const { pipeline } = module;
  console.log('Transformers.js loaded successfully.');
  return pipeline;
}

// Dot class for representing and visualizing each data point
class Dot {
  constructor(x, y, sentence, embedding) {
    this.x = x;
    this.y = y;
    this.sentence = sentence;
    this.embedding = embedding;
    this.r = 4;
  }

  // Display the dot on canvas
  show() {
    fill(175);
    stroke(255);
    circle(this.x, this.y, this.r * 2);
  }

  // Display the associated text of the dot
  showText() {
    fill(255);
    noStroke();
    textSize(24);
    text(this.sentence, 10, height - 10);
  }

  // Check if a point (x, y) is over this dot
  over(x, y) {
    let d = dist(x, y, this.x, this.y);
    return d < this.r;
  }
}

let dots = [];
let extractor;

async function setup() {
  createCanvas(800, 800);
  console.log('Fetching sentences from p5.txt...');
  let raw = await fetch('p5.txt');
  let terms = await raw.text();
  let sentences = terms.split(/\n+/);
  console.log('Sentences loaded:', sentences);

  let umap = new UMAP({ nNeighbors: 15, minDist: 0.1, nComponents: 2 });
  let pipeline = await loadTransformers();
  extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  console.log('Extracting embeddings for sentences one by one...');

  let embeddings = [];
  for (let sentence of sentences) {
    let embeddingResult = await embedding(sentence);
    embeddings.push(embeddingResult);
  }

  let umapResults = umap.fit(embeddings);
  console.log('UMAP results:', umapResults);

  // Mapping UMAP results to pixel space for visualization
  let [maxW, minW, maxH, minH] = mapUMAPToPixelSpace(umapResults);

  // Creating Dot objects from UMAP results and sentences
  for (let i = 0; i < umapResults.length; i++) {
    let x = map(umapResults[i][0], minW, maxW, 10, width - 10);
    let y = map(umapResults[i][1], minH, maxH, 10, height - 10);
    let dot = new Dot(x, y, sentences[i], embeddings[i]);
    dots.push(dot);
  }
  console.log('Dots created:', dots);
}

function draw() {
  if (dots.length > 0) {
    background(0);
    for (let dot of dots) {
      dot.show();
    }
    for (let dot of dots) {
      if (dot.over(mouseX, mouseY)) {
        dot.showText();
        return;
      }
    }
  }
}

// Function to extract embedding from a sentence using transformers.js
async function embedding(sentence) {
  console.log('Extracting embedding for sentence:', sentence);
  let output = await extractor(sentence, { pooling: 'mean', normalize: true });
  return output.data;
}

// Function to map UMAP results to pixel space for visualization
function mapUMAPToPixelSpace(umapResults) {
  // Initialize variables to track the maximum and minimum values in width and height
  let maxW = 0;
  let minW = Infinity;
  let maxH = 0;
  let minH = Infinity;

  // Iterate over each UMAP result to find the extreme values
  for (let i = 0; i < umapResults.length; i++) {
    // Update maxW and minW with the maximum and minimum x-coordinates
    maxW = Math.max(maxW, umapResults[i][0]);
    minW = Math.min(minW, umapResults[i][0]);

    // Update maxH and minH with the maximum and minimum y-coordinates
    maxH = Math.max(maxH, umapResults[i][1]);
    minH = Math.min(minH, umapResults[i][1]);
  }

  // Return the extreme values which define the bounding box for UMAP results
  return [maxW, minW, maxH, minH];
}
