// Importing pipeline from transformers.js for text processing and feature extraction
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Dot class for representing and visualizing each data point
class Dot {
  constructor(x, y, sentence, embedding, sketch) {
    this.x = x;
    this.y = y;
    this.sentence = sentence;
    this.embedding = embedding;
    this.sketch = sketch;
    this.r = 4;
  }

  // Display the dot on canvas
  show() {
    this.sketch.fill(175);
    this.sketch.stroke(255);
    this.sketch.circle(this.x, this.y, this.r * 2);
  }

  // Display the associated text of the dot
  showText() {
    this.sketch.fill(255);
    this.sketch.noStroke();
    this.sketch.textSize(24);
    this.sketch.text(this.sentence, 10, this.sketch.height - 10);
  }

  // Check if a point (x, y) is over this dot
  over(x, y) {
    let d = this.sketch.dist(x, y, this.x, this.y);
    return d < this.r;
  }
}

// Instance mode in p5.js for encapsulating the sketch, enabling compatibility with external modules
new p5((sketch) => {
  let dots = [];
  let extractor;

  // Setup function to initialize the canvas and process data
  sketch.setup = async () => {
    sketch.createCanvas(800, 800);
    let raw = await fetch('p5.txt');
    let terms = await raw.text();
    let sentences = terms.split(/\n+/);
    let umap = new UMAP({ nNeighbors: 15, minDist: 0.1, nComponents: 2 });
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    let embeddings = await Promise.all(sentences.map(embedding));

    let umapResults = umap.fit(embeddings);
    // Mapping UMAP results to pixel space for visualization
    let [maxW, minW, maxH, minH] = mapUMAPToPixelSpace(umapResults, sketch);

    // Creating Dot objects from UMAP results and sentences
    for (let i = 0; i < umapResults.length; i++) {
      let x = sketch.map(umapResults[i][0], minW, maxW, 10, sketch.width - 10);
      let y = sketch.map(umapResults[i][1], minH, maxH, 10, sketch.height - 10);
      let dot = new Dot(x, y, sentences[i], embeddings[i], sketch);
      dots.push(dot);
    }
  };

  // Draw function to render dots on canvas
  sketch.draw = () => {
    if (dots.length > 0) {
      sketch.background(0);
      dots.forEach((dot) => dot.show());
      dots.forEach((dot) => {
        if (dot.over(sketch.mouseX, sketch.mouseY)) {
          dot.showText();
          return;
        }
      });
    }
  };

  // Function to extract embedding from a sentence using transformers.js
  async function embedding(sentence) {
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
});
