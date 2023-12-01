import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

class Dot {
  constructor(x, y, sentence, embedding, sketch) {
    this.x = x;
    this.y = y;
    this.sentence = sentence;
    this.embedding = embedding;
    this.sketch = sketch;
    this.r = 4;
  }

  show() {
    this.sketch.fill(175);
    this.sketch.stroke(255);
    this.sketch.circle(this.x, this.y, this.r * 2);
  }

  showText() {
    this.sketch.fill(255);
    this.sketch.noStroke();
    this.sketch.textSize(24);
    this.sketch.text(this.sentence, 10, this.sketch.height - 10);
  }

  over(x, y) {
    let d = this.sketch.dist(x, y, this.x, this.y);
    return d < this.r;
  }
}

new p5((sketch) => {
  let dots = [];
  let extractor;

  sketch.setup = async () => {
    sketch.createCanvas(800, 800);
    let raw = await fetch('p5.txt');
    let terms = await raw.text();
    let sentences = terms.split(/\n+/);
    let umap = new UMAP({ nNeighbors: 15, minDist: 0.1, nComponents: 2 });
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    let embeddings = await Promise.all(sentences.map(embedding));
    let umapResults = umap.fit(embeddings);

    let maxW = 0;
    let minW = Infinity;
    let maxH = 0;
    let minH = Infinity;
    for (let i = 0; i < umapResults.length; i++) {
      maxW = sketch.max(maxW, umapResults[i][0]);
      minW = sketch.min(minW, umapResults[i][0]);
      maxH = sketch.max(maxH, umapResults[i][1]);
      minH = sketch.min(minH, umapResults[i][1]);
    }

    for (let i = 0; i < umapResults.length; i++) {
      let x = sketch.map(umapResults[i][0], minW, maxW, 10, sketch.width - 10);
      let y = sketch.map(umapResults[i][1], minH, maxH, 10, sketch.height - 10);
      let dot = new Dot(x, y, sentences[i], embeddings[i], sketch);
      dots.push(dot);
    }
  };

  sketch.draw = () => {
    if (dots.length > 0) {
      sketch.background(0);
      for (let i = 0; i < dots.length; i++) {
        dots[i].show();
      }
      for (let i = 0; i < dots.length; i++) {
        if (dots[i].over(sketch.mouseX, sketch.mouseY)) {
          dots[i].showText();
          break;
        }
      }
    }
  };

  async function embedding(sentence) {
    let output = await extractor(sentence, { pooling: 'mean', normalize: true });
    return output.data;
  }
});
