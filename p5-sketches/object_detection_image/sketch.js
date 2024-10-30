// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

let img;
let objectDetection;

async function preload() {
  // Load the image of a dog and cat before the sketch starts
  img = loadImage('dog_cat.jpg');
}

async function setup() {
  // Create a canvas and display the loaded image
  createCanvas(640, 480);
  image(img, 0, 0);

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  objectDetection = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
    device: 'webgpu',
  });

  // Start object detection
  detectObjects();
}

// Asynchronous function to detect objects in the image
async function detectObjects() {
  // Convert image to data URL and run detection
  const results = await objectDetection(img.canvas.toDataURL());

  // Log results to the console for inspection
  console.log(results);

  // Loop through detected objects and display them
  for (const result of results) {
    // Extract label, bounding box, and score using destructuring
    const { label, box, score } = result;
    const { xmin, ymin, xmax, ymax } = box;

    // Draw bounding box around detected object
    stroke(255, 0, 255);
    fill(255, 0, 255, 50);
    rectMode(CORNERS);
    rect(xmin, ymin, xmax, ymax);

    // Display label and formatted score above the bounding box
    noStroke();
    fill(255);
    textSize(12);
    text(`${label} (${nf(score, 1, 4)})`, xmin, ymin - 5);
  }
}
