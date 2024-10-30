// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

// Asynchronous video capture and object detection setup
let video;
let objectDetector;
let results;

async function setup() {
  // Create canvas and initialize video capture
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();
  objectDetector = await pipeline("object-detection", "Xenova/detr-resnet-50", {
    device: "webgpu",
  });

  // Start object detection loop
  detectObjects();

  // Confirm model loading
  console.log("model loaded");
}

// Asynchronous function to detect objects in video frames
async function detectObjects() {
  // Convert video frame to data URL and run detection
  results = await objectDetector(video.canvas.toDataURL());

  // Continue detection loop
  detectObjects();
}

function draw() {
  // Display the video frame on canvas
  image(video, 0, 0);

  // If results exist, display detected objects
  if (results) {
    for (let i = 0; i < results.length; i++) {
      // Extract label, score, and bounding box using destructuring
      let { label, score, box } = results[i];
      let { xmin, ymin, xmax, ymax } = box;

      // Display label and score above bounding box
      fill(255);
      noStroke();
      textSize(16);
      text(label, xmin, ymin - 16);
      text(score, xmin, ymin);

      // Draw bounding box around detected object
      noFill();
      stroke(0);
      strokeWeight(2);
      rect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
  }
}
