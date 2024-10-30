// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

let video;
let depthResult;
let depthEstimation;
let results;

async function setup() {
  // Create canvas and set up video capture with constraints
  createCanvas(640, 480);
  video = createCapture();
  video.size(320, 240);

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  // Initialize the depth estimation model
  depthEstimation = await pipeline(
    "depth-estimation",
    "onnx-community/depth-anything-v2-small",
    { device: "webgpu" }
  );

  // Start processing the video for depth estimation
  processVideo();
}

function draw() {
  // Draw the video on the canvas
  image(video, 0, 0);

  // If depth results are available, visualize them using pixel manipulation
  if (results) {
    const { depth } = results;

    // Create an image to store the depth visualization
    let depthImg = createImage(depth.width, depth.height);

    // Load pixels of the depth image for manipulation
    depthImg.loadPixels();

    // Loop through each row of the depth map
    for (let y = 0; y < depth.height; y++) {
      // Loop through each column of the depth map
      for (let x = 0; x < depth.width; x++) {
        // Calculate the 1D array index from 2D coordinates
        let index = x + y * depth.width;

        // Get the depth value for the current pixel
        let depthValue = depth.data[index];

        // Calculate the corresponding pixel index in the depth image
        let pixelIndex = index * 4;

        // Set the RGB values to the depth value for a grayscale effect
        depthImg.pixels[pixelIndex] = depthValue;
        depthImg.pixels[pixelIndex + 1] = depthValue;
        depthImg.pixels[pixelIndex + 2] = depthValue;

        // Set the alpha value to fully opaque
        depthImg.pixels[pixelIndex + 3] = 255;
      }
    }

    // Update the pixels of the depth image
    depthImg.updatePixels();

    // Draw the depth image on the canvas
    image(depthImg, 0, 0, width, height);
  }
}

// Asynchronous function to continuously process video frames
async function processVideo() {
  // Convert video frame to data URL and run depth estimation
  results = await depthEstimation(video.canvas.toDataURL());

  // Recursively call processVideo() to keep processing frames
  processVideo();
}
