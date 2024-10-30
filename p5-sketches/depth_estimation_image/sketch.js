// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

let depthEstimation;
let img;
let depthImg;

function preload() {
  // Load an image of a dog and cat before the sketch starts
  img = loadImage('dog_cat.jpg');
}

async function setup() {
  // Create a canvas for displaying the image and depth map
  createCanvas(640, 380);

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  // Initialize the depth estimation model with specified options
  depthEstimation = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
    device: 'webgpu',
  });

  // Create an image to hold the depth map, matching the original image size
  depthImg = createImage(img.width, img.height);

  // Estimate depth and display it on the canvas
  await estimateDepth();
  image(depthImg, 0, 0);
}

// Asynchronous function to estimate depth in the image
async function estimateDepth() {
  // Convert image to data URL and run depth estimation
  let results = await depthEstimation(img.canvas.toDataURL());

  // Extract the depth map from the results
  let { depth } = results;

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
}
