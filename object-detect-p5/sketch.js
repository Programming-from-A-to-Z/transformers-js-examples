import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

new p5((sketch) => {
  let img;
  let statusMsg = 'Loading model...';
  let detector;
  let detections = [];

  sketch.preload = () => {
    // Preload an image
    img = sketch.loadImage('dog_cat.jpg');
  };

  sketch.setup = async () => {
    // Setup the canvas
    sketch.createCanvas(640, 480);

    // Load the object detection model
    statusMsg = 'Loading model...';
    detector = await pipeline('object-detection', 'Xenova/detr-resnet-50');
    statusMsg = 'Model loaded. Detecting objects...';
    detect();
  };

  sketch.draw = () => {
    sketch.background(0);
    // Display the image
    sketch.image(img, 0, 0, sketch.width, sketch.height);

    // Display status message
    sketch.fill(255);
    sketch.noStroke();
    sketch.text(statusMsg, 10, 20);

    // If detections are available, display them
    if (detections.length > 0) {
      drawBoxes();
    }
  };

  async function detect() {
    // Detect objects in the image
    console.log;
    const output = await detector(img.canvas.toDataURL(), {
      threshold: 0.5,
      percentage: true,
    });
    detections = output;
    statusMsg = 'Detection complete.';
  }

  function drawBoxes() {
    // Draw bounding boxes and labels
    detections.forEach((d) => {
      const { box, label } = d;
      const { xmax, xmin, ymax, ymin } = box;

      // Generate a random color for the box
      const boxColor = sketch.color(0);
      // Draw the box
      sketch.stroke(boxColor);
      sketch.strokeWeight(2);
      sketch.noFill();
      sketch.rect(
        xmin * sketch.width,
        ymin * sketch.height,
        (xmax - xmin) * sketch.width,
        (ymax - ymin) * sketch.height
      );

      // Draw the label
      sketch.fill(boxColor);
      sketch.noStroke();
      sketch.textSize(16);
      sketch.text(label, xmin * sketch.width, ymin * sketch.height - 5);
    });
  }
}, document.body);
