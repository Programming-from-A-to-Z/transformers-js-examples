// Introduction to Machine Learning for the Arts
// https://github.com/ml5js/Intro-ML-Arts-IMA-F24

let generator;
let inputText;
let outputText;

async function setup() {
  // Create a canvas and text input field
  createCanvas(400, 200);
  inputText = createInput('Type a prompt here...');

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  // Create a text generation pipeline with specific model and options
  generator = await pipeline('text-generation', 'onnx-community/Llama-3.2-1B-Instruct-q4f16', {
    dtype: 'q4',
    device: 'webgpu',
  });

  // Create a button after model is loaded
  let button = createButton('Generate Text');
  button.mousePressed(generateText);
}

// Asynchronous function to generate text based on user input
async function generateText() {
  // Ensure the model is loaded
  if (generator) {
    // Define the prompt structure for the text generation model
    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: inputText.value() },
    ];

    // Generate a response based on the input prompt
    const output = await generator(messages, { max_new_tokens: 128 });

    // Extract and display the generated text
    let outputText = output[0].generated_text.at(-1).content;
    background(240);
    text(outputText, 10, 10, width - 20, height - 20);
  } else {
    // Log a message if the model is not yet loaded
    console.log('Model not loaded yet, try again in a minute.');
  }
}
