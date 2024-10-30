// Programming A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F24

let generator;
let inputText;
let outputText;

async function setup() {
  // Create a canvas and text input field
  createCanvas(400, 200);
  inputText = createInput('Type a prompt here...');

  // Load the Transformers.js model pipeline with async/await
  let pipeline = await loadTransformers();

  // Try
  // https://huggingface.co/HuggingFaceTB/SmolLM-135M
  // https://huggingface.co/HuggingFaceTB/SmolLM-360M

  // Create a text generation pipeline with specific model and options
  generator = await pipeline('text-generation', 'HuggingFaceTB/SmolLM-135M', {
    dtype: 'q4',
    device: 'webgpu',
    progress_callback: (x) => {
      console.log(x);
    },
  });

  // Create a button after model is loaded
  let button = createButton('Generate Text');
  button.mousePressed(generateText);
}

// Asynchronous function to generate text based on user input
async function generateText() {
  // Ensure the model is loaded
  if (generator) {
    // Complete the user's text
    const output = await generator(inputText.value(), { max_new_tokens: 128 });
    console.log(output);
    // Extract and display the generated text
    let outputText = output[0].generated_text;
    background(240);
    text(outputText, 10, 10, width - 20, height - 20);
  } else {
    // Log a message if the model is not yet loaded
    console.log('Model not loaded yet, try again in a minute.');
  }
}
