// Programming A to Z, Fall 2024
// https://github.com/Programming-from-A-to-Z/A2Z-F24

let conversationHistory = [];
let inputBox;
let chatLog = '';
let chatP;
let generator;

async function setup() {
  noCanvas();
  inputBox = createInput();
  inputBox.size(300);
  let sendButton = createButton('Send');
  sendButton.mousePressed(sendMessage);
  chatP = createP();
  conversationHistory.push({
    role: 'system',
    content: 'You are a helpful assistant.',
    // content:
    //   'You are a frog that only ever says ribbit. No matter what anyone else says you only say Ribbit.',
  });

  // Load the Transformers.js model pipeline
  let pipeline = await loadTransformers();
  generator = await pipeline('text-generation', 'onnx-community/Llama-3.2-1B-Instruct-q4f16', {
    dtype: 'q4f16',
    device: 'webgpu',
    progress_callback: (x) => {
      console.log(x);
    },
  });
}

async function sendMessage() {
  let userInput = inputBox.value();
  conversationHistory.push({ role: 'user', content: userInput });
  chatLog = `You: ${userInput}</br></br>` + chatLog;
  chatP.html(chatLog);

  if (generator) {
    try {
      // Generate a response based on the input prompt
      const output = await generator(conversationHistory, { max_new_tokens: 128 });
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/at
      const reply = output[0].generated_text.at(-1).content;
      conversationHistory.push({ role: 'assistant', content: reply });
      chatLog = `Chatbot: ${reply}</br></br>` + chatLog;
      chatP.html(chatLog);
    } catch (error) {
      console.error('Error communicating with Transformers.js:', error);
      chatLog += 'Error: Unable to communicate with the chatbot';
    }
  } else {
    console.log('Model not loaded yet, try again in a minute.');
  }
}
