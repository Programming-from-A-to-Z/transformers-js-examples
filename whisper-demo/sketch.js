// Programming A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F24

// Import the Transformers.js pipeline for speech recognition
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let transcriptDiv = document.getElementById('transcript');
let recordButton = document.getElementById('recordButton');

// Function to downsample audio buffer to 16000Hz for Whisper model
function downsampleAudioBuffer(buffer, targetSampleRate) {
  const sampleRate = buffer.sampleRate;

  // If sample rate matches target, return original buffer
  if (sampleRate === targetSampleRate) {
    return buffer;
  }

  // Calculate downsample ratio and new buffer length
  const ratio = sampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const newBuffer = new Float32Array(newLength);

  // Populate new buffer with downsampled audio data
  for (let i = 0; i < newLength; i++) {
    newBuffer[i] = buffer.getChannelData(0)[Math.round(i * ratio)];
  }
  return newBuffer;
}

// Asynchronous function to transcribe audio using Whisper
async function transcribeAudio(blob) {
  // Show transcribing status
  transcriptDiv.textContent = 'transcribing...';

  // Load Whisper model from Transformers.js
  const model = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');

  // Convert Blob to ArrayBuffer for audio decoding
  const arrayBuffer = await blob.arrayBuffer();

  // Decode ArrayBuffer to audio data using Web Audio API
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Downsample the audio to 16000Hz as required by the model
  const downsampledAudio = downsampleAudioBuffer(audioBuffer, 16000);

  // Perform transcription with Whisper model
  const result = await model(downsampledAudio);

  // Display transcription result in the DOM
  transcriptDiv.textContent = result.text;
}

// Function to start audio recording
function startRecording() {
  // Request microphone access from the user
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      // Initialize MediaRecorder with the audio stream
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();
      isRecording = true;
      recordButton.textContent = 'stop recording';

      // Collect audio data while recording
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      // Stop recording and start transcription
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        transcribeAudio(audioBlob);
        audioChunks = [];
      };
    })
    .catch((error) => {
      // Handle errors when accessing the microphone
      console.error('Error accessing microphone: ', error);
    });
}

// Function to stop audio recording
function stopRecording() {
  mediaRecorder.stop();
  isRecording = false;
  recordButton.textContent = 'start recording';
}

// Event listener to toggle recording state
recordButton.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});
