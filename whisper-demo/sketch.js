import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let transcriptDiv = document.getElementById('transcript');
let recordButton = document.getElementById('recordButton');

// Downsample audio to 16000Hz
function downsampleAudioBuffer(buffer, targetSampleRate) {
  const sampleRate = buffer.sampleRate;
  if (sampleRate === targetSampleRate) {
    return buffer;
  }
  const ratio = sampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const newBuffer = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    newBuffer[i] = buffer.getChannelData(0)[Math.round(i * ratio)];
  }
  return newBuffer;
}

async function transcribeAudio(blob) {
  transcriptDiv.textContent = 'transcribing...';
  const model = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
  // Convert Blob to ArrayBuffer
  const arrayBuffer = await blob.arrayBuffer();
  // Use the Web Audio API to decode the ArrayBuffer into audio data
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Downsample audio to 16000Hz, as required by Whisper
  const downsampledAudio = downsampleAudioBuffer(audioBuffer, 16000);

  // Perform transcription with Whisper
  const result = await model(downsampledAudio);

  // Display the transcription result
  transcriptDiv.textContent = result.text;
}

function startRecording() {
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();
      isRecording = true;
      recordButton.textContent = 'stop recording';
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      // Stop recording and transcribe the audio
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        transcribeAudio(audioBlob);
        audioChunks = [];
      };
    })
    .catch((error) => {
      console.error('Error accessing microphone: ', error);
    });
}

function stopRecording() {
  mediaRecorder.stop();
  isRecording = false;
  recordButton.textContent = 'start recording';
}

recordButton.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});
