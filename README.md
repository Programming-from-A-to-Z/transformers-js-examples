# Transformers.js Examples

## Overview

This repository contains several demonstrations of running machine learning NLP models directly in the browser using transformers.js.

### 1. Embeddings

Embeds sentences using `Xenova/all-MiniLM-L6-v2` and visualizes the results in two different contexts:

- **Comparison Matrix**: Visualizes the cosine similarity between different sentences using p5.js to render a matrix.
- **Clustering**: Plots sentence embeddings in a two-dimensional space with umap-js for dimensionality reduction.

- Feature Extraction (embeddings): `Xenova/all-MiniLM-L6-v2`

### 2. Language Models

A set of demos using language models to generate or complete text interactively in the browser.

- **Chatbot Demo**: Implements a simple chatbot using `onnx-community/Qwen2.5-0.5B-Instruct`.
- **Text Generation**: Generate text using various models including `onnx-community/Llama-3.2-1B-Instruct-q4f16` and `HuggingFaceTB/SmolLM-135M`.

### 3. Vision Models

- **Depth Estimation**: Estimates depth from an image using `onnx-community/depth-anything-v2-small` and visualizes it using p5.js.
- **Object Detection**: Detects objects in images and video in real-time using `Xenova/detr-resnet-50` with p5.js for visualization.

### 4. Whisper

Implements a real-time audio transcription system using `Xenova/whisper-tiny.en`.

### Example Code

```javascript
import { pipeline } from '@xenova/transformers';

// Allocate a pipeline for sentiment-analysis
let pipe = await pipeline('sentiment-analysis');

let out = await pipe('I love transformers!');
// [{'label': 'POSITIVE', 'score': 0.999817686}]
```

```javascript
import { pipeline } from '@xenova/transformers';

// Allocate a pipeline for feature extraction
let extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

let out = await extractor('Your sentence here.');
// Example output: Array of feature embeddings
```

## About Transformers.js

Transformers.js allows for the direct execution of 🤗 transformer models in the browser with no server required. It is designed to be functionally equivalent to the 🤗 [Python transformers library](https://github.com/huggingface/transformers), supporting a range of tasks in NLP, computer vision, audio, and multimodal applications.

- Run pretrained models for tasks such as text summarization, classification, and generation, image classification, and speech recognition.
- Functionally similar to Hugging Face's Python library with the pipeline API.

For further information and a comprehensive list of supported tasks and models, refer to the [Transformers.js Documentation](https://huggingface.co/docs/transformers.js/index).

The [Transformers.js Model Hub](https://huggingface.co/models?library=transformers.js) lists models compatible with the library, searchable by tasks.

## p5.js Integration and Usage

To integrate transformers.js with p5.js, instance mode can be used to enable compatibility with ES6 import statements required by transformers.js. However, dynamic import statements can also be used directly with p5.js as follows.

### Example of Loading Transformers.js

```javascript
// Function to load transformers.js dynamically
async function loadTransformers() {
  console.log('Loading transformers.js...');
  const module = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
  const { pipeline } = module;
  console.log('Transformers.js loaded successfully.');
  return pipeline;
}
```

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique used to visualize high-dimensional data, like sentence embeddings from transformers.js.

For [more about UMAP in JavaScript, here's the umap-js repo](https://github.com/PAIR-code/umap-js). This accompanying [Understanding UMAP](https://pair-code.github.io/understanding-umap/) article is also a fantastic read!
