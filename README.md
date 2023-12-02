# Transformers.js Examples

## Overview

Single page demonstration on running machine learning NLP models in the browser with transformers.js.

### 1. Comparison Matrix

Visualizes the cosine similarity between sentences and p5.js for rendering a matrix.

- Feature Extraction (embeddings): Xenova/all-MiniLM-L6-v2

### 2. Clustering

Embeds sentences using `Xenova/all-MiniLM-L6-v2` and plots them in a two-dimensional space with umap-js dimensionality reduction.

- Feature Extraction (embeddings): Xenova/all-MiniLM-L6-v2

### 3. Other Model Demos

A few demos using other models from transformers.js.

- Summarization: Xenova/distilbart-cnn-6-6
- Text Generation: Xenova/LaMini-Flan-T5-783M
- Text Classification: Xenova/toxic-bert

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

## p5.js Integration and Instance Mode

To integrate transformers.js with p5.js, instance mode is required to enable compatibility with ES6 import statements required by transformers.js.

[Learn more about p5.js instance mode in this wiki](https://github.com/processing/p5.js/wiki/Global-and-instance-mode) or [instance mode video tutorial](https://youtu.be/Su792jEauZg).

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique used to visualize high-dimensional data, like sentence embeddings from transformers.js.

For [more about UMAP in JavaScript, here's the umap-js repo](https://github.com/PAIR-code/umap-js). This accompanying [Understanding UMAP](https://pair-code.github.io/understanding-umap/) article is also a fantastic read!
