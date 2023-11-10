# Transformers.js Examples

## Overview

Single page demonstration on running machine learning NLP models in the browser with transformers.js.

## Models Used

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

## About Transformers.js

Transformers.js allows for the direct execution of 🤗 transformer models in the browser with no server required. It is designed to be functionally equivalent to the 🤗 [Python transformers library](https://github.com/huggingface/transformers), supporting a range of tasks in NLP, computer vision, audio, and multimodal applications.

- Run pretrained models for tasks such as text summarization, classification, and generation, image classification, and speech recognition.
- Functionally similar to Hugging Face's Python library with the pipeline API.

For further information and a comprehensive list of supported tasks and models, refer to the [Transformers.js Documentation](https://huggingface.co/docs/transformers.js/index).

The [Transformers.js Model Hub](https://huggingface.co/models?library=transformers.js) lists models compatible with the library, searchable by tasks.
