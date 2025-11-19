# Mental Health Chatbot 🧠💬

A fine-tuned LLM-based chatbot for mental health counseling, built with Llama-3.1-8B and deployed on Modal for production inference.

## Overview

This project fine-tunes Llama-3.1-8B on 16K mental health conversations to create an empathetic, safe, and helpful counseling assistant. The model achieved a **7% improvement in empathy scores** while maintaining safety and helpfulness standards.

## Tech Stack

- **Model**: Llama-3.1-8B (8B parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) on Modal H100 GPUs
- **Inference**: vLLM on Modal A100 GPUs
- **Dataset**: ShenLab/MentalChat16K
- **Framework**: Unsloth, Transformers, TRL

## Key Features

- ✅ **Parameter-efficient fine-tuning** using LoRA (r=16)
- ✅ **Production-ready API** with async batch processing (32 concurrent requests)
- ✅ **Cost-optimized deployment** (H100 for training, A100 for serving)
- ✅ **Comprehensive evaluation** using custom metrics (empathy, safety, helpfulness)

## Project Structure
```
├── fine_tune.py           # LoRA fine-tuning script
├── deploy_model.py        # vLLM inference server deployment
├── batch_inference.ipynb  # Async batch inference for evaluation
├── metrics.py             # Model evaluation pipeline
└── template_alpaca.jinja  # Prompt formatting template
```

## Results

| Metric | Fine-tuned Model | Baseline |
|--------|-----------------|----------|
| Empathy Score | 4.952 | 4.620 |
| Safety Score | 4.797 | 4.882 |
| Helpfulness | 4.910 | 4.814 |

## Setup & Usage

1. **Fine-tune the model**:
```bash
modal run fine_tune.py
```

2. **Deploy inference API**:
```bash
modal deploy deploy_model.py
```

3. **Run evaluation**:
```bash
jupyter notebook batch_inference.ipynb
```

## API Usage
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://your-modal-endpoint.modal.run/v1",
    api_key="your-secret-key"
)

response = await client.chat.completions.create(
    model="unsloth/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "I'm feeling anxious about work..."}]
)
```

**Note**: This chatbot is for educational purposes. It should not replace professional mental health services.
