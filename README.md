# AI Fine-Tuned Interview Assistant

A lightweight Generative AI project demonstrating supervised fine-tuning of a Large Language Model (LLM) using LoRA (Low-Rank Adaptation) and Hugging Face Transformers. This project fine-tunes TinyLlama on a custom AI interview dataset and deploys the model through a Streamlit-based web application for interactive AI question answering.

---

# Project Overview

This project focuses on practical implementation of parameter-efficient fine-tuning (PEFT) for Large Language Models. Instead of retraining all model parameters, LoRA adapters are used to reduce GPU memory usage, training cost, and storage requirements.

The project was trained on a custom instruction-response dataset containing AI, Machine Learning, NLP, Transformers, LoRA, QLoRA, and fine-tuning related interview questions.

The final application allows users to ask AI/LLM-related questions through a web interface and receive responses from the fine-tuned model.

---

# Features

- Fine-tuned TinyLlama using LoRA
- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- Custom instruction-response dataset
- Hugging Face Transformers integration
- Streamlit web application
- AI interview question answering
- Lightweight and beginner-friendly implementation
- GPU training using Google Colab
- Local inference pipeline

---

# Tech Stack

- Python
- Hugging Face Transformers
- PEFT (LoRA)
- TRL
- TinyLlama
- PyTorch
- Streamlit
- Google Colab

---

# Project Architecture

```text
User Question
       ↓
Streamlit Frontend
       ↓
Fine-Tuned TinyLlama Model
       ↓
LoRA Adapter Weights
       ↓
Generated AI Response
```

---

# Folder Structure

```text
ai-finetuning-interview-assistant/
│
├── dataset/
│   └── interview_data.json
│
├── inference/
│
├── models/
│   └── fine_tuned_model/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
│
├── app.py
├── finetune.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Understanding Fine-Tuning

Fine-tuning is the process of adapting a pretrained language model using task-specific or domain-specific data.

Instead of training a model from scratch, pretrained models already contain general language understanding learned from massive datasets. Fine-tuning improves the model’s ability to generate responses relevant to a particular use case.

In this project, TinyLlama was fine-tuned for AI and machine learning interview-related question answering.

---

# What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique used for large language models.

Traditional fine-tuning updates all model parameters, which requires large GPU memory and computational resources. LoRA solves this problem by training only small low-rank matrices while keeping most of the original model weights frozen.

Benefits of LoRA:
- Lower GPU memory usage
- Faster training
- Smaller model size
- Reduced computational cost
- Efficient deployment

---

# Model Used

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

# LoRA Configuration

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

# Dataset Format

The custom dataset was created in JSON instruction-response format.

Example:

```json
[
  {
    "instruction": "What is LoRA in AI fine-tuning?",
    "output": "LoRA stands for Low-Rank Adaptation and is a parameter-efficient fine-tuning technique for large language models."
  },
  {
    "instruction": "What is fine-tuning?",
    "output": "Fine-tuning is the process of adapting a pretrained model using task-specific data."
  }
]
```

---

# Fine-Tuning Workflow

1. Created a custom instruction-response dataset
2. Loaded TinyLlama tokenizer and model
3. Applied LoRA configuration using PEFT
4. Configured supervised fine-tuning settings
5. Trained model using GPU in Google Colab
6. Saved LoRA adapter weights
7. Integrated fine-tuned model into Streamlit frontend
8. Evaluated generated responses

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AadyaMishra2805/ai-finetuning-interview-assistant.git
cd ai-finetuning-interview-assistant
```

---

# Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

## Windows

```bash
venv\Scripts\activate
```

## Linux/Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit Application

```bash
streamlit run app.py
```

---

# Sample Questions

```text
What is LoRA in AI fine-tuning?
Explain supervised learning.
What are transformers?
What is parameter-efficient fine-tuning?
What is QLoRA?
```

---

# Challenges Faced

## GPU Runtime Issues

Initial Colab runtime did not detect GPU acceleration correctly. This was fixed by enabling GPU runtime manually.

---

## TorchAO Version Error

An incompatible torchao version caused PEFT loading failure. Restarting runtime and reinstalling packages resolved the issue.

---

## GitHub Large File Error

The virtual environment folder contained large PyTorch DLL files exceeding GitHub’s file size limit.

Solution:
- Added venv/ to .gitignore
- Reset Git tracking
- Reinitialized repository

---

## Local Inference Segmentation Fault

Running TinyLlama locally on CPU caused memory allocation issues on Windows systems.

Solution:
- Used low_cpu_mem_usage=True
- Forced CPU-safe inference pipeline

---

# Observations & Learnings

- Small datasets lead to weaker specialization
- Prompt formatting significantly affects output quality
- LoRA reduces training cost and GPU memory requirements
- Pretrained models retain original knowledge biases
- Instruction tuning improves domain-specific responses
- Ambiguous prompts can lead to hallucinated outputs

---

# Limitations

- Small dataset size
- Lightweight model capacity
- CPU inference can be slow
- Responses may occasionally hallucinate
- Limited domain coverage

---

# Future Improvements

- Larger and higher-quality dataset
- Better instruction formatting
- QLoRA optimization
- GPU deployment
- Hugging Face Spaces hosting
- Conversational memory
- Improved UI/UX
- Multi-turn conversation support

---

# Model Evaluation

The model was tested on multiple AI-related prompts.

Example prompts:
- What is LoRA in AI fine-tuning?
- Explain supervised learning.
- What are transformers?
- What is fine-tuning?

Evaluation focused on:
- Response relevance
- Domain specificity
- Instruction following
- Generated response quality

---

Author
Aadya Mishra
