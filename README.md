AI Fine-Tuned Interview Assistant

A lightweight Generative AI project that demonstrates supervised fine-tuning of a Large Language Model (LLM) using LoRA (Low-Rank Adaptation) and Hugging Face Transformers. The project fine-tunes TinyLlama on a custom AI interview dataset and deploys the model through a Streamlit web application for interactive question answering.

Project Overview

This project focuses on practical LLM fine-tuning workflows using parameter-efficient training techniques. Instead of retraining an entire language model, LoRA adapters are applied to significantly reduce GPU memory usage and training cost.

The system was trained on a custom instruction-response dataset related to AI, machine learning, LLMs, LoRA, transformers, and fine-tuning concepts.

The final application allows users to ask AI/LLM-related questions through a Streamlit interface and receive generated responses from the fine-tuned model.

Features
Fine-tuned TinyLlama using LoRA
Supervised Fine-Tuning (SFT) workflow
Custom instruction-response dataset
Hugging Face Transformers integration
PEFT (Parameter Efficient Fine-Tuning)
Streamlit-based interactive UI
AI interview question answering
Lightweight and beginner-friendly implementation
Local inference pipeline
GPU training using Google Colab
Tech Stack
Python
Hugging Face Transformers
PEFT (LoRA)
TRL
TinyLlama
PyTorch
Streamlit
Google Colab
Project Architecture
User Question
       ↓
Streamlit Frontend
       ↓
Fine-Tuned TinyLlama Model
       ↓
LoRA Adapter Weights
       ↓
Generated AI Response
Folder Structure
ai-finetuning-interview-assistant/
│
├── dataset/
│   └── interview_data.json
│
├── models/
│   └── fine_tuned_model/
│
├── app.py
├── finetune.py
├── requirements.txt
├── .gitignore
└── README.md
Fine-Tuning Workflow
Prepared a custom instruction-response dataset
Loaded TinyLlama pretrained model
Applied LoRA configuration using PEFT
Performed supervised fine-tuning
Trained model on GPU using Google Colab
Saved LoRA adapter weights
Integrated fine-tuned model into Streamlit app
Evaluated generated responses
Model Used
TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA Configuration
r=8
lora_alpha=16
lora_dropout=0.05
target_modules=["q_proj", "v_proj"]
Installation
Clone Repository
git clone https://github.com/YOUR_USERNAME/ai-finetuning-interview-assistant.git
cd ai-finetuning-interview-assistant
Create Virtual Environment
python -m venv venv

Activate environment:

Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Run Streamlit Application
streamlit run app.py
Sample Questions
What is LoRA in AI fine-tuning?
Explain supervised fine-tuning.
What are transformers in deep learning?
What is parameter-efficient fine-tuning?
Observations & Learnings
Small datasets lead to weaker specialization
Prompt formatting significantly affects output quality
LoRA reduces training cost and GPU memory requirements
Pretrained models retain original knowledge biases
Instruction tuning improves domain-specific responses
Limitations
Small custom dataset
Lightweight model capacity
CPU inference can be slow
Responses may occasionally hallucinate
Limited domain coverage
Future Improvements
Larger high-quality dataset
Better instruction formatting
QLoRA optimization
GPU deployment
Hugging Face Spaces deployment
Chat-style conversational memory
Improved UI/UX
Resume Description
Built and fine-tuned a lightweight Large Language Model using TinyLlama and LoRA with Hugging Face Transformers. Implemented supervised fine-tuning on a custom instruction dataset, developed a Streamlit-based AI interview assistant, and evaluated model performance and response quality.
Interview Explanation

This project demonstrates practical implementation of parameter-efficient LLM fine-tuning using LoRA. TinyLlama was fine-tuned on a custom AI interview dataset using Hugging Face PEFT and TRL libraries. The project also includes inference handling, Streamlit deployment, prompt formatting, and evaluation of fine-tuned responses against pretrained model behavior.

Author
Aadya Mishra
