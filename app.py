import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

st.set_page_config(
    page_title="AI Fine-Tuned Interview Assistant",
    layout="centered"
)

st.title("AI Fine-Tuned Interview Assistant")

# Base model
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "models/fine_tuned_model"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

question = st.text_input("Ask AI/LLM Question")

if st.button("Generate Answer"):

    prompt = f"Instruction: {question}\nResponse:"

    result = pipe(
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    generated_text = result[0]["generated_text"]

    answer = generated_text.replace(prompt, "").strip()

    st.write(answer)