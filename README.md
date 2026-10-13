# Llama 3.1 8B — Medical Q&A Fine-Tune (Unsloth QLoRA) 🩺

A Colab notebook that fine-tunes **Meta Llama 3.1 8B** on a medical question-and-answer dataset with **Unsloth QLoRA**, then exports the model to a 4-bit **GGUF** file so it can run locally.

> Built during my Data Science internship at Analyzinn Solutions as a prototype. It isn't a production service.

## 🔧 What the notebook does

1. **Loads the base model in 4-bit** (`unsloth/Meta-Llama-3.1-8B`, max sequence length 2048) to fit on a free Colab T4 GPU.
2. **Adds LoRA adapters** (r = 16, alpha = 16) to the attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`), using Unsloth gradient checkpointing.
3. **Formats the data** into an Instruction / Input / Response prompt template with an EOS token.
4. **Trains** with TRL's `SFTTrainer` (batch size 2, gradient accumulation 4).
5. **Tests inference** on medical drug-interaction questions.
6. **Exports** the model to GGUF (`q4_k_m`) and saves it to Google Drive for llama.cpp or local chatbot use.

## 💻 Stack

Python · Unsloth · PyTorch · Hugging Face Transformers · TRL · PEFT (LoRA) · bitsandbytes · llama.cpp / GGUF · Google Colab

## 🚀 Run

Open `llama3_1_medical_qa_unsloth.ipynb` in Google Colab, select a **T4 GPU** runtime and run the cells in order.
