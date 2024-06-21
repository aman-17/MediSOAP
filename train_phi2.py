import math
from typing import List, Tuple
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
import os
import pandas as pd
import numpy as np
from lora_scratch import LinearLoRA, freeze_model, ExtendedModel
from trl import SFTTrainer
from peft import LoraConfig, PeftModel

import logging
logging.basicConfig(level=logging.INFO)

# Set environment variables if needed
os.environ['HF_HOME'] = '/data1/aman/programs/'
os.environ["TRANSFORMERS_CACHE"] = '/data1/aman/programs/'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Load dataset
dataset = pd.read_csv('data/train_llama_formatted.csv')
hf_dataset = Dataset.from_pandas(dataset)
hf_dataset = hf_dataset.rename_column("data", "text")
hf_dataset = hf_dataset.map(lambda examples: {'text': examples['text']})

# Load model and tokenizer
phi_base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", trust_remote_code=True, cache_dir='/data1/aman/programs/')
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, cache_dir='/data1/aman/programs/')
phi_tokenizer.pad_token = phi_tokenizer.eos_token
phi_tokenizer.padding_side = "right"

# Ensure tokenizer and model vocab sizes match
if phi_base_model.config.vocab_size != len(phi_tokenizer):
    phi_tokenizer.add_tokens([phi_tokenizer.unk_token] * (phi_base_model.config.vocab_size - len(phi_tokenizer)))
    phi_base_model.resize_token_embeddings(len(phi_tokenizer))

# Apply LoRA
phi_lora_model = ExtendedModel(phi_base_model)
freeze_model(phi_lora_model)

# Configuration for BitsAndBytes (if applicable)
# use_4bit = True
# bnb_4bit_compute_dtype = "bfloat16"
# bnb_4bit_quant_type = "nf4"
# use_nested_quant = False

# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=use_nested_quant,
# )

# Training arguments
output_dir = "/data1/aman/programs/"
num_train_epochs = 10
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25


training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=False,
    bf16=False,
    max_grad_norm=max_grad_norm,
    lr_scheduler_type=lr_scheduler_type
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=phi_lora_model,
    train_dataset=hf_dataset,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=phi_tokenizer,
    args=training_arguments,
    packing=False,
)

# Start training
# trainer.train()
for epoch in range(num_train_epochs):
    trainer.train()
    train_metrics = trainer.evaluate()
    logging.info(f"Epoch {epoch + 1}/{num_train_epochs}, Train Loss: {train_metrics['loss']:.4f}")

