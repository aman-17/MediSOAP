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

os.environ['HF_HOME'] = '/data1/aman/programs/'
os.environ["TRANSFORMERS_CACHE"] = '/data1/aman/programs/'
dataset = pd.read_csv('data/train_llama_formatted.csv')

# Load dataset
hf_dataset = Dataset.from_pandas(dataset)
hf_dataset = hf_dataset.rename_column("data", "text")
hf_dataset = hf_dataset.map(lambda examples: {'text': examples['text']})#, remove_columns=['__index_level_0__'])

# Load model and tokenizer
llama_base_model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf", cache_dir='/data1/aman/programs/')
llama_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", cache_dir='/data1/aman/programs/')
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Ensure tokenizer and model vocab sizes match
if llama_base_model.config.vocab_size != len(llama_tokenizer):
    llama_tokenizer.add_tokens([llama_tokenizer.unk_token] * (llama_base_model.config.vocab_size - len(llama_tokenizer)))
    llama_base_model.resize_token_embeddings(len(llama_tokenizer))

# Apply LoRA
phi_lora_model = ExtendedModel(llama_base_model)
freeze_model(phi_lora_model)

# Load dataset
dataset = pd.read_csv('data/train_llama_formatted.csv')

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "/data1/aman/programs/"
num_train_epochs = 10
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_8bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 10
max_seq_length = None
packing = False
device_map = {"": 0}

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()

# Training arguments
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
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

trainer = SFTTrainer(
    model=phi_lora_model,
    train_dataset=hf_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=llama_tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()