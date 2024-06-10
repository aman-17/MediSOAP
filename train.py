from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset

import torch
from lora_scratch import LinearLoRA, freeze_model, ExtendedModel
from trl import SFTTrainer

llama_base_model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
llama_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

llama_lora_model = ExtendedModel(llama_base_model)

freeze_model(llama_lora_model)

dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "/data1/aman/programs/"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 1  
per_device_eval_batch_size = 1  
gradient_accumulation_steps = 4  
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
max_seq_length = None
packing = False
device_map = {"": 0}

dataset = load_dataset(dataset_name, split="train")
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf', cache_dir='/data1/aman/programs/', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    # report_to="tensorboard",
    gradient_checkpointing=gradient_checkpointing,
)

trainer = SFTTrainer(
    model=llama_lora_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
