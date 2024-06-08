from transformers import AutoTokenizer, LlamaForCausalLM
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     TrainingArguments,
#     pipeline,
#     logging,
# )

from lora_scratch import (
    LinearLoRA,
    create_lora,
    add_lora_layers,
    freeze_model,
    unfreeze_model,
    create_linear,
    merge_lora_layers,
)

# llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

llama_base_model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# llama_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# llama_tokenizer.padding_side = "right"

add_lora_layers(llama_base_model, r=4, lora_alpha=8)  # inject the LoRA layers into the model
freeze_model(llama_base_model)

n_params = 0
n_trainable_params = 0

# count the number of trainable parameters
for n, p in llama_base_model.named_parameters():
    n_params += p.numel()
    if p.requires_grad:
        n_trainable_params += p.numel()

print(f"Total parameters: {n_params}")
print(f"Trainable parameters: {n_trainable_params}")
print(f"Percentage trainable: {round(n_trainable_params / n_params * 100, 2)}%")