import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLoRA(nn.Module):    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.,    
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) 
        
        # Check that the rank is at least 1
        assert r > 0, "Variable 'r' is not greater than zero. Choose a rank of 1 or greater."
            
        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the low-rank A matrix and initialize with same method as in Hugging Face PEFT library
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # create the low-rank B matrix and initialize to zero 
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight, 0)

        # scaling constant
        self.scaling = self.lora_alpha / self.r
                        
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling        
        return pretrained_out + lora_out
    
    
def freeze_model(model):
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

            
def create_lora(module, r, lora_dropout, lora_alpha):
    k, d = module.weight.shape 
    lora = LinearLoRA(d, k, r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias)        
    return lora
                

def add_lora_layers(
    model,  
    module_names: Tuple=("query", "value"), 
    r: int=8, 
    lora_alpha: float=16,
    lora_dropout: float=0.1, 
    ignore_layers: List[int]=[]  
):                  
    module_types: Tuple=(nn.Linear,)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for name, module in model.named_modules():  # Change this line
        print(f"Checking layer {name} of type {type(module)}")  # Add this line
        if isinstance(module, module_types) and any(mod_name in name for mod_name in module_names):
            temp_lora = create_lora(module, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            parent_module._modules[child_name] = temp_lora
        else:
            ignore_layers_str = [str(i) for i in ignore_layers]
            if name not in ignore_layers_str:
                add_lora_layers(module, module_names, r, lora_dropout, lora_alpha, ignore_layers)


def unfreeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def create_linear(module):
    k, d = module.pretrained.weight.shape
    linear = nn.Linear(d, k, bias=True)
    with torch.no_grad():
        linear.weight.copy_(module.pretrained.weight + (module.lora_B.weight @ module.lora_A.weight) * module.scaling)
        linear.bias.copy_(module.pretrained.bias)
    return linear


def merge_lora_layers(model, module_names: Tuple=("query", "value"), dropout=0.1):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    for name, module in model.named_children():
        if name in module_names and hasattr(module, "pretrained"):
            temp_linear = create_linear(module)
            setattr(model, name, temp_linear)                  
        else:
            merge_lora_layers(module, module_names=module_names, dropout=0.1)


class ExtendedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.lora = LinearLoRA(base_model.config.hidden_size, base_model.config.hidden_size, r=8, lora_alpha=16, lora_dropout=0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lora_output = self.lora(last_hidden_state)
        return lora_output

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.lora.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)