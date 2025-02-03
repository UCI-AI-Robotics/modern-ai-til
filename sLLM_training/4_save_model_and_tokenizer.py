# Save fine-tuned model and upload to Hugging Face Hub

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

base_model = 'beomi/Yi-Ko-6B'
finetuned_model = 'yi-ko-6b-text2sql'

model_name = base_model
device_map = {"": 0}  # Assign the model to GPU 0

# Merge LoRA and base model parameters
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,  # Optimize memory usage while loading the model
    return_dict=True,  # Return output as a dictionary
    torch_dtype=torch.float16,  # Use 16-bit floating point precision
    device_map=device_map,  # Assign the model to the correct device
)
model = PeftModel.from_pretrained(base_model, finetuned_model)
model = model.merge_and_unload()  # Merge LoRA parameters and unload unused weights

# Set up the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to the EOS token
tokenizer.padding_side = "right"  # Set padding to the right side

# Save the model and tokenizer to the Hugging Face Hub
model.push_to_hub(finetuned_model, use_temp_dir=False)
tokenizer.push_to_hub(finetuned_model, use_temp_dir=False)
