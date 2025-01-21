# Keep trach GPU memory usage 
# Four most types of memory exists 
# Model/Gradient/Optimizer/

import gc
import torch
import numpy as np

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    AutoTokenizer, 
    AdamW,
)

# print gpu usage utilizing torch's API
def print_gpu_usage():
    # if device is GPU
    if torch.cuda.is_available():
        # get allocated memory and print result
        used_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Used GPU Memory: {used_memory}")
    else:
        print("Current device is CPU")

# memory cleanup helper function
def cleanup():
    if 'model' in globals():
        del globals()['model']
    if 'dataset' in globals():
        del globals()['dataset']
    gc.collect()
    torch.cuda.empty_cache()

# Model and Tokenizer Memory Tracking
def load_model_and_tokenizer(model_id, peft=None):
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # define model\
    model = None
    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})

    # print gpu usage with helper func
    print_gpu_usage()

    # return model & tokenizer
    return model, tokenizer

# Gradient memory tracking
def estimate_memory_of_gradients(model):
    total_memory = 0

    # traverse all parameters in model
    # Sum nelement * element_size all through traversing
    for param in model.parameters():
        if param.grad is not None:
            total_memory += param.grad.nelement() * param.grad.element_size()

    return total_memory

# Optimizer memory tracking
def estimate_memory_of_optimizer(optimizer):
    total_memory = 0

    # traverse all parameters in optimizer
    # Sum nelement * element_size of all values in optimizer
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total_memory += v.nelement() * v.element_size()

    return total_memory

# Memory usage tracking during training
def train_model(model, dataset, training_args):
    # Option1. Gradient Checkpoint
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare Dataloader/Optimizer and then switch model mode into train mode 
    train_dataloader = DataLoader(
        dataset, 
        batch_size=training_args.per_device_train_batch_size
    )
    optimizer = AdamW(model.parameters())
    model.train()

    # first time printing toggle
    gpu_utilization_printed = False
    
    # traverse DataLoader
    for step, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Back propagation
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        # Option2. Gradient Accumulation
        if step % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            gradients_memory = estimate_memory_of_gradients(model)
            optimizer_memory = estimate_memory_of_optimizer(optimizer)
            if not gpu_utilization_printed:
                print_gpu_usage()
                gpu_utilization_printed = True
            optimizer.zero_grad()

    print(f"Optimizer Memory Usage: {optimizer_memory / (1024 ** 3):.3f} GB")
    print(f"Gradient Memory Usage: {gradients_memory / (1024 ** 3):.3f} GB")

# Create Dummy Dataset for test
def make_dummy_dataset():
    seq_len, dataset_size = 256, 64
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(100, 30000, (dataset_size, seq_len)),
    }
    dataset = Dataset.from_dict(dummy_data)
    dataset.set_format("pt")
    return dataset

# learning_rate = 5e-5
# model_id = "EleutherAI/polyglot-ko-1.3b"
# model, tokenizer = load_model_and_tokenizer(model_id) # GPU 메모리 사용량: 2.599 GB
# optimizer = AdamW(model.parameters(), lr=learning_rate)

# estimate_memory_of_gradients(model)
# estimate_memory_of_optimizer(optimizer)

# train_model()