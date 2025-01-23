from utils import (
    load_model_and_tokenizer,
    make_dummy_dataset,
    print_gpu_usage,
    train_model,
    cleanup,
)

import gc
import torch
from transformers import TrainingArguments

# Helper function for GPU usage experiments
def gpu_memory_experiment(
        batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        model_id="EleutherAI/polyglot-ko-1.3b",
        peft=None
    ):
    print(f"batch_size: {batch_size}")
    
    # prepare model/tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, peft=peft)
    
    # Gradient Checkpoint option setup
    if gradient_checkpointing == True or peft == 'qlora':
        model.config.use_cache = False

    # prepare dummy dataset
    dataset = make_dummy_dataset()

    # prepare training argument
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        output_dir="./result",
        num_train_epochs=1
      )

    # model train start with model/dataset/training_args 
    try:
        train_model(model, dataset, training_args)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
        else:
            raise e
    finally:
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_usage()

cleanup()
print_gpu_usage()

gpu_memory_experiment(
    batch_size=4,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True, 
    peft='lora'
)
# batch_size: 4
# trainable params: 1,572,864 || all params: 1,333,383,168 || trainable%: 0.11796039111242178
# load_model_and_tokenizer Used GPU Memory : 2.578369617462158
# train_model Used GPU Memory : 3.035417079925537
# Optimizer Memory Usage: 0.006 GB
# Gradient Memory Usage: 0.003 GB
#  Used GPU Memory : 0.015869140625

### Why gradient checkpointing is not compatible with LoRA?
# Gradient checkpointing reduces memory usage by recomputing intermediate activations
# during the backward pass, but it’s not always compatible with modifications like LoRA, 
# which adds trainable parameters (low-rank updates) to certain layers. 
# The key issue is that when gradient checkpointing is used alongside LoRA, 
# some tensors might lose their requires_grad property, leading to this error.

# gpu_memory_experiment(
#     batch_size=16,
#     peft='lora'
# )
# batch_size: 16
# trainable params: 1,572,864 || all params: 1,333,383,168 || trainable%: 0.11796039111242178
# load_model_and_tokenizer Used GPU Memory : 2.578369617462158
# train_model Used GPU Memory : 4.333558559417725
# Optimizer Memory Usage: 0.006 GB
# Gradient Memory Usage: 0.003 GB
#  Used GPU Memory : 0.015869140625

# => model itself / optimizer / gradient memory all those three parts has decreased
# => Total memory decreased from 10.468698024749756 to 4.333558559417725