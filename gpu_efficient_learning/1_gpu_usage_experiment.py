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

for batch_size in [4, 8, 16]:
    gpu_memory_experiment(batch_size)
    torch.cuda.empty_cache()

# ######## Result ########
### batch_size: 4
# load_model_and_tokenizer Used GPU Memory : 2.575439929962158
# train_model Used GPU Memory : 10.468698024749756
# Optimizer Memory Usage: 4.961 GB
# Gradient Memory Usage: 2.481 GB
#  Used GPU Memory : 0.015869140625
### batch_size: 8
# load_model_and_tokenizer Used GPU Memory : 2.591309070587158
# train_model Used GPU Memory : 10.901574611663818
# Optimizer Memory Usage: 4.961 GB
# Gradient Memory Usage: 2.481 GB
#  Used GPU Memory : 0.015869140625
### batch_size: 16
# load_model_and_tokenizer Used GPU Memory : 2.591309070587158
# train_model Used GPU Memory : 11.765862941741943
# Optimizer Memory Usage: 4.961 GB
# Gradient Memory Usage: 2.481 GB
#  Used GPU Memory : 0.015869140625

# => model / gradient / optimizer memory would be same 
# => but total memory usage (which is train_model Used GPU Memory) increases
# => this means feed forward calculation memory increasing